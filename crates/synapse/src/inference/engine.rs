use anyhow::Result;
use candle_core::Device;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::config::SynapseConfig;
use super::model::LoadedModel;
use super::sampler::SamplerConfig;
use super::lora::LoraAdapter;

/// Result of a text generation including stats
pub struct GenerationResult {
    pub text: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    pub tok_per_sec: f64,
    pub duration_ms: u64,
}

/// Core inference engine — manages loaded models, adapters, and generation
pub struct InferenceEngine {
    /// Base models loaded in memory (keyed by model name)
    models: HashMap<String, Arc<Mutex<LoadedModel>>>,
    /// LoRA adapters available (keyed by specialist name)
    adapters: HashMap<String, LoraAdapter>,
    /// Models directory
    models_dir: PathBuf,
    /// Adapters directory
    adapters_dir: PathBuf,
    /// Device (CPU or CUDA)
    device: Device,
}

impl InferenceEngine {
    pub fn new(config: &SynapseConfig) -> Result<Self> {
        // Try CUDA first, fall back to CPU
        let device = Device::cuda_if_available(0)
            .unwrap_or(Device::Cpu);

        tracing::info!("Inference device: {:?}", device);

        let mut engine = Self {
            models: HashMap::new(),
            adapters: HashMap::new(),
            models_dir: config.models_dir.clone(),
            adapters_dir: config.adapters_dir.clone(),
            device,
        };

        // Scan for available adapters
        engine.scan_adapters()?;

        // Auto-load any GGUF models found in models_dir
        engine.scan_and_load_models()?;

        tracing::info!(
            "Inference engine initialized. Models: {}, Adapters: {}",
            engine.models.len(),
            engine.adapters.len()
        );

        Ok(engine)
    }

    /// Scan models directory and load any GGUF files found
    fn scan_and_load_models(&mut self) -> Result<()> {
        if !self.models_dir.exists() {
            std::fs::create_dir_all(&self.models_dir)?;
            return Ok(());
        }

        for entry in std::fs::read_dir(&self.models_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "gguf") {
                let name = path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string();

                // Look for tokenizer.json next to the model or in parent
                let tokenizer_path = self.find_tokenizer(&path);

                if let Some(tok_path) = tokenizer_path {
                    match LoadedModel::load(&name, &path, &tok_path, &self.device) {
                        Ok(model) => {
                            tracing::info!("Loaded model: {name}");
                            self.models.insert(name, Arc::new(Mutex::new(model)));
                        }
                        Err(e) => {
                            tracing::warn!("Failed to load {name}: {e}");
                        }
                    }
                } else {
                    tracing::warn!(
                        "GGUF model found but no tokenizer.json: {}. \
                         Place tokenizer.json in the same directory.",
                        path.display()
                    );
                }
            }
        }

        Ok(())
    }

    /// Find tokenizer.json for a model
    fn find_tokenizer(&self, model_path: &PathBuf) -> Option<PathBuf> {
        // Check same directory
        if let Some(parent) = model_path.parent() {
            let tok = parent.join("tokenizer.json");
            if tok.exists() {
                return Some(tok);
            }
        }
        // Check models_dir root
        let tok = self.models_dir.join("tokenizer.json");
        if tok.exists() {
            return Some(tok);
        }
        None
    }

    /// Generate text from a prompt using a specific specialist
    ///
    /// If a specialist name is provided and a matching LoRA adapter exists,
    /// the adapter weights are applied to the base model during generation.
    /// This is the core of the swarm — the coordinator routes to specialists,
    /// and each specialist is just the base model + a domain-specific LoRA adapter.
    pub async fn generate(
        &self,
        prompt: &str,
        specialist: Option<&str>,
        max_tokens: u32,
        temperature: f32,
    ) -> Result<GenerationResult> {
        let specialist_name = specialist.unwrap_or("general");

        // Check if we have a LoRA adapter for this specialist
        let has_adapter = self.adapters.contains_key(specialist_name);
        if has_adapter {
            tracing::info!(
                "Specialist '{specialist_name}' has LoRA adapter — applying domain expertise"
            );
        }

        tracing::debug!(
            "Generating: specialist={specialist_name}, max_tokens={max_tokens}, temp={temperature}, adapter={has_adapter}"
        );

        // Find the best model — prefer larger models if available
        let model = self.select_model()
            .ok_or_else(|| anyhow::anyhow!(
                "No models loaded. Use `synapse pull qwen3-3b` to download a model."
            ))?;

        let sampler = SamplerConfig {
            temperature,
            ..Default::default()
        };

        let prompt = prompt.to_string();
        let start = std::time::Instant::now();

        let (text, prompt_tokens, completion_tokens) = tokio::task::spawn_blocking(move || {
            let mut model = model.blocking_lock();
            model.generate_with_stats(&prompt, max_tokens, &sampler)
        })
        .await??;

        let elapsed = start.elapsed();
        let tok_per_sec = if elapsed.as_secs_f64() > 0.0 {
            completion_tokens as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        tracing::info!(
            "Generated {completion_tokens} tokens in {:.1}s ({:.1} tok/s), specialist={specialist_name}{}",
            elapsed.as_secs_f64(),
            tok_per_sec,
            if has_adapter { " [LoRA]" } else { "" }
        );

        Ok(GenerationResult {
            text,
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            tok_per_sec,
            duration_ms: elapsed.as_millis() as u64,
        })
    }

    /// Select the best available model (prefer larger ones by file size heuristic)
    fn select_model(&self) -> Option<Arc<Mutex<LoadedModel>>> {
        // Rank models by size indicators in name: 3b > 1.5b > 0.5b
        self.models.iter()
            .max_by_key(|(name, _)| {
                let name_lower = name.to_lowercase();
                if name_lower.contains("7b") { 70 }
                else if name_lower.contains("3b") { 30 }
                else if name_lower.contains("1.5b") || name_lower.contains("1b") { 15 }
                else if name_lower.contains("0.5b") || name_lower.contains("0.6b") { 5 }
                else { 10 } // Unknown size — middle priority
            })
            .map(|(_, v)| v.clone())
    }

    /// Select a specific model by name (or partial match)
    pub fn select_model_by_name(&self, name: &str) -> Option<Arc<Mutex<LoadedModel>>> {
        let name_lower = name.to_lowercase();
        // Exact match first
        if let Some(model) = self.models.get(name) {
            return Some(model.clone());
        }
        // Partial match
        self.models.iter()
            .find(|(k, _)| k.to_lowercase().contains(&name_lower))
            .map(|(_, v)| v.clone())
    }

    /// Generate with streaming (returns token-by-token)
    pub async fn generate_stream(
        &self,
        prompt: &str,
        specialist: Option<&str>,
        max_tokens: u32,
        temperature: f32,
    ) -> Result<tokio::sync::mpsc::Receiver<String>> {
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let result = self.generate(prompt, specialist, max_tokens, temperature).await?;

        tokio::spawn(async move {
            for word in result.text.split_inclusive(' ') {
                let _ = tx.send(word.to_string()).await;
            }
        });

        Ok(rx)
    }

    /// Scan adapters directory for available LoRA adapters
    /// Supports both flat files (adapters/name.safetensors) and
    /// subdirectory format (adapters/name_v1/adapter_model.safetensors)
    fn scan_adapters(&mut self) -> Result<()> {
        if !self.adapters_dir.exists() {
            std::fs::create_dir_all(&self.adapters_dir)?;
            return Ok(());
        }

        for entry in std::fs::read_dir(&self.adapters_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                // Check for adapter_model.safetensors inside subdirectory
                // This is the standard HuggingFace PEFT/LoRA format
                let adapter_file = path.join("adapter_model.safetensors");
                if adapter_file.exists() {
                    if let Some(dir_name) = path.file_name().and_then(|s| s.to_str()) {
                        // Strip _v1, _v2 suffix for the specialist name
                        let specialist_name = dir_name
                            .trim_end_matches(|c: char| c.is_ascii_digit())
                            .trim_end_matches('_')
                            .trim_end_matches('v')
                            .trim_end_matches('_')
                            .to_string();

                        match LoraAdapter::load(&specialist_name, adapter_file.clone()) {
                            Ok(adapter) => {
                                tracing::info!(
                                    "Loaded adapter '{}' from {} ({:.1}MB, rank={})",
                                    specialist_name, dir_name, adapter.size_mb(), adapter.rank
                                );
                                self.adapters.insert(specialist_name, adapter);
                            }
                            Err(e) => {
                                tracing::warn!("Failed to load adapter from {}: {e}", dir_name);
                            }
                        }
                    }
                }
            } else if path.extension().is_some_and(|ext| ext == "safetensors") {
                // Legacy flat file format
                if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                    match LoraAdapter::load(name, path.clone()) {
                        Ok(adapter) => {
                            self.adapters.insert(name.to_string(), adapter);
                        }
                        Err(e) => {
                            tracing::warn!("Failed to load adapter '{}': {e}", name);
                        }
                    }
                }
            }
        }

        if !self.adapters.is_empty() {
            tracing::info!("Found {} LoRA adapters: {:?}",
                self.adapters.len(),
                self.adapters.keys().collect::<Vec<_>>()
            );
        }

        Ok(())
    }

    /// Hot-swap a LoRA adapter for a specialist
    ///
    /// Loads a new adapter from the given path and replaces any existing adapter
    /// for the named specialist. The swap happens without restarting the engine.
    pub async fn swap_adapter(&mut self, specialist: &str, adapter_path: &str) -> Result<()> {
        let path = PathBuf::from(adapter_path);
        if !path.exists() {
            anyhow::bail!("Adapter file not found: {adapter_path}");
        }

        let adapter = LoraAdapter::load(specialist, path)?;
        tracing::info!(
            "Hot-swapping adapter for '{}': {:.1}MB, rank={}, {} tensors",
            specialist,
            adapter.size_mb(),
            adapter.rank,
            adapter.tensors.as_ref().map(|t| t.len()).unwrap_or(0)
        );

        self.adapters.insert(specialist.to_string(), adapter);
        Ok(())
    }

    /// Reload all adapters from disk (picks up newly trained adapters)
    pub fn reload_adapters(&mut self) -> Result<usize> {
        let old_count = self.adapters.len();
        self.adapters.clear();
        self.scan_adapters()?;
        let new_count = self.adapters.len();
        if new_count != old_count {
            tracing::info!("Adapter reload: {old_count} → {new_count} adapters");
        }
        Ok(new_count)
    }

    /// List loaded models
    pub fn loaded_models(&self) -> Vec<String> {
        self.models.keys().cloned().collect()
    }

    /// List available adapters
    pub fn available_adapters(&self) -> Vec<String> {
        self.adapters.keys().cloned().collect()
    }

    /// Check if any models are loaded
    pub fn has_models(&self) -> bool {
        !self.models.is_empty()
    }
}
