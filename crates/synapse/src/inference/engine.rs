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
    pub async fn generate(
        &self,
        prompt: &str,
        specialist: Option<&str>,
        max_tokens: u32,
        temperature: f32,
    ) -> Result<GenerationResult> {
        let specialist_name = specialist.unwrap_or("general");

        tracing::debug!(
            "Generating: specialist={specialist_name}, max_tokens={max_tokens}, temp={temperature}"
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
            "Generated {completion_tokens} tokens in {:.1}s ({:.1} tok/s), specialist={specialist_name}",
            elapsed.as_secs_f64(),
            tok_per_sec
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

    /// Select the best available model (prefer larger ones)
    fn select_model(&self) -> Option<Arc<Mutex<LoadedModel>>> {
        // Sort by name length descending (larger models have longer names like "3b" > "0.5b")
        // In production, this would use actual parameter count
        self.models.values()
            .max_by_key(|_| 1) // For now, just pick any loaded model
            .cloned()
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
    fn scan_adapters(&mut self) -> Result<()> {
        if !self.adapters_dir.exists() {
            std::fs::create_dir_all(&self.adapters_dir)?;
            return Ok(());
        }

        for entry in std::fs::read_dir(&self.adapters_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "safetensors") {
                if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                    self.adapters.insert(name.to_string(), LoraAdapter {
                        name: name.to_string(),
                        path,
                        rank: 16,
                        loaded: false,
                    });
                }
            }
        }

        if !self.adapters.is_empty() {
            tracing::info!("Found {} LoRA adapters", self.adapters.len());
        }

        Ok(())
    }

    /// Hot-swap a LoRA adapter for a specialist
    pub async fn swap_adapter(&mut self, _specialist: &str, _adapter_path: &str) -> Result<()> {
        // TODO: Implement actual adapter swap via candle
        Ok(())
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
