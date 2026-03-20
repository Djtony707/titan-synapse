use anyhow::Result;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::config::SynapseConfig;
use super::model::LoadedModel;
use super::lora::LoraAdapter;

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
}

impl InferenceEngine {
    pub fn new(config: &SynapseConfig) -> Result<Self> {
        let mut engine = Self {
            models: HashMap::new(),
            adapters: HashMap::new(),
            models_dir: config.models_dir.clone(),
            adapters_dir: config.adapters_dir.clone(),
        };

        // Scan for available adapters
        engine.scan_adapters()?;

        tracing::info!(
            "Inference engine initialized. Models dir: {}, Adapters: {}",
            config.models_dir.display(),
            engine.adapters.len()
        );

        Ok(engine)
    }

    /// Generate text from a prompt using a specific specialist
    pub async fn generate(
        &self,
        prompt: &str,
        specialist: Option<&str>,
        max_tokens: u32,
        temperature: f32,
    ) -> Result<String> {
        // For now, return a placeholder that indicates the engine is working
        // This will be replaced with actual candle inference
        let specialist_name = specialist.unwrap_or("general");

        tracing::debug!(
            "Generating with specialist={specialist_name}, max_tokens={max_tokens}, temp={temperature}"
        );

        // Check if we have the model loaded
        if self.models.is_empty() {
            return Ok(format!(
                "TITAN Synapse engine is running but no models are loaded yet. \
                 Use `synapse pull qwen3-3b` to download a model. \
                 Specialist '{specialist_name}' was selected for this query."
            ));
        }

        // TODO: Real inference with candle
        // 1. Get or load base model
        // 2. Apply LoRA adapter for specialist
        // 3. Tokenize prompt
        // 4. Run forward pass
        // 5. Sample tokens
        // 6. Decode and return
        Ok(format!("[{specialist_name}] Inference placeholder — model loading coming next"))
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
        let response = self.generate(prompt, specialist, max_tokens, temperature).await?;

        tokio::spawn(async move {
            for word in response.split_inclusive(' ') {
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
    pub async fn swap_adapter(&mut self, specialist: &str, adapter_path: &str) -> Result<()> {
        tracing::info!("Hot-swapping adapter for specialist '{specialist}': {adapter_path}");
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
}
