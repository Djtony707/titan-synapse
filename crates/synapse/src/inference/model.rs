use anyhow::Result;
use candle_core::{Device, IndexOp, Tensor};
use candle_core::quantized::gguf_file;
use candle_transformers::models::quantized_qwen2::ModelWeights;
use tokenizers::Tokenizer;
use std::path::PathBuf;

use super::sampler::SamplerConfig;

/// Represents a loaded quantized model in memory
pub struct LoadedModel {
    pub name: String,
    pub path: PathBuf,
    pub model: ModelWeights,
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub eos_token_id: u32,
}

impl LoadedModel {
    /// Load a GGUF model + tokenizer from disk
    pub fn load(name: &str, model_path: &PathBuf, tokenizer_path: &PathBuf, device: &Device) -> Result<Self> {
        tracing::info!("Loading model '{name}' from {}", model_path.display());
        let start = std::time::Instant::now();

        // Load GGUF
        let mut file = std::fs::File::open(model_path)?;
        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| anyhow::anyhow!("Failed to read GGUF: {e}"))?;

        let model = ModelWeights::from_gguf(content, &mut file, device)
            .map_err(|e| anyhow::anyhow!("Failed to load model weights: {e}"))?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

        // Find EOS token
        let eos_token_id = tokenizer.token_to_id("<|endoftext|>")
            .or_else(|| tokenizer.token_to_id("<|im_end|>"))
            .or_else(|| tokenizer.token_to_id("</s>"))
            .unwrap_or(2);

        tracing::info!(
            "Model '{name}' loaded in {:.1}s (eos_token_id={eos_token_id})",
            start.elapsed().as_secs_f32()
        );

        Ok(Self {
            name: name.to_string(),
            path: model_path.clone(),
            model,
            tokenizer,
            device: device.clone(),
            eos_token_id,
        })
    }

    /// Generate text from a prompt
    pub fn generate(&mut self, prompt: &str, max_tokens: u32, sampler: &SamplerConfig) -> Result<String> {
        // Tokenize
        let encoding = self.tokenizer.encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("Tokenize error: {e}"))?;
        let mut tokens: Vec<u32> = encoding.get_ids().to_vec();
        let prompt_len = tokens.len();

        tracing::debug!("Prompt: {} tokens, generating up to {max_tokens}", prompt_len);

        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut pos = 0;

        // Process prompt (prefill)
        let input = Tensor::new(tokens.as_slice(), &self.device)?
            .unsqueeze(0)?; // (1, seq_len)
        let logits = self.model.forward(&input, pos)?;
        pos += tokens.len();

        // Get logits for last position
        let logits = logits.squeeze(0)?; // (seq_len, vocab)
        let last_logits = logits.i(logits.dim(0)? - 1)?; // (vocab,)
        let logits_vec: Vec<f32> = last_logits.to_vec1()?;

        // Sample first token
        let next_token = sampler.sample(&logits_vec);
        generated_tokens.push(next_token);

        if next_token == self.eos_token_id {
            return self.decode_tokens(&generated_tokens);
        }

        // Autoregressive generation
        for _ in 1..max_tokens {
            let input = Tensor::new(&[next_token], &self.device)?
                .unsqueeze(0)?;
            let logits = self.model.forward(&input, pos)?;
            pos += 1;

            let logits = logits.squeeze(0)?;
            let last_logits = logits.i(logits.dim(0)? - 1)?;
            let logits_vec: Vec<f32> = last_logits.to_vec1()?;

            let next_token = sampler.sample(&logits_vec);

            if next_token == self.eos_token_id {
                break;
            }

            generated_tokens.push(next_token);
        }

        self.decode_tokens(&generated_tokens)
    }

    fn decode_tokens(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(tokens, true)
            .map_err(|e| anyhow::anyhow!("Decode error: {e}"))
    }
}
