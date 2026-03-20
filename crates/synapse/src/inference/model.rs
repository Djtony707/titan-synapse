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
    /// Additional stop token IDs (im_start, im_end, etc.)
    pub stop_token_ids: Vec<u32>,
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

        // Find EOS and stop tokens
        let eos_token_id = tokenizer.token_to_id("<|endoftext|>")
            .or_else(|| tokenizer.token_to_id("<|im_end|>"))
            .or_else(|| tokenizer.token_to_id("</s>"))
            .unwrap_or(2);

        // Collect all tokens that should stop generation
        let stop_candidates = ["<|im_end|>", "<|im_start|>", "<|endoftext|>", "</s>", "<|end|>"];
        let stop_token_ids: Vec<u32> = stop_candidates.iter()
            .filter_map(|tok| tokenizer.token_to_id(tok))
            .collect();

        tracing::info!(
            "Model '{name}' loaded in {:.1}s (eos={eos_token_id}, stop_tokens={})",
            start.elapsed().as_secs_f32(),
            stop_token_ids.len()
        );

        Ok(Self {
            name: name.to_string(),
            path: model_path.clone(),
            model,
            tokenizer,
            device: device.clone(),
            eos_token_id,
            stop_token_ids,
        })
    }

    /// Format prompt with chat template
    fn format_chat_prompt(&self, prompt: &str) -> String {
        format!(
            "<|im_start|>system\nYou are a helpful AI assistant powered by TITAN Synapse.<|im_end|>\n\
             <|im_start|>user\n{prompt}<|im_end|>\n\
             <|im_start|>assistant\n"
        )
    }

    /// Generate text from a prompt
    pub fn generate(&mut self, prompt: &str, max_tokens: u32, sampler: &SamplerConfig) -> Result<String> {
        let formatted = self.format_chat_prompt(prompt);

        // Tokenize
        let encoding = self.tokenizer.encode(formatted.as_str(), true)
            .map_err(|e| anyhow::anyhow!("Tokenize error: {e}"))?;
        let tokens: Vec<u32> = encoding.get_ids().to_vec();

        if tokens.is_empty() {
            return Ok("(empty prompt)".into());
        }

        tracing::info!("Prompt: {} tokens, generating up to {max_tokens}", tokens.len());

        let mut generated_tokens: Vec<u32> = Vec::new();

        // Process prompt (prefill) — feed all tokens at once
        let input = Tensor::new(tokens.as_slice(), &self.device)?
            .unsqueeze(0)?; // (1, seq_len)
        // Model forward returns (batch_size, vocab_size) — already extracts last position
        let logits = self.model.forward(&input, 0)?;
        let mut pos = tokens.len();

        // logits shape: (1, vocab_size) → squeeze to (vocab_size,)
        let logits_flat = logits.squeeze(0)?;
        let logits_vec: Vec<f32> = logits_flat.to_vec1()?;

        // Sample first token
        let mut next_token = sampler.sample(&logits_vec);

        if self.is_stop_token(next_token) {
            return Ok(String::new());
        }
        generated_tokens.push(next_token);

        // Autoregressive generation
        for _ in 1..max_tokens {
            let input = Tensor::new(&[next_token], &self.device)?
                .unsqueeze(0)?; // (1, 1)
            let logits = self.model.forward(&input, pos)?;
            pos += 1;

            // (1, vocab_size) → (vocab_size,)
            let logits_flat = logits.squeeze(0)?;
            let logits_vec: Vec<f32> = logits_flat.to_vec1()?;

            next_token = sampler.sample(&logits_vec);
            if self.is_stop_token(next_token) {
                break;
            }
            generated_tokens.push(next_token);
        }

        tracing::info!(
            "Generated {} tokens from {} prompt tokens",
            generated_tokens.len(),
            tokens.len()
        );

        self.decode_tokens(&generated_tokens)
    }

    fn is_stop_token(&self, token: u32) -> bool {
        token == self.eos_token_id || self.stop_token_ids.contains(&token)
    }

    fn decode_tokens(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(tokens, true)
            .map_err(|e| anyhow::anyhow!("Decode error: {e}"))
    }
}
