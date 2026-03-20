use anyhow::Result;
use std::collections::HashMap;
use std::path::PathBuf;

/// LoRA adapter that can be hot-swapped onto a base model
pub struct LoraAdapter {
    pub name: String,
    pub path: PathBuf,
    pub rank: u32,
    pub loaded: bool,
    /// Adapter tensors keyed by layer name (e.g., "model.layers.0.self_attn.q_proj.lora_A")
    pub tensors: Option<HashMap<String, Vec<f32>>>,
}

impl LoraAdapter {
    /// Load adapter weights from SafeTensors file
    pub fn load(name: &str, path: PathBuf) -> Result<Self> {
        tracing::info!("Loading LoRA adapter '{name}' from {}", path.display());

        let mut adapter = Self {
            name: name.to_string(),
            path: path.clone(),
            rank: 16,
            loaded: false,
            tensors: None,
        };

        // Try to actually load SafeTensors weights
        if path.exists() && path.extension().is_some_and(|ext| ext == "safetensors") {
            match adapter.load_safetensors() {
                Ok(tensor_count) => {
                    tracing::info!("LoRA adapter '{name}' loaded: {tensor_count} tensors");
                    adapter.loaded = true;
                }
                Err(e) => {
                    tracing::warn!("Failed to load LoRA tensors for '{name}': {e}");
                    // Still usable as a placeholder — will be trained later
                }
            }
        }

        Ok(adapter)
    }

    /// Load SafeTensors file and extract tensor data
    fn load_safetensors(&mut self) -> Result<usize> {
        let data = std::fs::read(&self.path)?;
        let tensors = safetensors::SafeTensors::deserialize(&data)
            .map_err(|e| anyhow::anyhow!("SafeTensors parse error: {e}"))?;

        let mut loaded_tensors = HashMap::new();
        let mut detected_rank = 0u32;

        for (name, tensor_view) in tensors.tensors() {
            let shape = tensor_view.shape();

            // Detect LoRA rank from lora_A shape (rank is the smaller dimension)
            if name.contains("lora_A") && shape.len() == 2 {
                detected_rank = shape[0].min(shape[1]) as u32;
            }

            // Store tensor data as f32 (convert from whatever dtype)
            let float_data: Vec<f32> = match tensor_view.dtype() {
                safetensors::Dtype::F32 => {
                    tensor_view.data()
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect()
                }
                safetensors::Dtype::F16 => {
                    tensor_view.data()
                        .chunks_exact(2)
                        .map(|b| {
                            let bits = u16::from_le_bytes([b[0], b[1]]);
                            half::f16::from_bits(bits).to_f32()
                        })
                        .collect()
                }
                safetensors::Dtype::BF16 => {
                    tensor_view.data()
                        .chunks_exact(2)
                        .map(|b| {
                            let bits = u16::from_le_bytes([b[0], b[1]]);
                            half::bf16::from_bits(bits).to_f32()
                        })
                        .collect()
                }
                other => {
                    tracing::debug!("Skipping tensor {name} with unsupported dtype: {other:?}");
                    continue;
                }
            };

            loaded_tensors.insert(name.to_string(), float_data);
        }

        if detected_rank > 0 {
            self.rank = detected_rank;
        }

        let count = loaded_tensors.len();
        self.tensors = Some(loaded_tensors);
        Ok(count)
    }

    /// Size in MB (actual if loaded, estimated otherwise)
    pub fn size_mb(&self) -> f32 {
        if let Some(ref tensors) = self.tensors {
            let total_bytes: usize = tensors.values()
                .map(|t| t.len() * 4) // f32 = 4 bytes
                .sum();
            total_bytes as f32 / (1024.0 * 1024.0)
        } else {
            // Estimate: for rank=16, 3B model: ~10MB
            self.rank as f32 * 0.625
        }
    }

    /// Get tensor names that match a pattern
    pub fn matching_tensors(&self, pattern: &str) -> Vec<&str> {
        match &self.tensors {
            Some(tensors) => tensors.keys()
                .filter(|k| k.contains(pattern))
                .map(|k| k.as_str())
                .collect(),
            None => vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_adapter_placeholder() {
        let adapter = LoraAdapter {
            name: "test".into(),
            path: PathBuf::from("/nonexistent/test.safetensors"),
            rank: 16,
            loaded: false,
            tensors: None,
        };
        assert_eq!(adapter.size_mb(), 10.0);
        assert!(adapter.matching_tensors("lora_A").is_empty());
    }

    #[test]
    fn test_lora_adapter_with_tensors() {
        let mut tensors = HashMap::new();
        tensors.insert("layer.0.lora_A".into(), vec![0.0f32; 1024]);
        tensors.insert("layer.0.lora_B".into(), vec![0.0f32; 1024]);

        let adapter = LoraAdapter {
            name: "test".into(),
            path: PathBuf::from("/test.safetensors"),
            rank: 16,
            loaded: true,
            tensors: Some(tensors),
        };

        assert!(adapter.size_mb() > 0.0);
        assert_eq!(adapter.matching_tensors("lora_A").len(), 1);
        assert_eq!(adapter.matching_tensors("lora_B").len(), 1);
    }
}
