use std::path::PathBuf;

/// LoRA adapter that can be hot-swapped onto a base model
pub struct LoraAdapter {
    pub name: String,
    pub path: PathBuf,
    pub rank: u32,
    pub loaded: bool,
}

impl LoraAdapter {
    /// Load adapter weights from SafeTensors file
    pub fn load(name: &str, path: PathBuf) -> anyhow::Result<Self> {
        // TODO: Load actual SafeTensors weights via candle
        // let tensors = safetensors::SafeTensors::deserialize(&data)?;
        tracing::info!("Loading LoRA adapter '{name}' from {}", path.display());

        Ok(Self {
            name: name.to_string(),
            path,
            rank: 16,
            loaded: true,
        })
    }

    /// Size in MB (approximate)
    pub fn size_mb(&self) -> f32 {
        // LoRA adapters are tiny — typically 5-50MB
        // Size depends on rank and number of layers
        // For rank=16, 3B model: ~10MB
        self.rank as f32 * 0.625 // rough estimate
    }
}
