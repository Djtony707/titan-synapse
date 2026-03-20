use serde::{Deserialize, Serialize};

/// Manifest for .synapse model format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapseManifest {
    pub format: String,
    pub version: String,
    pub name: String,
    pub base_model: String,
    pub base_quantization: String,
    pub vram_estimate_mb: u64,
    pub capabilities: Vec<String>,
    pub adapter_count: u32,
    pub training_pairs_collected: u32,
    pub last_trained: Option<String>,
    pub performance_score: f32,
    pub created_by: String,
}

impl SynapseManifest {
    pub fn new(name: &str, base_model: &str) -> Self {
        Self {
            format: "synapse".into(),
            version: env!("CARGO_PKG_VERSION").into(),
            name: name.into(),
            base_model: base_model.into(),
            base_quantization: "Q4_K_M".into(),
            vram_estimate_mb: 2100,
            capabilities: vec![],
            adapter_count: 0,
            training_pairs_collected: 0,
            last_trained: None,
            performance_score: 0.0,
            created_by: format!("titan-synapse/{}", env!("CARGO_PKG_VERSION")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manifest_creation() {
        let m = SynapseManifest::new("python_expert", "Qwen3-3B");
        assert_eq!(m.format, "synapse");
        assert_eq!(m.name, "python_expert");
    }

    #[test]
    fn test_manifest_serialization() {
        let m = SynapseManifest::new("test", "Qwen3-3B");
        let json = serde_json::to_string_pretty(&m).unwrap();
        let parsed: SynapseManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.name, "test");
    }
}
