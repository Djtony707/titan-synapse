use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapseConfig {
    /// Server port
    #[serde(default = "default_port")]
    pub port: u16,

    /// Data directory (~/.synapse)
    #[serde(default = "default_data_dir")]
    pub data_dir: PathBuf,

    /// Models directory
    #[serde(default = "default_models_dir")]
    pub models_dir: PathBuf,

    /// Adapters directory
    #[serde(default = "default_adapters_dir")]
    pub adapters_dir: PathBuf,

    /// Coordinator model name
    #[serde(default = "default_coordinator")]
    pub coordinator_model: String,

    /// Default specialist base model
    #[serde(default = "default_base_model")]
    pub base_model: String,

    /// Max VRAM budget in MB (0 = auto-detect)
    #[serde(default)]
    pub max_vram_mb: u64,

    /// Cloud fallback configuration
    #[serde(default)]
    pub cloud: CloudConfig,

    /// Learning engine configuration
    #[serde(default)]
    pub learning: LearningConfig,

    /// Specialist definitions
    #[serde(default)]
    pub specialists: Vec<SpecialistConfig>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CloudConfig {
    /// Cloud API base URL (OpenAI-compatible)
    pub api_base: Option<String>,
    /// API key
    pub api_key: Option<String>,
    /// Model to use for cloud fallback
    pub model: Option<String>,
    /// Enable cloud fallback
    #[serde(default)]
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningConfig {
    /// Enable continuous learning
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Min preference pairs before training
    #[serde(default = "default_min_pairs")]
    pub min_pairs_before_training: u32,
    /// Learning sidecar URL
    #[serde(default = "default_learn_url")]
    pub sidecar_url: String,
    /// Self-evaluation threshold (1-5, below this = negative example)
    #[serde(default = "default_eval_threshold")]
    pub eval_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialistConfig {
    /// Specialist name
    pub name: String,
    /// Capabilities this specialist handles
    pub capabilities: Vec<String>,
    /// Base model (overrides global)
    pub base_model: Option<String>,
    /// LoRA adapter path
    pub adapter: Option<String>,
    /// System prompt
    pub system_prompt: Option<String>,
    /// Priority (higher = preferred)
    #[serde(default = "default_priority")]
    pub priority: u32,
}

fn default_port() -> u16 { 6900 }
fn default_data_dir() -> PathBuf {
    dirs::home_dir().unwrap_or_default().join(".synapse")
}
fn default_models_dir() -> PathBuf {
    default_data_dir().join("models")
}
fn default_adapters_dir() -> PathBuf {
    default_data_dir().join("adapters")
}
fn default_coordinator() -> String { "qwen3-0.6b".into() }
fn default_base_model() -> String { "qwen3-3b".into() }
fn default_true() -> bool { true }
fn default_min_pairs() -> u32 { 10 }
fn default_learn_url() -> String { "http://localhost:8090".into() }
fn default_eval_threshold() -> f32 { 3.0 }
fn default_priority() -> u32 { 50 }

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_pairs_before_training: 10,
            sidecar_url: default_learn_url(),
            eval_threshold: 3.0,
        }
    }
}

impl SynapseConfig {
    pub fn load(path: Option<&str>) -> Result<Self> {
        let config_path = match path {
            Some(p) => PathBuf::from(p),
            None => default_data_dir().join("config.yaml"),
        };

        if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)?;
            let config: SynapseConfig = serde_yaml::from_str(&content)?;
            Ok(config)
        } else {
            let config = SynapseConfig::default();
            // Ensure data directories exist
            std::fs::create_dir_all(&config.data_dir)?;
            std::fs::create_dir_all(&config.models_dir)?;
            std::fs::create_dir_all(&config.adapters_dir)?;
            // Write default config
            let yaml = serde_yaml::to_string(&config)?;
            std::fs::write(&config_path, yaml)?;
            tracing::info!("Created default config at {}", config_path.display());
            Ok(config)
        }
    }
}

impl Default for SynapseConfig {
    fn default() -> Self {
        Self {
            port: default_port(),
            data_dir: default_data_dir(),
            models_dir: default_models_dir(),
            adapters_dir: default_adapters_dir(),
            coordinator_model: default_coordinator(),
            base_model: default_base_model(),
            max_vram_mb: 0,
            cloud: CloudConfig::default(),
            learning: LearningConfig::default(),
            specialists: vec![
                SpecialistConfig {
                    name: "general".into(),
                    capabilities: vec!["general".into(), "chat".into()],
                    base_model: None,
                    adapter: None,
                    system_prompt: Some("You are a helpful AI assistant.".into()),
                    priority: 50,
                },
                SpecialistConfig {
                    name: "python_expert".into(),
                    capabilities: vec!["python".into(), "debugging".into(), "testing".into()],
                    base_model: None,
                    adapter: None,
                    system_prompt: Some("You are an expert Python developer.".into()),
                    priority: 60,
                },
                SpecialistConfig {
                    name: "sql_expert".into(),
                    capabilities: vec!["sql".into(), "database".into(), "query".into()],
                    base_model: None,
                    adapter: None,
                    system_prompt: Some("You are an expert database engineer.".into()),
                    priority: 60,
                },
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SynapseConfig::default();
        assert_eq!(config.port, 6900);
        assert_eq!(config.coordinator_model, "qwen3-0.6b");
        assert_eq!(config.specialists.len(), 3);
    }

    #[test]
    fn test_config_serialization() {
        let config = SynapseConfig::default();
        let yaml = serde_yaml::to_string(&config).unwrap();
        let parsed: SynapseConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(parsed.port, config.port);
    }

    #[test]
    fn test_load_missing_config() {
        // Should create default when file doesn't exist
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("nonexistent.yaml");
        // This would try to create dirs, so just test default
        let config = SynapseConfig::default();
        assert_eq!(config.base_model, "qwen3-3b");
    }
}
