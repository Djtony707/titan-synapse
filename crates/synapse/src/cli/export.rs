use anyhow::Result;
use crate::config::SynapseConfig;
use crate::format::{SynapseManifest, packer};

pub async fn run(config: &SynapseConfig, name: &str, output: Option<&str>) -> Result<()> {
    let output_path = output
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(format!("{name}.synapse")));

    let manifest = SynapseManifest::new(name, &config.base_model);
    let staging_dir = config.data_dir.join("staging").join(name);
    std::fs::create_dir_all(&staging_dir)?;

    packer::pack(&manifest, &staging_dir, &output_path)?;
    println!("Exported specialist '{name}' to {}", output_path.display());

    Ok(())
}
