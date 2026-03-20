use anyhow::Result;
use crate::config::SynapseConfig;
use crate::format::packer;

pub async fn run(config: &SynapseConfig, path: &str) -> Result<()> {
    let synapse_path = std::path::PathBuf::from(path);
    let output_dir = config.data_dir.join("specialists");
    std::fs::create_dir_all(&output_dir)?;

    let manifest = packer::unpack(&synapse_path, &output_dir)?;
    println!("Imported specialist '{}' (base: {}, adapters: {})",
        manifest.name, manifest.base_model, manifest.adapter_count);

    Ok(())
}
