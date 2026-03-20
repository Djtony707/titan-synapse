use anyhow::Result;
use colored::Colorize;
use crate::config::SynapseConfig;
use crate::format::{SynapseManifest, packer};

pub async fn run(config: &SynapseConfig, name: &str, output: Option<&str>) -> Result<()> {
    let output_path = output
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(format!("{name}.synapse")));

    println!("{} specialist '{}'...", "Exporting".bold().cyan(), name.yellow());

    let mut manifest = SynapseManifest::new(name, &config.base_model);

    // Count adapters
    if config.adapters_dir.exists() {
        manifest.adapter_count = std::fs::read_dir(&config.adapters_dir)?
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
            .count() as u32;
    }

    // Set capabilities from config
    if let Some(spec) = config.specialists.iter().find(|s| s.name == name) {
        manifest.capabilities = spec.capabilities.clone();
    }

    let knowledge_db = config.data_dir.join("knowledge.db");
    let db_path = if knowledge_db.exists() { Some(knowledge_db.as_path()) } else { None };

    packer::pack(
        &manifest,
        &config.models_dir,
        &config.adapters_dir,
        db_path,
        &output_path,
    )?;

    println!("{} Exported to {}", "Done!".bold().green(), output_path.display());
    println!("  Model: {}", manifest.base_model);
    println!("  Adapters: {}", manifest.adapter_count);
    println!("  Capabilities: {}", manifest.capabilities.join(", "));

    Ok(())
}
