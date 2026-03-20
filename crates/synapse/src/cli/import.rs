use anyhow::Result;
use colored::Colorize;
use crate::config::SynapseConfig;
use crate::format::packer;

pub async fn run(config: &SynapseConfig, path: &str) -> Result<()> {
    let synapse_path = std::path::PathBuf::from(path);

    if !synapse_path.exists() {
        anyhow::bail!("File not found: {path}");
    }

    println!("{} specialist from {}...", "Importing".bold().cyan(), path.yellow());

    let manifest = packer::unpack(
        &synapse_path,
        &config.models_dir,
        &config.adapters_dir,
    )?;

    println!("{} Imported specialist '{}'", "Done!".bold().green(), manifest.name);
    println!("  Base model: {}", manifest.base_model);
    println!("  Quantization: {}", manifest.base_quantization);
    println!("  Adapters: {}", manifest.adapter_count);
    println!("  Capabilities: {}", manifest.capabilities.join(", "));
    println!("  Performance score: {:.2}", manifest.performance_score);

    if manifest.adapter_count > 0 {
        println!("\n  Adapters installed to: {}", config.adapters_dir.display());
    }

    println!("\n  Restart the server to load the new specialist.");

    Ok(())
}
