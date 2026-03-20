use anyhow::Result;
use colored::Colorize;
use crate::config::SynapseConfig;

pub async fn run(config: &SynapseConfig) -> Result<()> {
    println!("{}", "Available Models".bold().cyan());
    println!("{}", "═".repeat(50));

    // Scan models directory
    if !config.models_dir.exists() {
        println!("  No models found. Use {} to download one.", "synapse pull <model>".yellow());
        return Ok(());
    }

    let mut found = false;
    for entry in std::fs::read_dir(&config.models_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            let size_mb = std::fs::metadata(&path)?.len() / (1024 * 1024);
            println!("  {} ({} MB)", name.green(), size_mb);
            found = true;
        }
    }

    if !found {
        println!("  No models found. Use {} to download one.", "synapse pull <model>".yellow());
        println!("\n  Available models:");
        println!("    {} — Coordinator (0.5 GB)", "qwen3-0.6b".yellow());
        println!("    {} — Specialist base (2.1 GB)", "qwen3-3b".yellow());
        println!("    {} — Generalist (4.5 GB)", "qwen3-7b".yellow());
    }

    Ok(())
}
