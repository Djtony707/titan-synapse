use anyhow::Result;
use colored::Colorize;
use crate::config::SynapseConfig;
use crate::vram::VramManager;

pub async fn run(config: &SynapseConfig) -> Result<()> {
    println!("{}", "TITAN Synapse Status".bold().cyan());
    println!("{}", "═".repeat(50));

    // Version
    println!("  {} {}", "Version:".bold(), env!("CARGO_PKG_VERSION"));
    println!("  {} {}", "Data dir:".bold(), config.data_dir.display());

    // GPU info
    println!("\n{}", "GPU".bold().yellow());
    match VramManager::gpu_info().await {
        Ok(info) => {
            println!("  {} {}", "Name:".bold(), info.name);
            println!("  {} {} MB / {} MB ({:.1}% used)",
                "VRAM:".bold(),
                info.vram_used_mb, info.vram_total_mb,
                (info.vram_used_mb as f32 / info.vram_total_mb.max(1) as f32) * 100.0
            );
            println!("  {} {:.1}%", "GPU Util:".bold(), info.utilization_percent);
            if let Some(temp) = info.temperature_c {
                println!("  {} {}°C", "Temp:".bold(), temp);
            }
        }
        Err(e) => println!("  {} {e}", "Error:".red()),
    }

    // Models
    println!("\n{}", "Configuration".bold().yellow());
    println!("  {} {}", "Coordinator:".bold(), config.coordinator_model);
    println!("  {} {}", "Base model:".bold(), config.base_model);
    println!("  {} {}", "Specialists:".bold(), config.specialists.len());
    for spec in &config.specialists {
        println!("    {} [{}]",
            format!("• {}", spec.name).green(),
            spec.capabilities.join(", ")
        );
    }

    // Learning
    println!("\n{}", "Learning".bold().yellow());
    println!("  {} {}", "Enabled:".bold(),
        if config.learning.enabled { "yes".green() } else { "no".red() }
    );
    println!("  {} {}", "Sidecar:".bold(), config.learning.sidecar_url);

    Ok(())
}
