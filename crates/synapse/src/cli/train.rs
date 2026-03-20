use anyhow::Result;
use colored::Colorize;
use crate::config::SynapseConfig;
use std::process::Command;

/// Train our own Synapse model — not someone else's, OURS.
///
/// Pipeline:
/// 1. Generate training data (swarm routing + honesty + public datasets + user prefs)
/// 2. SFT (Supervised Fine-Tuning) with QLoRA on the base architecture
/// 3. DPO (Direct Preference Optimization) using collected preference pairs
/// 4. Export to GGUF for the Synapse inference engine
///
/// The base architecture (Qwen2.5-3B) is Apache 2.0 licensed.
/// Once we fine-tune it, the result is OUR model: synapse-3b.
pub async fn run(config: &SynapseConfig, stage: &str, base_model: &str, output: &str) -> Result<()> {
    println!("{}", "╔══════════════════════════════════════════════════╗".bold().purple());
    println!("{}", "║     TITAN SYNAPSE — Model Training Pipeline     ║".bold().purple());
    println!("{}", "║     Building OUR model. Not theirs. OURS.       ║".bold().purple());
    println!("{}", "╚══════════════════════════════════════════════════╝".bold().purple());
    println!();

    println!("  {} {}", "Stage:".bold(), stage.cyan());
    println!("  {} {}", "Base Architecture:".bold(), base_model);
    println!("  {} {}", "Output Model:".bold(), output.green().bold());
    println!("  {} {}", "License:".bold(), "Apache 2.0 (our model, our weights)");
    println!();

    // Check if Python training script exists
    let train_script = config.data_dir
        .parent().unwrap_or(&config.data_dir)
        .join("python/synapse_learn/train_base.py");

    // Also check relative to current directory
    let script_paths = [
        std::path::PathBuf::from("python/synapse_learn/train_base.py"),
        train_script.clone(),
        config.data_dir.join("train_base.py"),
    ];

    let script = script_paths.iter().find(|p| p.exists());

    if let Some(script_path) = script {
        println!("  {} Found training script: {}", "✓".green(), script_path.display());
        println!();

        // Run the Python training pipeline
        println!("  {} Starting training...", "⚡".yellow());
        println!("  {}", "This will take a while. Go grab coffee. Or three.".dimmed());
        println!();

        let result = Command::new("python")
            .args([
                script_path.to_str().unwrap(),
                "--stage", stage,
                "--base-model", base_model,
                "--output", output,
            ])
            .env("SYNAPSE_DATA_DIR", config.data_dir.to_str().unwrap())
            .status();

        match result {
            Ok(status) if status.success() => {
                println!();
                println!("  {}", "═".repeat(50).purple());
                println!("  {} {}", "✓".green().bold(), "Model training complete!".bold().green());
                println!();
                println!("  Your model: {}", format!("{}.gguf", output).cyan().bold());
                println!("  Location: {}", config.models_dir.display());
                println!();
                println!("  {} Start using it:", "→".bold());
                println!("    {} synapse up", "$".dimmed());
                println!("    Then open {} in your browser", "http://localhost:6900".cyan());
                println!();
                println!("  {}", "This model is YOURS. Train it more. Make it smarter.".bold());
                println!("  {}", "Every conversation it has makes it better.".dimmed());
            }
            Ok(status) => {
                println!("  {} Training exited with code: {:?}", "⚠".yellow(), status.code());
            }
            Err(e) => {
                println!("  {} Failed to run training: {e}", "✗".red());
                println!();
                println!("  {} Install Python dependencies:", "→".bold());
                println!("    pip install torch transformers peft trl bitsandbytes datasets");
            }
        }
    } else {
        // No Python script found — print manual instructions
        println!("  {} Training script not found at expected paths.", "⚠".yellow());
        println!();
        println!("  {} Manual training:", "→".bold());
        println!("    cd python/synapse_learn");
        println!("    pip install torch transformers peft trl bitsandbytes datasets");
        println!("    python train_base.py --stage {} --base-model {} --output {}", stage, base_model, output);
    }

    Ok(())
}
