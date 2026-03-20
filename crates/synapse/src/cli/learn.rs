use anyhow::Result;
use colored::Colorize;
use crate::config::SynapseConfig;
use crate::learn::LearningEngine;

pub async fn status(config: &SynapseConfig) -> Result<()> {
    println!("{}", "Learning Engine Status".bold().cyan());
    println!("{}", "═".repeat(50));

    let engine = LearningEngine::new(&config.learning.sidecar_url, config.learning.enabled);

    match engine.status().await {
        Ok(status) => {
            println!("  {} {}", "Pairs collected:".bold(), status.pairs_collected);
            println!("  {} {}", "Training queue:".bold(), status.training_queue);
            println!("  {} {}", "Adapters created:".bold(), status.adapters_created);
            if let Some(last) = status.last_trained {
                println!("  {} {}", "Last trained:".bold(), last);
            }
        }
        Err(e) => {
            println!("  {} Learning sidecar not reachable: {e}", "⚠".yellow());
            println!("  Start it with: {}", "docker compose up synapse-learn".yellow());
        }
    }

    Ok(())
}

pub async fn train_now(config: &SynapseConfig) -> Result<()> {
    println!("{}", "Triggering training...".bold().cyan());

    let engine = LearningEngine::new(&config.learning.sidecar_url, config.learning.enabled);

    let request = crate::learn::engine::TrainRequest {
        specialist: "general".into(),
        base_model: config.base_model.clone(),
    };

    match engine.train_now(request).await {
        Ok(result) => {
            println!("  {} Training complete!", "✓".green());
            println!("  Adapter: {}", result.adapter_path);
            println!("  Loss: {:.4}", result.loss);
            println!("  Pairs used: {}", result.pairs_used);
        }
        Err(e) => {
            println!("  {} Training failed: {e}", "✗".red());
        }
    }

    Ok(())
}
