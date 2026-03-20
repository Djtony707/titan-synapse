use anyhow::Result;
use colored::Colorize;
use crate::config::SynapseConfig;

/// Known model mappings to HuggingFace repos
const MODEL_REGISTRY: &[(&str, &str, &str)] = &[
    ("qwen3-0.6b", "Qwen/Qwen3-0.6B-GGUF", "qwen3-0.6b-q4_k_m.gguf"),
    ("qwen3-3b",   "Qwen/Qwen3-4B-GGUF",   "qwen3-4b-q4_k_m.gguf"),
    ("qwen3-7b",   "Qwen/Qwen3-8B-GGUF",   "qwen3-8b-q4_k_m.gguf"),
];

pub async fn run(config: &SynapseConfig, model: &str) -> Result<()> {
    let (name, repo, filename) = MODEL_REGISTRY.iter()
        .find(|(n, _, _)| *n == model)
        .ok_or_else(|| anyhow::anyhow!(
            "Unknown model '{model}'. Available: {}",
            MODEL_REGISTRY.iter().map(|(n, _, _)| *n).collect::<Vec<_>>().join(", ")
        ))?;

    let output_path = config.models_dir.join(filename);
    if output_path.exists() {
        println!("  {} {} already downloaded", "✓".green(), name);
        return Ok(());
    }

    std::fs::create_dir_all(&config.models_dir)?;

    println!("{} {} from {}", "Pulling".bold().cyan(), name, repo);
    println!("  Downloading to {}", output_path.display());

    // Use huggingface-cli if available, otherwise curl
    let hf_cli = tokio::process::Command::new("huggingface-cli")
        .args(["download", repo, filename, "--local-dir", &config.models_dir.to_string_lossy()])
        .output()
        .await;

    match hf_cli {
        Ok(out) if out.status.success() => {
            println!("  {} Downloaded {}", "✓".green(), name);
        }
        _ => {
            // Fallback to direct URL download
            let url = format!("https://huggingface.co/{repo}/resolve/main/{filename}");
            println!("  Using direct download: {url}");

            let output = tokio::process::Command::new("curl")
                .args(["-L", "-o", &output_path.to_string_lossy(), &url, "--progress-bar"])
                .status()
                .await?;

            if output.success() {
                println!("  {} Downloaded {}", "✓".green(), name);
            } else {
                anyhow::bail!("Failed to download model. Try manually:\n  huggingface-cli download {repo} {filename}");
            }
        }
    }

    Ok(())
}
