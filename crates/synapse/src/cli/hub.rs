use anyhow::Result;
use colored::Colorize;
use crate::config::SynapseConfig;

/// Community Specialist Hub — share and discover trained specialists.
///
/// "Models produced by real users, not corporations."
///
/// Push your trained specialist to HuggingFace or pull community specialists.
/// Every user trains on their own data. The best specialists get shared.
/// This is how we build an AI that's smarter than any single 120B model:
/// a community of specialists, each an expert in their domain.

pub async fn push(config: &SynapseConfig, specialist: &str) -> Result<()> {
    println!("{} {specialist}", "Pushing specialist".bold().cyan());

    // Find the specialist's .synapse bundle or adapter
    let adapter_dir = config.adapters_dir.join(format!("{specialist}_qlora"));
    let synapse_file = config.data_dir.join(format!("{specialist}.synapse"));

    let source = if synapse_file.exists() {
        println!("  Found .synapse bundle: {}", synapse_file.display());
        synapse_file
    } else if adapter_dir.exists() {
        println!("  Found LoRA adapter: {}", adapter_dir.display());
        adapter_dir
    } else {
        anyhow::bail!(
            "No trained specialist '{specialist}' found.\n  \
             Train one first: synapse learn train-now\n  \
             Or export: synapse export {specialist}"
        );
    };

    // Check for HuggingFace CLI
    let hf_user = get_hf_username().await;
    let repo_name = match &hf_user {
        Some(user) => format!("{user}/synapse-{specialist}"),
        None => {
            println!("  {} HuggingFace not configured", "⚠".yellow());
            println!("  Run: pip install huggingface-cli && huggingface-cli login");
            println!("  Or: export HF_TOKEN=your_token_here");
            println!();
            println!("  For now, you can share manually:");
            println!("  1. Create a repo on huggingface.co");
            println!("  2. Upload {}", source.display());
            return Ok(());
        }
    };

    println!("  Uploading to {}", repo_name.bold());

    // Create repo if needed
    let _ = tokio::process::Command::new("huggingface-cli")
        .args(["repo", "create", &format!("synapse-{specialist}"), "--type", "model", "-y"])
        .output()
        .await;

    // Upload
    let upload_result = tokio::process::Command::new("huggingface-cli")
        .args(["upload", &repo_name, &source.to_string_lossy(), "."])
        .status()
        .await;

    match upload_result {
        Ok(status) if status.success() => {
            println!("  {} Pushed to https://huggingface.co/{repo_name}", "✓".green());
            println!();
            println!("  Others can now install it:");
            println!("    synapse hub install {repo_name}");
        }
        _ => {
            println!("  {} Upload failed. Try manually:", "✗".red());
            println!("    huggingface-cli upload {repo_name} {}", source.display());
        }
    }

    Ok(())
}

pub async fn install(_config: &SynapseConfig, repo: &str) -> Result<()> {
    println!("{} {repo}", "Installing specialist".bold().cyan());

    let parts: Vec<&str> = repo.split('/').collect();
    if parts.len() != 2 {
        anyhow::bail!("Invalid repo format. Use: user/synapse-specialist-name");
    }

    // Download from HuggingFace
    let output = tokio::process::Command::new("huggingface-cli")
        .args(["download", repo, "--local-dir", &format!("{}", _config.adapters_dir.join(parts[1]).display())])
        .status()
        .await;

    match output {
        Ok(status) if status.success() => {
            println!("  {} Installed {}", "✓".green(), parts[1]);
            println!("  Specialist will be available on next server restart.");
        }
        _ => {
            anyhow::bail!("Failed to download. Make sure the repo exists and is public.");
        }
    }

    Ok(())
}

pub async fn search(query: &str) -> Result<()> {
    println!("{} '{query}'", "Searching specialists".bold().cyan());
    println!();

    // Search HuggingFace for synapse specialists
    let client = reqwest::Client::new();
    let url = format!(
        "https://huggingface.co/api/models?search=synapse-{query}&sort=downloads&limit=10"
    );

    match client.get(&url).send().await {
        Ok(resp) => {
            let models: Vec<serde_json::Value> = resp.json().await.unwrap_or_default();

            if models.is_empty() {
                println!("  No specialists found for '{query}'.");
                println!("  Be the first! Train a specialist and push it:");
                println!("    synapse hub push {query}_expert");
                return Ok(());
            }

            for model in &models {
                let id = model["modelId"].as_str().unwrap_or("unknown");
                let downloads = model["downloads"].as_u64().unwrap_or(0);
                let likes = model["likes"].as_u64().unwrap_or(0);
                println!("  {} {} (↓{downloads} ♥{likes})", "•".cyan(), id);
            }

            println!();
            println!("  Install with: synapse hub install <model-id>");
        }
        Err(e) => {
            println!("  {} Search failed: {e}", "✗".red());
            println!("  Check your internet connection.");
        }
    }

    Ok(())
}

pub async fn list() -> Result<()> {
    println!("{}", "Community Specialist Hub".bold().cyan());
    println!("{}", "═".repeat(50));
    println!();
    println!("{}", "Popular Specialists:".bold());
    println!("  Coming soon — be the first to push a specialist!");
    println!();
    println!("{}", "Commands:".bold());
    println!("  synapse hub search <query>     Search for specialists");
    println!("  synapse hub install <repo>     Install a specialist");
    println!("  synapse hub push <name>        Share your specialist");
    println!();
    println!("Train a specialist, push it, help the community.");
    println!("That's how we make tiny models smarter than 120B.");

    Ok(())
}

async fn get_hf_username() -> Option<String> {
    // Check HF_TOKEN or whoami
    if let Ok(output) = tokio::process::Command::new("huggingface-cli")
        .args(["whoami"])
        .output()
        .await
    {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            return stdout.lines().next().map(|s| s.trim().to_string());
        }
    }
    None
}
