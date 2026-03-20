use anyhow::Result;
use colored::Colorize;
use crate::config::SynapseConfig;
use crate::inference::InferenceEngine;

pub async fn run(config: &SynapseConfig, model: Option<&str>) -> Result<()> {
    let model_name = model.unwrap_or(&config.base_model);
    println!("{} {model_name}", "Benchmarking".bold().cyan());
    println!("{}", "═".repeat(50));

    let engine = InferenceEngine::new(config)?;

    let prompts = [
        "What is a Python decorator?",
        "Write a SQL query to find the top 10 users by posts.",
        "Explain the difference between TCP and UDP.",
        "How does garbage collection work in Go?",
    ];

    let mut total_tokens = 0u64;
    let start = std::time::Instant::now();

    for prompt in &prompts {
        let prompt_start = std::time::Instant::now();
        let response = engine.generate(prompt, None, 256, 0.7).await?;
        let elapsed = prompt_start.elapsed();

        // Approximate token count (4 chars per token)
        let tokens = response.len() as u64 / 4;
        total_tokens += tokens;

        let tok_s = tokens as f64 / elapsed.as_secs_f64();
        println!("  {} ~{} tokens in {:.0}ms ({:.0} tok/s)",
            "•".green(), tokens, elapsed.as_millis(), tok_s);
    }

    let total_elapsed = start.elapsed();
    let avg_tok_s = total_tokens as f64 / total_elapsed.as_secs_f64();

    println!("\n{}", "Results".bold().yellow());
    println!("  {} {:.0} tok/s", "Average throughput:".bold(), avg_tok_s);
    println!("  {} {:.0}ms", "Total time:".bold(), total_elapsed.as_millis());
    println!("  {} {}", "Total tokens:".bold(), total_tokens);

    Ok(())
}
