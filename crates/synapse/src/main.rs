mod cli;
mod config;
mod dashboard;
mod inference;
mod server;
mod openai;
mod streaming;
mod swarm;
mod learn;
mod memory;
mod vram;
mod format;
mod arch;

use clap::{Parser, Subcommand};
use anyhow::Result;

#[derive(Parser)]
#[command(name = "synapse")]
#[command(about = "TITAN Synapse — Small models that think together. And learn.")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the Synapse inference server
    Serve {
        /// Port to listen on
        #[arg(short, long, default_value = "6900")]
        port: u16,
        /// Config file path
        #[arg(short, long)]
        config: Option<String>,
    },
    /// Show system status (GPU, loaded models, VRAM)
    Status,
    /// List available models
    Models,
    /// Pull a model from HuggingFace
    Pull {
        /// Model name (e.g., qwen3-3b, qwen3-0.6b)
        model: String,
    },
    /// Export a specialist as .synapse file
    Export {
        /// Specialist name
        name: String,
        /// Output path
        #[arg(short, long)]
        output: Option<String>,
    },
    /// Import a .synapse specialist file
    Import {
        /// Path to .synapse file
        path: String,
    },
    /// Show learning engine status
    Learn {
        #[command(subcommand)]
        command: LearnCommands,
    },
    /// Run inference benchmarks
    Bench {
        /// Model to benchmark
        #[arg(short, long)]
        model: Option<String>,
    },
    /// Run standardized evaluation (MMLU, HumanEval, MT-Bench, Safety)
    Eval,
    /// Community Specialist Hub — share and discover trained specialists
    Hub {
        #[command(subcommand)]
        command: HubCommands,
    },
    /// Train our own Synapse model from scratch (SFT + DPO + GGUF export)
    Train {
        /// Training stage: full, sft, dpo, export
        #[arg(short, long, default_value = "full")]
        stage: String,
        /// Base model architecture (Apache 2.0 licensed, we fine-tune into OUR model)
        #[arg(short, long, default_value = "Qwen/Qwen2.5-3B")]
        base_model: String,
        /// Output model name
        #[arg(short, long, default_value = "synapse-3b")]
        output: String,
    },
    /// Start the server (alias for serve)
    Up {
        /// Port to listen on
        #[arg(short, long, default_value = "6900")]
        port: u16,
    },
}

#[derive(Subcommand)]
enum LearnCommands {
    /// Show learning status
    Status,
    /// Force training now
    TrainNow,
}

#[derive(Subcommand)]
enum HubCommands {
    /// Search for community specialists
    Search {
        /// Search query
        query: String,
    },
    /// Install a specialist from HuggingFace
    Install {
        /// HuggingFace repo (e.g., user/synapse-python-expert)
        repo: String,
    },
    /// Push your trained specialist to HuggingFace
    Push {
        /// Specialist name to push
        name: String,
    },
    /// List hub info and commands
    List,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "synapse=info".into()),
        )
        .init();

    let cli = Cli::parse();
    let cfg = config::SynapseConfig::load(None)?;

    match cli.command {
        Commands::Serve { port, config: config_path } => {
            let cfg = if let Some(path) = config_path {
                config::SynapseConfig::load(Some(&path))?
            } else {
                cfg
            };
            server::run(cfg, port).await
        }
        Commands::Up { port } => {
            server::run(cfg, port).await
        }
        Commands::Status => {
            cli::status::run(&cfg).await
        }
        Commands::Models => {
            cli::models::run(&cfg).await
        }
        Commands::Pull { model } => {
            cli::pull::run(&cfg, &model).await
        }
        Commands::Export { name, output } => {
            cli::export::run(&cfg, &name, output.as_deref()).await
        }
        Commands::Import { path } => {
            cli::import::run(&cfg, &path).await
        }
        Commands::Learn { command } => match command {
            LearnCommands::Status => cli::learn::status(&cfg).await,
            LearnCommands::TrainNow => cli::learn::train_now(&cfg).await,
        },
        Commands::Bench { model } => {
            cli::bench::run(&cfg, model.as_deref()).await
        }
        Commands::Eval => {
            cli::eval::run(&cfg).await
        }
        Commands::Train { stage, base_model, output } => {
            cli::train::run(&cfg, &stage, &base_model, &output).await
        }
        Commands::Hub { command } => match command {
            HubCommands::Search { query } => cli::hub::search(&query).await,
            HubCommands::Install { repo } => cli::hub::install(&cfg, &repo).await,
            HubCommands::Push { name } => cli::hub::push(&cfg, &name).await,
            HubCommands::List => cli::hub::list().await,
        }
    }
}
