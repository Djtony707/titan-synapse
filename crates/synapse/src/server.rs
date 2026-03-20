use anyhow::Result;
use axum::{
    Router,
    routing::{get, post},
};
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use crate::config::SynapseConfig;
use crate::inference::InferenceEngine;
use crate::swarm::Orchestrator;
use crate::memory::KnowledgeGraph;

pub struct AppState {
    pub config: SynapseConfig,
    pub engine: InferenceEngine,
    pub orchestrator: Orchestrator,
    pub knowledge: KnowledgeGraph,
}

pub type SharedState = Arc<RwLock<AppState>>;

pub async fn run(config: SynapseConfig, port: u16) -> Result<()> {
    tracing::info!("Starting TITAN Synapse on port {port}");

    let knowledge = KnowledgeGraph::new(&config.data_dir.join("knowledge.db"))?;
    let engine = InferenceEngine::new(&config)?;
    let orchestrator = Orchestrator::new(&config);

    let state: SharedState = Arc::new(RwLock::new(AppState {
        config: config.clone(),
        engine,
        orchestrator,
        knowledge,
    }));

    let app = Router::new()
        // Web Dashboard — normal people can open a browser and chat
        .route("/", get(dashboard))
        // OpenAI-compatible endpoints
        .route("/v1/chat/completions", post(crate::openai::chat_completions))
        .route("/v1/models", get(crate::openai::list_models))
        // Health
        .route("/health", get(health))
        // Status + Metacognition
        .route("/api/status", get(api_status))
        .route("/api/confidence", get(api_confidence))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}")).await?;
    tracing::info!("TITAN Synapse ready at http://0.0.0.0:{port}");
    tracing::info!("Dashboard: http://0.0.0.0:{port}/");
    tracing::info!("OpenAI-compatible API: http://0.0.0.0:{port}/v1/chat/completions");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

async fn dashboard() -> axum::response::Html<&'static str> {
    axum::response::Html(crate::dashboard::DASHBOARD_HTML)
}

async fn health() -> &'static str {
    "ok"
}

async fn api_status(
    state: axum::extract::State<SharedState>,
) -> axum::Json<serde_json::Value> {
    let state = state.read().await;

    let fact_count = state.knowledge.fact_count().unwrap_or(0);
    let top_pathways = state.knowledge.top_pathways(5).unwrap_or_default();

    axum::Json(serde_json::json!({
        "status": "running",
        "version": env!("CARGO_PKG_VERSION"),
        "engine": "synapse",
        "models_loaded": state.engine.loaded_models(),
        "has_models": state.engine.has_models(),
        "specialists": state.config.specialists.iter().map(|s| &s.name).collect::<Vec<_>>(),
        "adapters": state.engine.available_adapters(),
        "coordinator": state.config.coordinator_model,
        "base_model": state.config.base_model,
        "knowledge": {
            "facts": fact_count,
            "conversations": state.knowledge.conversation_count().unwrap_or(0),
            "preference_pairs": state.knowledge.total_preference_count().unwrap_or(0),
        },
        "hebbian_routing": {
            "top_pathways": top_pathways.iter().map(|(p, s, avg)| {
                serde_json::json!({"pathway": p, "strength": s, "avg_score": avg})
            }).collect::<Vec<_>>(),
        },
    }))
}

/// Metacognitive confidence report — what the system knows it's good (and bad) at
async fn api_confidence(
    state: axum::extract::State<SharedState>,
) -> axum::Json<serde_json::Value> {
    let state = state.read().await;

    let specialist_confidence = state.knowledge.specialist_confidence_report().unwrap_or_default();
    let pathways = state.knowledge.top_pathways(10).unwrap_or_default();

    axum::Json(serde_json::json!({
        "metacognition": {
            "description": "Specialist confidence scores — the system knows what it knows",
            "specialists": specialist_confidence,
            "hebbian_pathways": pathways.iter().map(|(p, s, avg)| {
                serde_json::json!({
                    "pathway": p,
                    "strength": s,
                    "avg_score": avg,
                    "description": format!("Pathway {} has been reinforced {} times", p, s)
                })
            }).collect::<Vec<_>>(),
            "total_pathways": pathways.len(),
            "learning_status": {
                "preferences_collected": state.knowledge.total_preference_count().unwrap_or(0),
                "conversations_logged": state.knowledge.conversation_count().unwrap_or(0),
                "facts_known": state.knowledge.fact_count().unwrap_or(0),
            }
        }
    }))
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to install CTRL+C signal handler");
    tracing::info!("Shutting down TITAN Synapse...");
}
