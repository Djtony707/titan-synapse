use anyhow::Result;
use crate::config::SynapseConfig;
use crate::inference::InferenceEngine;
use crate::openai::Message;
use super::coordinator::Coordinator;
use super::synthesizer::Synthesizer;

/// Top-level swarm orchestrator — decides single vs multi-specialist routing
pub struct Orchestrator {
    coordinator: Coordinator,
    synthesizer: Synthesizer,
}

impl Orchestrator {
    pub fn new(config: &SynapseConfig) -> Self {
        Self {
            coordinator: Coordinator::new(config),
            synthesizer: Synthesizer::new(),
        }
    }

    /// Process a chat request — route to specialist(s) and return response
    pub async fn process(
        &self,
        messages: &[Message],
        engine: &InferenceEngine,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
    ) -> Result<String> {
        let last_message = messages.last()
            .map(|m| m.content.as_str())
            .unwrap_or("");

        let max_tokens = max_tokens.unwrap_or(2048);
        let temperature = temperature.unwrap_or(0.7);

        // Step 1: Coordinator decides routing
        let routing = self.coordinator.route(last_message);

        match routing {
            RoutingDecision::Single { specialist } => {
                tracing::info!("Routing to specialist: {specialist}");
                engine.generate(last_message, Some(&specialist), max_tokens, temperature).await
            }
            RoutingDecision::Swarm { subtasks } => {
                tracing::info!("Swarm mode: {} subtasks", subtasks.len());
                let mut results = Vec::new();

                for task in &subtasks {
                    let prompt = format!("Task: {}\n\nContext: {last_message}", task.description);
                    let result = engine.generate(
                        &prompt,
                        Some(&task.specialist),
                        max_tokens / subtasks.len() as u32,
                        temperature,
                    ).await?;
                    results.push((task.specialist.clone(), result));
                }

                self.synthesizer.merge(&results)
            }
        }
    }
}

pub enum RoutingDecision {
    Single { specialist: String },
    Swarm { subtasks: Vec<SubTask> },
}

pub struct SubTask {
    pub specialist: String,
    pub description: String,
}

impl std::fmt::Display for RoutingDecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RoutingDecision::Single { specialist } => write!(f, "Single({specialist})"),
            RoutingDecision::Swarm { subtasks } => {
                write!(f, "Swarm({})", subtasks.iter().map(|t| t.specialist.as_str()).collect::<Vec<_>>().join(", "))
            }
        }
    }
}
