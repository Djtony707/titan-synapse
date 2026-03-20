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
    pub async fn process(&self, messages: &[Message], engine: &InferenceEngine) -> Result<String> {
        let last_message = messages.last()
            .map(|m| m.content.as_str())
            .unwrap_or("");

        // Step 1: Coordinator decides routing
        let routing = self.coordinator.route(last_message);

        match routing {
            RoutingDecision::Single { specialist } => {
                // Simple query → single specialist
                tracing::info!("Routing to specialist: {specialist}");
                let prompt = self.build_prompt(messages, &specialist);
                engine.generate(&prompt, Some(&specialist), 2048, 0.7).await
            }
            RoutingDecision::Swarm { subtasks } => {
                // Complex query → multi-specialist swarm
                tracing::info!("Swarm mode: {} subtasks", subtasks.len());
                let mut results = Vec::new();

                for task in &subtasks {
                    let prompt = format!("Task: {}\n\nContext: {last_message}", task.description);
                    let result = engine.generate(
                        &prompt,
                        Some(&task.specialist),
                        1024,
                        0.7,
                    ).await?;
                    results.push((task.specialist.clone(), result));
                }

                // Synthesize results
                self.synthesizer.merge(&results)
            }
        }
    }

    fn build_prompt(&self, messages: &[Message], _specialist: &str) -> String {
        messages.iter()
            .map(|m| format!("{}: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n")
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
