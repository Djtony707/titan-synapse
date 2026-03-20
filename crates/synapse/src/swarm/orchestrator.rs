use anyhow::Result;
use crate::config::SynapseConfig;
use crate::inference::{InferenceEngine, GenerationResult};
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
    ) -> Result<GenerationResult> {
        let last_message = messages.last()
            .map(|m| m.content.as_str())
            .unwrap_or("");

        let max_tokens = max_tokens.unwrap_or(2048);
        let temperature = temperature.unwrap_or(0.7);

        let routing = self.coordinator.route(last_message);

        match routing {
            RoutingDecision::Single { specialist } => {
                tracing::info!("Routing to specialist: {specialist}");
                engine.generate(last_message, Some(&specialist), max_tokens, temperature).await
            }
            RoutingDecision::Swarm { subtasks } => {
                tracing::info!("Swarm mode: {} subtasks", subtasks.len());
                let mut texts = Vec::new();
                let mut total_prompt = 0u32;
                let mut total_completion = 0u32;

                for task in &subtasks {
                    let prompt = format!("Task: {}\n\nContext: {last_message}", task.description);
                    let result = engine.generate(
                        &prompt,
                        Some(&task.specialist),
                        max_tokens / subtasks.len() as u32,
                        temperature,
                    ).await?;
                    total_prompt += result.prompt_tokens;
                    total_completion += result.completion_tokens;
                    texts.push((task.specialist.clone(), result.text));
                }

                let merged = self.synthesizer.merge(&texts)?;
                Ok(GenerationResult {
                    text: merged,
                    prompt_tokens: total_prompt,
                    completion_tokens: total_completion,
                    total_tokens: total_prompt + total_completion,
                    tok_per_sec: 0.0,
                    duration_ms: 0,
                })
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
