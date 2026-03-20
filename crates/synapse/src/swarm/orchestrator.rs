use anyhow::Result;
use futures::future::join_all;
use crate::config::SynapseConfig;
use crate::inference::{InferenceEngine, GenerationResult};
use crate::learn::CloudFallback;
use crate::memory::KnowledgeGraph;
use crate::openai::Message;
use super::coordinator::Coordinator;
use super::synthesizer::Synthesizer;

/// Top-level swarm orchestrator — decides single vs multi-specialist routing
/// Uses Hebbian routing and parallel specialist execution for swarm mode
/// Cloud fallback: when confidence is low, routes to cloud and learns from the response
pub struct Orchestrator {
    coordinator: Coordinator,
    synthesizer: Synthesizer,
    cloud_fallback: Option<CloudFallback>,
}

impl Orchestrator {
    pub fn new(config: &SynapseConfig) -> Self {
        Self {
            coordinator: Coordinator::new(config),
            synthesizer: Synthesizer::new(),
            cloud_fallback: CloudFallback::new(&config.cloud),
        }
    }

    /// Build context from full message history (not just last message)
    fn build_context(messages: &[Message]) -> String {
        if messages.len() <= 1 {
            return messages.last().map(|m| m.content.clone()).unwrap_or_default();
        }

        // Include recent conversation context (last 4 messages max)
        let recent: Vec<&Message> = messages.iter().rev().take(4).collect::<Vec<_>>().into_iter().rev().collect();
        let mut context = String::new();
        for msg in &recent[..recent.len().saturating_sub(1)] {
            context.push_str(&format!("[{}]: {}\n", msg.role, msg.content));
        }
        // Last message is the actual query
        if let Some(last) = recent.last() {
            context.push_str(&last.content);
        }
        context
    }

    /// Process a chat request — route to specialist(s) and return response
    pub async fn process(
        &self,
        messages: &[Message],
        engine: &InferenceEngine,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        knowledge: Option<&KnowledgeGraph>,
    ) -> Result<GenerationResult> {
        let last_message = messages.last()
            .map(|m| m.content.as_str())
            .unwrap_or("");

        let context = Self::build_context(messages);
        let max_tokens = max_tokens.unwrap_or(2048);
        let temperature = temperature.unwrap_or(0.7);

        let routing = self.coordinator.route(last_message, knowledge);

        match routing {
            RoutingDecision::Single { specialist, confidence } => {
                tracing::info!("Routing to specialist: {specialist} (confidence: {confidence:.2})");

                // Cloud fallback: if confidence is too low and cloud is available,
                // generate locally first, then ask cloud and learn from the difference
                let cloud_threshold = CloudFallback::confidence_threshold();
                if confidence < cloud_threshold {
                    if let Some(ref fallback) = self.cloud_fallback {
                        tracing::info!(
                            "⚡ Low confidence ({confidence:.2} < {cloud_threshold:.2}) — trying cloud fallback"
                        );

                        // Try local generation first (we still want the local attempt for DPO)
                        let local_result = engine.generate(&context, Some(&specialist), max_tokens, temperature).await;
                        let local_text = local_result.as_ref().ok().map(|r| r.text.as_str());

                        // Ask cloud for the better answer
                        if let Some(kg) = knowledge {
                            match fallback.fallback(last_message, &specialist, local_text, kg).await {
                                Ok(cloud_result) => {
                                    tracing::info!(
                                        "☁️ Cloud fallback used {}, learned={}",
                                        cloud_result.model_used, cloud_result.learned
                                    );
                                    // Return the cloud's better response
                                    return Ok(GenerationResult {
                                        text: cloud_result.text,
                                        prompt_tokens: 0,
                                        completion_tokens: 0,
                                        total_tokens: 0,
                                        tok_per_sec: 0.0,
                                        duration_ms: 0,
                                    });
                                }
                                Err(e) => {
                                    tracing::warn!("Cloud fallback failed: {e}, using local response");
                                    // Fall through to local response
                                }
                            }
                        }

                        // Cloud failed, return local result if we have one
                        if let Ok(result) = local_result {
                            return Ok(result);
                        }
                    }
                }

                let result = engine.generate(&context, Some(&specialist), max_tokens, temperature).await?;

                // Reinforce the pathway on successful generation
                if let Some(kg) = knowledge {
                    let _ = kg.reinforce_pathway(&[specialist.clone()], confidence);
                    let _ = kg.update_specialist_stats(
                        &specialist, "general", confidence, result.tok_per_sec,
                    );
                }

                Ok(result)
            }
            RoutingDecision::Swarm { subtasks } => {
                tracing::info!("⚡ Swarm mode: {} subtasks (PARALLEL)", subtasks.len());
                let start = std::time::Instant::now();
                let tokens_per_task = max_tokens / subtasks.len() as u32;

                // Execute ALL subtasks in parallel
                let futures: Vec<_> = subtasks.iter().map(|task| {
                    let prompt = format!("Task: {}\n\nContext: {context}", task.description);
                    let specialist = task.specialist.clone();
                    async move {
                        let result = engine.generate(
                            &prompt,
                            Some(&specialist),
                            tokens_per_task,
                            temperature,
                        ).await;
                        (specialist, result)
                    }
                }).collect();

                let results = join_all(futures).await;

                let mut texts = Vec::new();
                let mut total_prompt = 0u32;
                let mut total_completion = 0u32;
                let mut specialists_used = Vec::new();

                for (specialist, result) in results {
                    match result {
                        Ok(gen_result) => {
                            total_prompt += gen_result.prompt_tokens;
                            total_completion += gen_result.completion_tokens;
                            specialists_used.push(specialist.clone());
                            texts.push((specialist, gen_result.text));
                        }
                        Err(e) => {
                            tracing::warn!("Specialist {specialist} failed: {e}");
                            // Continue with other specialists — graceful degradation
                        }
                    }
                }

                if texts.is_empty() {
                    return Err(anyhow::anyhow!("All specialists failed in swarm mode"));
                }

                let elapsed = start.elapsed();
                let merged = self.synthesizer.merge(&texts)?;

                // Reinforce the swarm pathway
                if let Some(kg) = knowledge {
                    let _ = kg.reinforce_pathway(&specialists_used, 4.0);
                }

                let tok_per_sec = if elapsed.as_secs_f64() > 0.0 {
                    total_completion as f64 / elapsed.as_secs_f64()
                } else {
                    0.0
                };

                tracing::info!(
                    "⚡ Swarm complete: {} specialists, {} tokens in {:.1}s ({:.1} tok/s)",
                    specialists_used.len(), total_completion, elapsed.as_secs_f64(), tok_per_sec
                );

                Ok(GenerationResult {
                    text: merged,
                    prompt_tokens: total_prompt,
                    completion_tokens: total_completion,
                    total_tokens: total_prompt + total_completion,
                    tok_per_sec,
                    duration_ms: elapsed.as_millis() as u64,
                })
            }
        }
    }
}

pub enum RoutingDecision {
    Single { specialist: String, confidence: f32 },
    Swarm { subtasks: Vec<SubTask> },
}

pub struct SubTask {
    pub specialist: String,
    pub description: String,
}

impl std::fmt::Display for RoutingDecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RoutingDecision::Single { specialist, confidence } => write!(f, "Single({specialist}, confidence={confidence:.2})"),
            RoutingDecision::Swarm { subtasks } => {
                write!(f, "Swarm({})", subtasks.iter().map(|t| t.specialist.as_str()).collect::<Vec<_>>().join(", "))
            }
        }
    }
}
