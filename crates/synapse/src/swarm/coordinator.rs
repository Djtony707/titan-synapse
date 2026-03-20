use crate::config::SynapseConfig;
use crate::memory::KnowledgeGraph;
use super::orchestrator::{RoutingDecision, SubTask};

/// Coordinator — routes requests to the right specialist(s)
/// Uses Hebbian routing: pathways that fire together, wire together
/// Includes metacognitive confidence scoring
pub struct Coordinator {
    /// Keyword → specialist mapping (will be replaced by learned routing)
    keyword_routes: Vec<(Vec<String>, String)>,
}

impl Coordinator {
    pub fn new(config: &SynapseConfig) -> Self {
        let mut keyword_routes = Vec::new();

        for specialist in &config.specialists {
            keyword_routes.push((
                specialist.capabilities.clone(),
                specialist.name.clone(),
            ));
        }

        Self {
            keyword_routes,
        }
    }

    /// Route a query to the appropriate specialist(s)
    /// Returns routing decision with metacognitive confidence score
    pub fn route(&self, query: &str, knowledge: Option<&KnowledgeGraph>) -> RoutingDecision {
        let query_lower = query.to_lowercase();
        let words: Vec<&str> = query_lower.split_whitespace().collect();

        // Score each specialist based on keyword matches
        let mut scores: Vec<(String, f32)> = self.keyword_routes.iter()
            .map(|(keywords, name)| {
                let keyword_matches = keywords.iter()
                    .filter(|kw| words.iter().any(|w| w.contains(kw.as_str())))
                    .count() as f32;

                // Normalize by total keywords — more specific matches = higher confidence
                let keyword_ratio = if keywords.is_empty() {
                    0.0
                } else {
                    keyword_matches / keywords.len() as f32
                };

                // Base confidence: keyword match ratio (0.0 - 1.0)
                let confidence = keyword_matches + keyword_ratio * 2.0;
                (name.clone(), confidence)
            })
            .filter(|(_, score)| *score > 0.0)
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Boost scores using Hebbian pathway strengths from the knowledge graph
        if let Some(kg) = knowledge {
            for (name, score) in &mut scores {
                let pathway = vec![name.clone()];
                if let Ok(strength) = kg.pathway_strength(&pathway) {
                    // Add pathway strength as bonus (clamped to reasonable range)
                    *score += (strength.min(10.0) as f32) * 0.5;
                }
            }
            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }

        // Detect complexity indicators for swarm mode
        let complexity_keywords = ["and", "also", "plus", "then", "after", "build", "create", "implement"];
        let complexity = complexity_keywords.iter()
            .filter(|kw| query_lower.contains(*kw))
            .count();

        // Calculate confidence — how sure are we about the routing?
        let top_confidence = scores.first().map(|(_, s)| *s).unwrap_or(0.0);
        let second_confidence = scores.get(1).map(|(_, s)| *s).unwrap_or(0.0);
        let confidence_gap = if second_confidence > 0.0 {
            (top_confidence - second_confidence) / top_confidence
        } else if top_confidence > 0.0 {
            1.0  // Only one match — high confidence
        } else {
            0.0  // No matches — low confidence, use general
        };

        // Normalize to 0-5 scale for pathway reinforcement
        let routing_confidence = (top_confidence.min(5.0)).max(1.0);

        if complexity >= 2 && scores.len() >= 2 {
            // Complex query — use swarm with parallel execution
            let subtasks: Vec<SubTask> = scores.iter()
                .take(3)
                .map(|(specialist, _)| SubTask {
                    specialist: specialist.clone(),
                    description: format!("Handle {specialist} aspects of: {query}"),
                })
                .collect();

            RoutingDecision::Swarm { subtasks }
        } else if let Some((specialist, _)) = scores.first() {
            RoutingDecision::Single {
                specialist: specialist.clone(),
                confidence: routing_confidence,
            }
        } else {
            // Default to general specialist — low confidence
            RoutingDecision::Single {
                specialist: "general".into(),
                confidence: 1.0,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{SynapseConfig, SpecialistConfig};

    fn test_config() -> SynapseConfig {
        let mut config = SynapseConfig::default();
        config.specialists = vec![
            SpecialistConfig {
                name: "python_expert".into(),
                capabilities: vec!["python".into(), "decorator".into(), "django".into(), "flask".into()],
                base_model: None,
                adapter: None,
                system_prompt: None,
                priority: 60,
            },
            SpecialistConfig {
                name: "sql_expert".into(),
                capabilities: vec!["sql".into(), "database".into(), "query".into(), "postgres".into()],
                base_model: None,
                adapter: None,
                system_prompt: None,
                priority: 60,
            },
            SpecialistConfig {
                name: "devops_expert".into(),
                capabilities: vec!["docker".into(), "kubernetes".into(), "deploy".into(), "ci".into()],
                base_model: None,
                adapter: None,
                system_prompt: None,
                priority: 60,
            },
        ];
        config
    }

    #[test]
    fn test_single_routing() {
        let config = test_config();
        let coordinator = Coordinator::new(&config);

        let decision = coordinator.route("What is a Python decorator?", None);
        match decision {
            RoutingDecision::Single { specialist, confidence } => {
                assert_eq!(specialist, "python_expert");
                assert!(confidence >= 1.0);
            }
            _ => panic!("Expected single routing"),
        }
    }

    #[test]
    fn test_swarm_routing() {
        let config = test_config();
        let coordinator = Coordinator::new(&config);

        let decision = coordinator.route(
            "Build a Python API and deploy it with Docker and also create the database",
            None,
        );
        match decision {
            RoutingDecision::Swarm { subtasks } => {
                assert!(subtasks.len() >= 2, "Should route to multiple specialists");
            }
            _ => panic!("Expected swarm routing for complex query"),
        }
    }

    #[test]
    fn test_default_routing() {
        let config = test_config();
        let coordinator = Coordinator::new(&config);

        let decision = coordinator.route("What is the meaning of life?", None);
        match decision {
            RoutingDecision::Single { specialist, confidence } => {
                assert_eq!(specialist, "general");
                assert_eq!(confidence, 1.0);
            }
            _ => panic!("Expected default general routing"),
        }
    }
}
