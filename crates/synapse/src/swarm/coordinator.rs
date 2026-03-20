use crate::config::SynapseConfig;
use super::orchestrator::{RoutingDecision, SubTask};

/// Coordinator — routes requests to the right specialist(s)
/// Uses Hebbian routing: pathways that fire together, wire together
pub struct Coordinator {
    /// Keyword → specialist mapping (will be replaced by learned routing)
    keyword_routes: Vec<(Vec<String>, String)>,
    /// Hebbian pathway strengths (specialist_combo → success_count)
    pathway_strengths: std::collections::HashMap<String, u32>,
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
            pathway_strengths: std::collections::HashMap::new(),
        }
    }

    /// Route a query to the appropriate specialist(s)
    pub fn route(&self, query: &str) -> RoutingDecision {
        let query_lower = query.to_lowercase();
        let words: Vec<&str> = query_lower.split_whitespace().collect();

        // Score each specialist based on keyword matches
        let mut scores: Vec<(String, u32)> = self.keyword_routes.iter()
            .map(|(keywords, name)| {
                let score = keywords.iter()
                    .filter(|kw| words.iter().any(|w| w.contains(kw.as_str())))
                    .count() as u32;
                (name.clone(), score)
            })
            .filter(|(_, score)| *score > 0)
            .collect();

        scores.sort_by(|a, b| b.1.cmp(&a.1));

        // Detect complexity indicators for swarm mode
        let complexity_keywords = ["and", "also", "plus", "then", "after", "build", "create", "implement"];
        let complexity = complexity_keywords.iter()
            .filter(|kw| query_lower.contains(*kw))
            .count();

        if complexity >= 2 && scores.len() >= 2 {
            // Complex query — use swarm
            let subtasks: Vec<SubTask> = scores.iter()
                .take(3)
                .map(|(specialist, _)| SubTask {
                    specialist: specialist.clone(),
                    description: format!("Handle {specialist} aspects of: {query}"),
                })
                .collect();

            RoutingDecision::Swarm { subtasks }
        } else if let Some((specialist, _)) = scores.first() {
            RoutingDecision::Single { specialist: specialist.clone() }
        } else {
            // Default to general specialist
            RoutingDecision::Single { specialist: "general".into() }
        }
    }

    /// Record a successful pathway (Hebbian learning)
    pub fn reinforce_pathway(&mut self, specialists: &[String]) {
        let key = specialists.join("+");
        *self.pathway_strengths.entry(key).or_insert(0) += 1;
    }
}
