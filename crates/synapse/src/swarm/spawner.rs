use anyhow::Result;
use crate::memory::KnowledgeGraph;
use crate::config::SpecialistConfig;

/// Specialist Auto-Spawner — the system that creates new brain regions.
///
/// When the coordinator repeatedly routes to "general" for a domain,
/// and the confidence is low, the spawner:
/// 1. Detects the pattern ("many Python questions going to general")
/// 2. Creates a new specialist config for that domain
/// 3. Queues training data collection
/// 4. Once enough data: triggers QLoRA training
/// 5. New specialist joins the swarm automatically
///
/// The system literally grows new specialists as needed.
/// A music producer will end up with audio_expert, midi_expert, mixing_expert.
/// A data scientist will get pandas_expert, sklearn_expert, visualization_expert.
/// No configuration needed. The system figures it out.
pub struct SpecialistSpawner {
    /// Minimum requests in a domain before considering spawning
    min_requests: u32,
    /// Maximum confidence score that triggers spawning (below this = specialist needed)
    confidence_threshold: f32,
    /// Domains that already have specialists (don't spawn duplicates)
    covered_domains: Vec<String>,
}

/// A proposal for a new specialist
#[derive(Debug, Clone)]
pub struct SpawnProposal {
    pub name: String,
    pub domain: String,
    pub capabilities: Vec<String>,
    pub reason: String,
    pub requests_in_domain: u32,
    pub current_avg_score: f64,
}

impl SpecialistSpawner {
    pub fn new(covered_domains: Vec<String>) -> Self {
        Self {
            min_requests: 5,
            confidence_threshold: 3.0,
            covered_domains,
        }
    }

    /// Analyze the knowledge graph for domains that need specialists
    pub fn detect_spawn_candidates(&self, kg: &KnowledgeGraph) -> Result<Vec<SpawnProposal>> {
        let mut proposals = Vec::new();

        // Get all specialist stats
        let stats = kg.specialist_confidence_report().unwrap_or_default();

        // Look for domains where "general" is handling too many requests with low scores
        for stat in &stats {
            let specialist = stat["specialist"].as_str().unwrap_or("");
            let domain = stat["domain"].as_str().unwrap_or("");
            let requests = stat["requests"].as_u64().unwrap_or(0) as u32;
            let avg_score = stat["avg_score"].as_f64().unwrap_or(0.0);

            // If general specialist is handling many requests in a specific domain
            // with below-threshold scores, propose a new specialist
            if specialist == "general"
                && requests >= self.min_requests
                && avg_score < self.confidence_threshold as f64
                && !self.is_domain_covered(domain)
            {
                let capabilities = Self::infer_capabilities(domain);
                let name = format!("{}_expert", domain.replace(' ', "_"));

                proposals.push(SpawnProposal {
                    name,
                    domain: domain.to_string(),
                    capabilities,
                    reason: format!(
                        "General specialist handling {} requests in '{}' domain with avg score {:.1} (below threshold {:.1})",
                        requests, domain, avg_score, self.confidence_threshold
                    ),
                    requests_in_domain: requests,
                    current_avg_score: avg_score,
                });
            }
        }

        // Also analyze conversation patterns for undetected domains
        if let Ok(top_pathways) = kg.top_pathways(20) {
            for (pathway, strength, avg_score) in &top_pathways {
                if *strength > 3 && *avg_score < self.confidence_threshold as f64 {
                    // This pathway is used often but scoring low
                    if !self.is_domain_covered(pathway) {
                        proposals.push(SpawnProposal {
                            name: format!("{}_expert", pathway.replace('+', "_")),
                            domain: pathway.clone(),
                            capabilities: vec![pathway.clone()],
                            reason: format!(
                                "Pathway '{}' reinforced {} times but avg score only {:.1}",
                                pathway, strength, avg_score
                            ),
                            requests_in_domain: *strength as u32,
                            current_avg_score: *avg_score,
                        });
                    }
                }
            }
        }

        Ok(proposals)
    }

    /// Convert a spawn proposal into a specialist config
    pub fn create_specialist_config(proposal: &SpawnProposal) -> SpecialistConfig {
        SpecialistConfig {
            name: proposal.name.clone(),
            capabilities: proposal.capabilities.clone(),
            base_model: None, // Use default base model
            adapter: None,    // Will be trained
            system_prompt: Some(format!(
                "You are an expert in {}. Provide detailed, accurate answers in your domain of expertise.",
                proposal.domain
            )),
            priority: 70, // Higher than general (50) but lower than existing experts
        }
    }

    fn is_domain_covered(&self, domain: &str) -> bool {
        self.covered_domains.iter().any(|d| {
            d.to_lowercase().contains(&domain.to_lowercase())
                || domain.to_lowercase().contains(&d.to_lowercase())
        })
    }

    fn infer_capabilities(domain: &str) -> Vec<String> {
        let domain_lower = domain.to_lowercase();
        let mut caps = vec![domain.to_string()];

        // Add related capabilities based on domain
        let related: Vec<(&str, &[&str])> = vec![
            ("python", &["debugging", "testing", "django", "flask", "fastapi"]),
            ("javascript", &["react", "node", "typescript", "frontend"]),
            ("sql", &["database", "query", "postgres", "mysql"]),
            ("rust", &["systems", "memory", "concurrency", "cargo"]),
            ("math", &["algebra", "calculus", "statistics", "probability"]),
            ("science", &["physics", "chemistry", "biology"]),
            ("writing", &["grammar", "style", "creative", "editing"]),
            ("music", &["audio", "production", "mixing", "midi"]),
            ("business", &["finance", "marketing", "strategy", "management"]),
        ];

        for (key, related_caps) in &related {
            if domain_lower.contains(key) {
                caps.extend(related_caps.iter().map(|s| s.to_string()));
                break;
            }
        }

        caps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infer_capabilities() {
        let caps = SpecialistSpawner::infer_capabilities("python");
        assert!(caps.contains(&"python".to_string()));
        assert!(caps.contains(&"debugging".to_string()));
    }

    #[test]
    fn test_is_domain_covered() {
        let spawner = SpecialistSpawner::new(vec![
            "python_expert".into(),
            "sql_expert".into(),
        ]);
        assert!(spawner.is_domain_covered("python"));
        assert!(spawner.is_domain_covered("sql"));
        assert!(!spawner.is_domain_covered("music"));
    }

    #[test]
    fn test_create_specialist_config() {
        let proposal = SpawnProposal {
            name: "music_expert".into(),
            domain: "music production".into(),
            capabilities: vec!["music".into(), "audio".into()],
            reason: "test".into(),
            requests_in_domain: 10,
            current_avg_score: 2.5,
        };

        let config = SpecialistSpawner::create_specialist_config(&proposal);
        assert_eq!(config.name, "music_expert");
        assert_eq!(config.priority, 70);
        assert!(config.system_prompt.unwrap().contains("music production"));
    }
}
