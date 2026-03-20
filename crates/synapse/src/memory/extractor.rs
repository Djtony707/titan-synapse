use anyhow::Result;
use super::graph::KnowledgeGraph;

/// Extracts structured knowledge from conversations in real-time.
/// This is what makes "learns from every conversation" real — not just logging,
/// but actually building a queryable knowledge graph from natural language.
pub struct KnowledgeExtractor;

impl KnowledgeExtractor {
    /// Extract facts from a conversation message and store in the knowledge graph.
    /// Uses pattern matching for common fact patterns:
    /// - "X is Y" → (X, is_a, Y)
    /// - "X was created by Y" → (X, created_by, Y)
    /// - "X uses Y" → (X, uses, Y)
    /// - "X runs on Y" → (X, runs_on, Y)
    /// - "X supports Y" → (X, supports, Y)
    pub fn extract_and_store(kg: &KnowledgeGraph, text: &str, source: &str) -> Result<u32> {
        let mut facts_added = 0;
        let sentences: Vec<&str> = text.split(['.', '!', '\n'])
            .map(|s| s.trim())
            .filter(|s| s.len() > 5 && s.len() < 500)
            .collect();

        for sentence in &sentences {
            let lower = sentence.to_lowercase();

            // Pattern: "X is a/an Y"
            if let Some(fact) = extract_is_pattern(&lower, sentence) {
                kg.add_fact(&fact.0, &fact.1, &fact.2, Some(source))?;
                facts_added += 1;
            }

            // Pattern: "X uses/runs/supports Y"
            for (keyword, predicate) in &[
                ("uses", "uses"),
                ("runs on", "runs_on"),
                ("supports", "supports"),
                ("requires", "requires"),
                ("depends on", "depends_on"),
                ("created by", "created_by"),
                ("built with", "built_with"),
                ("written in", "written_in"),
            ] {
                if let Some(fact) = extract_verb_pattern(&lower, sentence, keyword, predicate) {
                    kg.add_fact(&fact.0, &fact.1, &fact.2, Some(source))?;
                    facts_added += 1;
                }
            }
        }

        if facts_added > 0 {
            tracing::debug!("Extracted {facts_added} facts from conversation");
        }

        Ok(facts_added)
    }

    /// Extract user preferences from conversation patterns
    pub fn extract_preferences(kg: &KnowledgeGraph, user_msg: &str, assistant_msg: &str, specialist: &str) -> Result<()> {
        // If user says "good", "thanks", "correct", "exactly" — positive signal
        let positive_signals = ["good", "thanks", "correct", "exactly", "perfect", "great", "nice"];
        let negative_signals = ["wrong", "incorrect", "no,", "that's not", "actually,", "nope"];

        let user_lower = user_msg.to_lowercase();

        let is_positive = positive_signals.iter().any(|s| user_lower.contains(s));
        let is_negative = negative_signals.iter().any(|s| user_lower.contains(s));

        if is_positive && !is_negative {
            // This was a good response — could be used as chosen in DPO
            tracing::debug!("Positive feedback detected for {specialist}");
        } else if is_negative && !is_positive {
            // This was a bad response — store for improvement
            // Store as a preference pair with placeholder improved response
            kg.add_preference(specialist, user_msg, "(needs improvement)", assistant_msg)?;
            tracing::debug!("Negative feedback detected for {specialist} — stored preference pair");
        }

        Ok(())
    }
}

/// Extract "X is a/an Y" patterns
fn extract_is_pattern(lower: &str, _original: &str) -> Option<(String, String, String)> {
    // Look for "X is a Y" or "X is an Y"
    let patterns = [" is a ", " is an ", " is the "];
    for pattern in patterns {
        if let Some(pos) = lower.find(pattern) {
            let subject = lower[..pos].trim();
            let object = lower[pos + pattern.len()..].trim();

            // Only extract if both parts are reasonable length
            if subject.len() > 1 && subject.len() < 100 && object.len() > 1 && object.len() < 200 {
                // Clean up subject (take last noun phrase)
                let subject = subject.split_whitespace().collect::<Vec<_>>();
                let subject = if subject.len() > 3 {
                    subject[subject.len()-3..].join(" ")
                } else {
                    subject.join(" ")
                };

                return Some((
                    capitalize(&subject),
                    "is_a".to_string(),
                    object.to_string(),
                ));
            }
        }
    }
    None
}

/// Extract "X verb Y" patterns
fn extract_verb_pattern(lower: &str, _original: &str, keyword: &str, predicate: &str) -> Option<(String, String, String)> {
    if let Some(pos) = lower.find(keyword) {
        let subject = lower[..pos].trim();
        let object = lower[pos + keyword.len()..].trim();

        if subject.len() > 1 && subject.len() < 100 && object.len() > 1 && object.len() < 200 {
            let subject = subject.split_whitespace().collect::<Vec<_>>();
            let subject = if subject.len() > 3 {
                subject[subject.len()-3..].join(" ")
            } else {
                subject.join(" ")
            };

            return Some((
                capitalize(&subject),
                predicate.to_string(),
                object.to_string(),
            ));
        }
    }
    None
}

fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use std::path::Path;

    fn test_kg() -> KnowledgeGraph {
        let tmp = tempdir().unwrap();
        KnowledgeGraph::new(&tmp.path().join("test.db")).unwrap()
    }

    #[test]
    fn test_extract_is_pattern() {
        let kg = test_kg();
        let text = "Python is a programming language. It was designed by Guido van Rossum.";
        let count = KnowledgeExtractor::extract_and_store(&kg, text, "test").unwrap();
        assert!(count >= 1, "Should extract at least 1 fact, got {count}");

        let facts = kg.query_facts("Python").unwrap();
        assert!(!facts.is_empty(), "Should have facts about Python");
    }

    #[test]
    fn test_extract_verb_patterns() {
        let kg = test_kg();
        let text = "TITAN Synapse uses Rust for the inference engine. The project runs on CUDA GPUs.";
        let count = KnowledgeExtractor::extract_and_store(&kg, text, "test").unwrap();
        assert!(count >= 1, "Should extract at least 1 fact, got {count}");
    }

    #[test]
    fn test_extract_preferences_positive() {
        let kg = test_kg();
        KnowledgeExtractor::extract_preferences(
            &kg, "Thanks, that's correct!", "Python is dynamically typed.", "python_expert"
        ).unwrap();
        // Positive feedback shouldn't create a preference pair
        assert_eq!(kg.preference_count("python_expert").unwrap(), 0);
    }

    #[test]
    fn test_extract_preferences_negative() {
        let kg = test_kg();
        KnowledgeExtractor::extract_preferences(
            &kg, "No, that's not right at all", "Python is statically typed.", "python_expert"
        ).unwrap();
        // Negative feedback should create a preference pair
        assert_eq!(kg.preference_count("python_expert").unwrap(), 1);
    }

    #[test]
    fn test_empty_text() {
        let kg = test_kg();
        let count = KnowledgeExtractor::extract_and_store(&kg, "", "test").unwrap();
        assert_eq!(count, 0);
    }
}
