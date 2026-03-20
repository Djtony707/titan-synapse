use anyhow::Result;
use super::graph::KnowledgeGraph;

/// Hallucination detector — cross-references model outputs against the knowledge graph.
/// If a model claims something that contradicts known facts, flag it.
/// If a model claims something new, check confidence before presenting it.
///
/// This is how you make tiny models smarter than 120B: you don't let them lie.
/// A 3B model that knows what it doesn't know > a 120B model that confidently bullshits.
pub struct HallucinationDetector;

#[derive(Debug)]
pub struct VerificationResult {
    /// Overall confidence score (0.0 - 1.0)
    pub confidence: f64,
    /// Claims that were verified against knowledge graph
    pub verified_claims: Vec<Claim>,
    /// Claims that contradict known facts
    pub contradictions: Vec<Claim>,
    /// Claims that couldn't be verified (might be hallucination)
    pub unverified_claims: Vec<Claim>,
    /// Whether the response should be flagged
    pub flagged: bool,
}

#[derive(Debug)]
pub struct Claim {
    pub text: String,
    pub subject: String,
    pub status: ClaimStatus,
}

#[derive(Debug)]
pub enum ClaimStatus {
    /// Matches a known fact
    Verified,
    /// Contradicts a known fact
    Contradicted(String), // The known fact that contradicts
    /// No matching fact found
    Unverified,
}

impl HallucinationDetector {
    /// Check a response against the knowledge graph
    pub fn verify(kg: &KnowledgeGraph, response: &str) -> Result<VerificationResult> {
        let mut verified = Vec::new();
        let mut contradictions = Vec::new();
        let mut unverified = Vec::new();

        // Extract potential claims from the response
        let claims = Self::extract_claims(response);

        for claim in &claims {
            // Check if we have any facts about this subject
            match kg.query_facts(&claim.subject) {
                Ok(facts) if !facts.is_empty() => {
                    // We know something about this subject
                    let claim_lower = claim.text.to_lowercase();
                    let mut found_match = false;
                    let mut found_contradiction = false;

                    for (predicate, object, confidence) in &facts {
                        let fact_text = format!("{} {}", predicate, object).to_lowercase();

                        // Simple semantic overlap check
                        let overlap = word_overlap(&claim_lower, &fact_text);

                        if overlap > 0.3 && *confidence > 0.5 {
                            found_match = true;
                        }

                        // Check for explicit contradictions
                        if contains_negation(&claim_lower, &fact_text) {
                            found_contradiction = true;
                            contradictions.push(Claim {
                                text: claim.text.clone(),
                                subject: claim.subject.clone(),
                                status: ClaimStatus::Contradicted(format!("{} {} {}", claim.subject, predicate, object)),
                            });
                        }
                    }

                    if found_match && !found_contradiction {
                        verified.push(Claim {
                            text: claim.text.clone(),
                            subject: claim.subject.clone(),
                            status: ClaimStatus::Verified,
                        });
                    } else if !found_contradiction {
                        unverified.push(Claim {
                            text: claim.text.clone(),
                            subject: claim.subject.clone(),
                            status: ClaimStatus::Unverified,
                        });
                    }
                }
                _ => {
                    // No facts about this subject — can't verify
                    unverified.push(Claim {
                        text: claim.text.clone(),
                        subject: claim.subject.clone(),
                        status: ClaimStatus::Unverified,
                    });
                }
            }
        }

        let total = verified.len() + contradictions.len() + unverified.len();
        let confidence = if total > 0 {
            (verified.len() as f64 / total as f64).max(0.1)
        } else {
            0.5 // No claims to verify
        };

        let flagged = !contradictions.is_empty() || (unverified.len() > verified.len() * 2);

        Ok(VerificationResult {
            confidence,
            verified_claims: verified,
            contradictions,
            unverified_claims: unverified,
            flagged,
        })
    }

    /// Extract potential factual claims from text
    fn extract_claims(text: &str) -> Vec<SimpleClaim> {
        let mut claims = Vec::new();

        for sentence in text.split(['.', '!', '\n']) {
            let sentence = sentence.trim();
            if sentence.len() < 10 || sentence.len() > 500 {
                continue;
            }

            // Skip questions, code blocks, instructions
            if sentence.starts_with('?') || sentence.starts_with("```")
                || sentence.starts_with('#') || sentence.starts_with("//")
            {
                continue;
            }

            let lower = sentence.to_lowercase();

            // Look for definitive statements ("X is Y", "X was Y", "X has Y")
            let definitive_patterns = [" is ", " was ", " are ", " were ", " has ", " have "];
            for pattern in &definitive_patterns {
                if let Some(pos) = lower.find(pattern) {
                    if pos > 2 {
                        let subject_words: Vec<&str> = lower[..pos].split_whitespace().collect();
                        let subject = if subject_words.len() > 3 {
                            subject_words[subject_words.len()-3..].join(" ")
                        } else {
                            subject_words.join(" ")
                        };

                        if !subject.is_empty() {
                            claims.push(SimpleClaim {
                                text: sentence.to_string(),
                                subject: capitalize(&subject),
                            });
                            break; // One claim per sentence
                        }
                    }
                }
            }
        }

        claims
    }
}

struct SimpleClaim {
    text: String,
    subject: String,
}

/// Calculate word overlap ratio between two strings
fn word_overlap(a: &str, b: &str) -> f64 {
    let a_words: std::collections::HashSet<&str> = a.split_whitespace().collect();
    let b_words: std::collections::HashSet<&str> = b.split_whitespace().collect();

    if a_words.is_empty() || b_words.is_empty() {
        return 0.0;
    }

    let overlap = a_words.intersection(&b_words).count();
    let max_len = a_words.len().max(b_words.len());
    overlap as f64 / max_len as f64
}

/// Check if one text negates or contradicts the other
fn contains_negation(claim: &str, fact: &str) -> bool {
    let negation_words = ["not", "isn't", "aren't", "wasn't", "weren't", "never", "neither", "nor"];

    // If claim has a negation word and fact doesn't (or vice versa), possible contradiction
    let claim_negated = negation_words.iter().any(|neg| claim.contains(neg));
    let fact_negated = negation_words.iter().any(|neg| fact.contains(neg));

    // Simple: if one is negated and the other isn't, and they're about the same thing
    claim_negated != fact_negated && word_overlap(claim, fact) > 0.2
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

    fn test_kg_with_facts() -> KnowledgeGraph {
        let tmp = tempdir().unwrap();
        let kg = KnowledgeGraph::new(&tmp.path().join("test.db")).unwrap();

        // Add some known facts
        kg.add_fact("Python", "is_a", "programming language", Some("test")).unwrap();
        kg.add_fact("Python", "created_by", "Guido van Rossum", Some("test")).unwrap();
        kg.add_fact("Rust", "is_a", "systems programming language", Some("test")).unwrap();
        kg.add_fact("Rust", "created_by", "Mozilla", Some("test")).unwrap();

        kg
    }

    #[test]
    fn test_verify_correct_claim() {
        let kg = test_kg_with_facts();
        let response = "Python is a programming language that is widely used.";
        let result = HallucinationDetector::verify(&kg, response).unwrap();
        assert!(result.contradictions.is_empty(), "Should not flag correct claims");
    }

    #[test]
    fn test_verify_unknown_claim() {
        let kg = test_kg_with_facts();
        let response = "JavaScript was invented in 1995 by Brendan Eich.";
        let result = HallucinationDetector::verify(&kg, response).unwrap();
        // We don't know about JavaScript, so it should be unverified
        assert!(result.contradictions.is_empty());
    }

    #[test]
    fn test_word_overlap() {
        assert!(word_overlap("python is great", "python is good") > 0.3);
        assert!(word_overlap("rust memory safe", "java garbage collection") < 0.1);
    }

    #[test]
    fn test_empty_response() {
        let kg = test_kg_with_facts();
        let result = HallucinationDetector::verify(&kg, "").unwrap();
        assert!(!result.flagged);
    }
}
