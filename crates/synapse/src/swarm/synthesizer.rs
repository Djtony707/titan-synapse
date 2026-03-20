/// Merges outputs from multiple specialists into a coherent response
pub struct Synthesizer;

impl Synthesizer {
    pub fn new() -> Self { Self }

    /// Merge multiple specialist responses into one
    pub fn merge(&self, results: &[(String, String)]) -> anyhow::Result<String> {
        if results.is_empty() {
            return Ok("No specialist responses to merge.".into());
        }

        if results.len() == 1 {
            return Ok(results[0].1.clone());
        }

        // For now, concatenate with specialist attribution
        // In production, this will use the coordinator model to synthesize
        let mut output = String::new();
        for (specialist, response) in results {
            output.push_str(&format!("**[{specialist}]**\n{response}\n\n"));
        }

        Ok(output.trim().to_string())
    }
}
