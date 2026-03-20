use std::time::SystemTime;

/// Token sampling strategies
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub repetition_penalty: f32,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repetition_penalty: 1.1,
        }
    }
}

/// Simple fast RNG (xorshift64) — no external deps needed
struct FastRng(u64);

impl FastRng {
    fn new() -> Self {
        let seed = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        Self(seed | 1) // ensure non-zero
    }

    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 as f32) / (u64::MAX as f32)
    }
}

impl SamplerConfig {
    /// Sample a token from logits
    pub fn sample(&self, logits: &[f32]) -> u32 {
        if logits.is_empty() {
            return 0;
        }

        // Greedy mode
        if self.temperature <= 0.0 {
            return logits.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
        }

        // Temperature scaling
        let scaled: Vec<f32> = logits.iter().map(|&l| l / self.temperature).collect();

        // Softmax
        let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();

        // Top-k filtering
        let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.truncate(self.top_k as usize);

        // Top-p (nucleus) filtering
        let mut cumsum = 0.0;
        let mut filtered = Vec::new();
        for (idx, prob) in &indexed {
            cumsum += prob;
            filtered.push((*idx, *prob));
            if cumsum >= self.top_p {
                break;
            }
        }

        // Renormalize
        let total: f32 = filtered.iter().map(|(_, p)| p).sum();
        let normalized: Vec<(usize, f32)> = filtered.iter()
            .map(|(i, p)| (*i, p / total))
            .collect();

        // Random sampling
        let mut rng = FastRng::new();
        let r = rng.next_f32();
        let mut cumulative = 0.0;
        for (idx, prob) in &normalized {
            cumulative += prob;
            if r < cumulative {
                return *idx as u32;
            }
        }

        // Fallback to top token
        normalized.first()
            .map(|(i, _)| *i as u32)
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_sampling() {
        let sampler = SamplerConfig { temperature: 0.0, ..Default::default() };
        let logits = vec![0.1, 0.5, 0.3, 0.8, 0.2];
        assert_eq!(sampler.sample(&logits), 3); // argmax
    }

    #[test]
    fn test_empty_logits() {
        let sampler = SamplerConfig::default();
        assert_eq!(sampler.sample(&[]), 0);
    }

    #[test]
    fn test_stochastic_sampling() {
        let sampler = SamplerConfig { temperature: 1.0, ..Default::default() };
        let logits = vec![0.1, 0.5, 0.3, 0.8, 0.2];
        // Should return a valid token index
        let token = sampler.sample(&logits);
        assert!(token < 5);
    }
}
