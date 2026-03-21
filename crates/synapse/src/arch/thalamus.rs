//! Thalamus — Brain-Inspired Router
//!
//! Named after the brain's thalamus, which routes sensory information
//! to the appropriate cortical regions. This module decides which
//! specialist modules should process each token.
//!
//! Uses a Mamba backbone (O(n) state-space model) to analyze the token
//! stream and produce routing weights. Only the top-k specialists
//! activate per token — the rest stay dormant (sparse activation).
//!
//! OBSERVABILITY: Every routing decision is logged. You can see exactly
//! which modules fired, why, and how confident the router was.
//! No black box — full transparency into the model's decision-making.

use anyhow::Result;
use candle_core::{Device, Tensor, DType, D};

use super::mamba::{MambaConfig, MambaLayer};
use super::linear_bias;

/// Configuration for the Thalamus router
#[derive(Debug, Clone)]
pub struct ThalamusConfig {
    /// Input token dimension
    pub d_model: usize,
    /// Number of specialist modules to route to
    pub n_experts: usize,
    /// Number of experts to activate per token (top-k)
    pub top_k: usize,
    /// Mamba state-space dimension for the router backbone
    pub d_state: usize,
    /// Noise to add during training for load balancing
    pub noise_std: f32,
    /// Enable Hebbian pathway strengthening
    pub hebbian_learning: bool,
    /// Hebbian learning rate (how fast pathways strengthen)
    pub hebbian_lr: f32,
    /// Device
    pub device: Device,
}

impl Default for ThalamusConfig {
    fn default() -> Self {
        Self {
            d_model: 768,
            n_experts: 8,
            top_k: 2,
            d_state: 16,
            noise_std: 0.1,
            hebbian_learning: true,
            hebbian_lr: 0.01,
            device: Device::Cpu,
        }
    }
}

/// A single routing decision — fully transparent
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    /// Which experts were selected (indices)
    pub selected_experts: Vec<usize>,
    /// Routing weights for selected experts (sum to ~1.0)
    pub expert_weights: Vec<f32>,
    /// Raw routing scores for ALL experts (before top-k selection)
    pub all_scores: Vec<f32>,
    /// Confidence: max weight / second max weight (higher = more certain)
    pub confidence: f32,
    /// Load balance: how evenly distributed selections are across experts
    pub load_balance: f32,
    /// Which pathway pattern this represents (e.g., "lang+reasoning")
    pub pathway: String,
}

/// Full introspection of a Thalamus forward pass
#[derive(Debug, Clone)]
pub struct ThalamusIntrospection {
    /// Per-token routing decisions
    pub decisions: Vec<RoutingDecision>,
    /// Expert activation counts over this batch
    pub expert_activation_counts: Vec<usize>,
    /// Expert names (for display)
    pub expert_names: Vec<String>,
    /// Average confidence across all tokens
    pub avg_confidence: f32,
    /// Load balance score (1.0 = perfectly balanced)
    pub load_balance_score: f32,
    /// Hebbian pathway strengths (top 10)
    pub top_pathways: Vec<(String, f32)>,
    /// Tokens routed per expert (percentage)
    pub expert_utilization: Vec<f32>,
}

/// The Thalamus — brain-inspired router using Mamba backbone
pub struct Thalamus {
    config: ThalamusConfig,

    /// Mamba backbone — processes token stream to understand context
    mamba: MambaLayer,

    /// Router projection: d_model → n_experts
    router_proj: Tensor,
    router_bias: Tensor,

    /// Expert names (for human-readable introspection)
    expert_names: Vec<String>,

    /// Hebbian pathway strengths: pathway_key → strength
    /// pathway_key = sorted expert indices joined by "+"
    hebbian_strengths: std::collections::HashMap<String, f32>,

    /// Running count of expert activations (for load balancing)
    activation_counts: Vec<usize>,

    /// Last introspection
    last_introspection: Option<ThalamusIntrospection>,
}

impl Thalamus {
    pub fn new(config: ThalamusConfig) -> Result<Self> {
        let dev = &config.device;

        // Mamba backbone for understanding context
        let mamba_config = MambaConfig {
            d_model: config.d_model,
            d_state: config.d_state,
            d_conv: 4,
            expand: 2,
            device: dev.clone(),
        };
        let mamba = MambaLayer::new(mamba_config)?;

        // Router head
        let scale = (1.0 / config.d_model as f64).sqrt() as f32;
        let router_proj = Tensor::randn(
            0f32, scale,
            &[config.n_experts, config.d_model], dev
        )?;
        let router_bias = Tensor::zeros(&[config.n_experts], DType::F32, dev)?;

        // Default expert names
        let expert_names: Vec<String> = (0..config.n_experts)
            .map(|i| match i {
                0 => "language".to_string(),
                1 => "reasoning".to_string(),
                2 => "code".to_string(),
                3 => "math".to_string(),
                4 => "memory".to_string(),
                5 => "creative".to_string(),
                6 => "analysis".to_string(),
                7 => "planning".to_string(),
                _ => format!("expert_{i}"),
            })
            .collect();

        Ok(Self {
            mamba,
            router_proj,
            router_bias,
            expert_names,
            hebbian_strengths: std::collections::HashMap::new(),
            activation_counts: vec![0; config.n_experts],
            last_introspection: None,
            config,
        })
    }

    /// Set expert names for human-readable introspection
    pub fn set_expert_names(&mut self, names: Vec<String>) {
        self.expert_names = names;
    }

    /// Forward pass — route tokens to experts
    ///
    /// Input: (batch, seq_len, d_model)
    /// Output: (routing_weights, expert_indices) per token
    ///   routing_weights: (batch, seq_len, top_k)
    ///   expert_indices: Vec<Vec<Vec<usize>>> — [batch][seq][top_k]
    pub fn forward(&mut self, x: &Tensor) -> Result<(Tensor, Vec<Vec<Vec<usize>>>)> {
        let (batch, seq_len, _) = x.dims3()?;
        let dev = &self.config.device;
        let top_k = self.config.top_k;
        let n_experts = self.config.n_experts;

        // Process through Mamba backbone for context-aware routing
        let context = self.mamba.forward(x)?; // (batch, seq_len, d_model)

        // Compute routing logits: (batch, seq_len, n_experts)
        let logits = linear_bias(&context, &self.router_proj, &self.router_bias)?;

        // Softmax over experts
        let routing_probs = candle_nn::ops::softmax(&logits, D::Minus1)?;

        // Extract routing decisions per token
        let mut all_indices: Vec<Vec<Vec<usize>>> = Vec::with_capacity(batch);
        let mut all_weights_data: Vec<Vec<Vec<f32>>> = Vec::with_capacity(batch);
        let mut decisions: Vec<RoutingDecision> = Vec::new();
        let mut expert_counts = vec![0usize; n_experts];

        for b in 0..batch {
            let mut batch_indices = Vec::with_capacity(seq_len);
            let mut batch_weights = Vec::with_capacity(seq_len);

            for t in 0..seq_len {
                let probs_t = routing_probs.narrow(0, b, 1)?.narrow(1, t, 1)?
                    .squeeze(0)?.squeeze(0)?; // (n_experts,)
                let probs_vec: Vec<f32> = probs_t.to_vec1()?;

                // Top-k selection
                let mut indexed: Vec<(usize, f32)> = probs_vec.iter()
                    .enumerate()
                    .map(|(i, &p)| (i, p))
                    .collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                let selected: Vec<usize> = indexed[..top_k].iter().map(|(i, _)| *i).collect();
                let weights: Vec<f32> = indexed[..top_k].iter().map(|(_, w)| *w).collect();

                // Normalize weights to sum to 1
                let w_sum: f32 = weights.iter().sum();
                let norm_weights: Vec<f32> = weights.iter().map(|w| w / w_sum).collect();

                // Update activation counts
                for &idx in &selected {
                    expert_counts[idx] += 1;
                    self.activation_counts[idx] += 1;
                }

                // Compute confidence
                let confidence = if indexed.len() >= 2 && indexed[1].1 > 1e-8 {
                    indexed[0].1 / indexed[1].1
                } else {
                    f32::MAX
                };

                // Build pathway key
                let mut pathway_experts = selected.clone();
                pathway_experts.sort();
                let pathway: String = pathway_experts.iter()
                    .map(|&i| self.expert_names.get(i).map(|s| s.as_str()).unwrap_or("?"))
                    .collect::<Vec<_>>()
                    .join("+");

                // Hebbian learning: strengthen this pathway
                if self.config.hebbian_learning {
                    let strength = self.hebbian_strengths.entry(pathway.clone()).or_insert(0.0);
                    *strength += self.config.hebbian_lr;
                    // Decay all others slightly
                    let decay = 0.999;
                    for (_, s) in self.hebbian_strengths.iter_mut() {
                        *s *= decay;
                    }
                }

                // Record decision
                if b == 0 {
                    decisions.push(RoutingDecision {
                        selected_experts: selected.clone(),
                        expert_weights: norm_weights.clone(),
                        all_scores: probs_vec.clone(),
                        confidence,
                        load_balance: compute_load_balance(&expert_counts, n_experts),
                        pathway,
                    });
                }

                batch_indices.push(selected);
                batch_weights.push(norm_weights);
            }

            all_indices.push(batch_indices);
            all_weights_data.push(batch_weights);
        }

        // Build weight tensor: (batch, seq_len, top_k)
        let flat_weights: Vec<f32> = all_weights_data.iter()
            .flat_map(|b| b.iter().flat_map(|t| t.iter().copied()))
            .collect();
        let weight_tensor = Tensor::new(flat_weights.as_slice(), dev)?
            .reshape(&[batch, seq_len, top_k])?;

        // Compute introspection
        let total_tokens = (batch * seq_len) as f32;
        let expert_utilization: Vec<f32> = expert_counts.iter()
            .map(|&c| c as f32 / total_tokens)
            .collect();
        let avg_confidence = if !decisions.is_empty() {
            decisions.iter().map(|d| d.confidence.min(100.0)).sum::<f32>() / decisions.len() as f32
        } else {
            0.0
        };

        // Top Hebbian pathways
        let mut pathways: Vec<(String, f32)> = self.hebbian_strengths.iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        pathways.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        pathways.truncate(10);

        self.last_introspection = Some(ThalamusIntrospection {
            decisions,
            expert_activation_counts: expert_counts.clone(),
            expert_names: self.expert_names.clone(),
            avg_confidence,
            load_balance_score: compute_load_balance(&expert_counts, n_experts),
            top_pathways: pathways,
            expert_utilization,
        });

        Ok((weight_tensor, all_indices))
    }

    /// Get the last routing introspection — see exactly what the router decided
    pub fn introspect(&self) -> Option<&ThalamusIntrospection> {
        self.last_introspection.as_ref()
    }

    /// Get Hebbian pathway strengths
    pub fn pathway_strengths(&self) -> &std::collections::HashMap<String, f32> {
        &self.hebbian_strengths
    }

    /// Load Hebbian data from the knowledge graph (bootstrap from existing routing data)
    pub fn load_hebbian_data(&mut self, pathways: Vec<(String, f32)>) {
        for (pathway, strength) in pathways {
            self.hebbian_strengths.insert(pathway, strength);
        }
    }

    /// Reset state
    pub fn reset_state(&mut self) {
        self.mamba.reset_state();
        self.activation_counts = vec![0; self.config.n_experts];
    }

    /// Human-readable summary of current routing state
    pub fn status_summary(&self) -> String {
        let total: usize = self.activation_counts.iter().sum();
        if total == 0 {
            return "Thalamus: No tokens processed yet".to_string();
        }

        let mut lines = vec![format!("Thalamus Router — {} tokens processed", total)];

        // Expert utilization
        for (i, &count) in self.activation_counts.iter().enumerate() {
            let pct = (count as f32 / total as f32) * 100.0;
            let name = self.expert_names.get(i).map(|s| s.as_str()).unwrap_or("?");
            let bar_len = (pct / 5.0) as usize;
            let bar: String = "█".repeat(bar_len);
            lines.push(format!("  {:12} {:5.1}% {}", name, pct, bar));
        }

        // Top pathways
        if !self.hebbian_strengths.is_empty() {
            lines.push("\nTop pathways (Hebbian):".to_string());
            let mut sorted: Vec<_> = self.hebbian_strengths.iter().collect();
            sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
            for (pathway, strength) in sorted.iter().take(5) {
                lines.push(format!("  {} → {:.3}", pathway, strength));
            }
        }

        lines.join("\n")
    }
}

/// Compute load balance score (1.0 = perfectly balanced, 0.0 = all on one expert)
fn compute_load_balance(counts: &[usize], n_experts: usize) -> f32 {
    let total: usize = counts.iter().sum();
    if total == 0 {
        return 1.0;
    }
    let expected = total as f32 / n_experts as f32;
    let variance: f32 = counts.iter()
        .map(|&c| {
            let diff = c as f32 - expected;
            diff * diff
        })
        .sum::<f32>() / n_experts as f32;
    let cv = variance.sqrt() / expected; // coefficient of variation
    (1.0 - cv).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thalamus_creation() {
        let config = ThalamusConfig {
            d_model: 64,
            n_experts: 4,
            top_k: 2,
            d_state: 8,
            device: Device::Cpu,
            ..Default::default()
        };
        let router = Thalamus::new(config);
        assert!(router.is_ok());
    }

    #[test]
    fn test_thalamus_routing() {
        let config = ThalamusConfig {
            d_model: 64,
            n_experts: 4,
            top_k: 2,
            d_state: 8,
            device: Device::Cpu,
            ..Default::default()
        };
        let mut router = Thalamus::new(config).unwrap();
        let x = Tensor::randn(0f32, 1.0, &[1, 8, 64], &Device::Cpu).unwrap();

        let (weights, indices) = router.forward(&x).unwrap();
        assert_eq!(weights.dims(), &[1, 8, 2]); // (batch=1, seq=8, top_k=2)
        assert_eq!(indices.len(), 1); // batch=1
        assert_eq!(indices[0].len(), 8); // seq=8
        assert_eq!(indices[0][0].len(), 2); // top_k=2

        // All indices should be < n_experts
        for token_indices in &indices[0] {
            for &idx in token_indices {
                assert!(idx < 4);
            }
        }
    }

    #[test]
    fn test_thalamus_introspection() {
        let config = ThalamusConfig {
            d_model: 32,
            n_experts: 4,
            top_k: 2,
            d_state: 4,
            device: Device::Cpu,
            ..Default::default()
        };
        let mut router = Thalamus::new(config).unwrap();
        let x = Tensor::randn(0f32, 1.0, &[1, 4, 32], &Device::Cpu).unwrap();
        let _ = router.forward(&x).unwrap();

        let intro = router.introspect().unwrap();
        assert_eq!(intro.decisions.len(), 4);
        assert_eq!(intro.expert_names.len(), 4);
        assert!(intro.avg_confidence > 0.0);

        // Each decision should have valid data
        for decision in &intro.decisions {
            assert_eq!(decision.selected_experts.len(), 2);
            assert_eq!(decision.expert_weights.len(), 2);
            assert_eq!(decision.all_scores.len(), 4);
            assert!(!decision.pathway.is_empty());
        }
    }

    #[test]
    fn test_hebbian_learning() {
        let config = ThalamusConfig {
            d_model: 32,
            n_experts: 4,
            top_k: 2,
            d_state: 4,
            hebbian_learning: true,
            hebbian_lr: 0.1,
            device: Device::Cpu,
            ..Default::default()
        };
        let mut router = Thalamus::new(config).unwrap();

        // Process some tokens
        let x = Tensor::randn(0f32, 1.0, &[1, 4, 32], &Device::Cpu).unwrap();
        let _ = router.forward(&x).unwrap();

        // Hebbian strengths should have been recorded
        assert!(!router.pathway_strengths().is_empty());
    }

    #[test]
    fn test_status_summary() {
        let config = ThalamusConfig {
            d_model: 32,
            n_experts: 4,
            top_k: 2,
            d_state: 4,
            device: Device::Cpu,
            ..Default::default()
        };
        let mut router = Thalamus::new(config).unwrap();
        let x = Tensor::randn(0f32, 1.0, &[1, 4, 32], &Device::Cpu).unwrap();
        let _ = router.forward(&x).unwrap();

        let summary = router.status_summary();
        assert!(summary.contains("Thalamus Router"));
        assert!(summary.contains("tokens processed"));
    }
}
