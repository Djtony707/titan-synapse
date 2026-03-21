//! Expert — Sparse Mixture of Experts Blocks
//!
//! Each expert is a specialized feed-forward network that processes tokens
//! routed to it by the Thalamus. Only top-k experts activate per token,
//! giving us sparse activation — most of the model is dormant at any time.
//!
//! Key insight: A 3B parameter model with 8 experts and top-2 routing
//! only uses ~800M parameters per token. You get the knowledge capacity
//! of 3B params with the speed of 800M.
//!
//! OBSERVABILITY: Each expert tracks its own activation statistics,
//! specialization score, and contribution magnitude. You can see
//! exactly which experts are doing the heavy lifting and which are coasting.

use anyhow::Result;
use candle_core::{Device, Tensor, DType, D};

/// Configuration for the expert pool
#[derive(Debug, Clone)]
pub struct ExpertPoolConfig {
    /// Input/output dimension
    pub d_model: usize,
    /// Expert hidden dimension (typically 4x d_model, like transformer FFN)
    pub d_expert: usize,
    /// Number of experts
    pub n_experts: usize,
    /// Device
    pub device: Device,
}

impl Default for ExpertPoolConfig {
    fn default() -> Self {
        Self {
            d_model: 768,
            d_expert: 3072,
            n_experts: 8,
            device: Device::Cpu,
        }
    }
}

/// Introspection data for a single expert
#[derive(Debug, Clone)]
pub struct ExpertStats {
    /// Expert name/index
    pub name: String,
    /// How many tokens this expert processed
    pub tokens_processed: usize,
    /// Average output magnitude (L2 norm)
    pub avg_output_magnitude: f32,
    /// Average activation sparsity (% of hidden units near zero)
    pub avg_activation_sparsity: f32,
    /// Specialization score: how different this expert's output is from average
    pub specialization_score: f32,
}

/// Full introspection for the expert pool
#[derive(Debug, Clone)]
pub struct ExpertPoolIntrospection {
    /// Per-expert statistics
    pub expert_stats: Vec<ExpertStats>,
    /// Which experts contributed most to the output (by magnitude)
    pub top_contributors: Vec<(usize, f32)>,
    /// Total tokens processed in last forward pass
    pub total_tokens: usize,
    /// Sparsity: average fraction of experts activated
    pub activation_sparsity: f32,
}

/// A single expert — gated feed-forward network with SwiGLU activation
pub struct Expert {
    /// Up projection: d_model → d_expert
    w_up: Tensor,
    /// Gate projection: d_model → d_expert (for SwiGLU)
    w_gate: Tensor,
    /// Down projection: d_expert → d_model
    w_down: Tensor,

    // Running stats
    tokens_processed: usize,
    total_output_magnitude: f32,
    total_activation_sparsity: f32,
}

impl Expert {
    pub fn new(d_model: usize, d_expert: usize, device: &Device) -> Result<Self> {
        let scale_in = (1.0 / d_model as f64).sqrt() as f32;
        let scale_h = (1.0 / d_expert as f64).sqrt() as f32;

        let w_up = Tensor::randn(0f32, scale_in, &[d_expert, d_model], device)?;
        let w_gate = Tensor::randn(0f32, scale_in, &[d_expert, d_model], device)?;
        let w_down = Tensor::randn(0f32, scale_h, &[d_model, d_expert], device)?;

        Ok(Self {
            w_up,
            w_gate,
            w_down,
            tokens_processed: 0,
            total_output_magnitude: 0.0,
            total_activation_sparsity: 0.0,
        })
    }

    /// Forward pass: SwiGLU FFN
    /// Input: (tokens, d_model) — flat token batch
    /// Output: (tokens, d_model)
    pub fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        // SwiGLU: down(silu(gate(x)) * up(x))
        let gate = x.matmul(&self.w_gate.t()?)?;
        let up = x.matmul(&self.w_up.t()?)?;
        let gate_act = silu(&gate)?;
        let hidden = (&gate_act * &up)?;

        // Track activation sparsity (how many hidden units are near zero)
        let sparsity = compute_sparsity(&hidden)?;
        let n_tokens = x.dims()[0];
        self.tokens_processed += n_tokens;
        self.total_activation_sparsity += sparsity * n_tokens as f32;

        let output = hidden.matmul(&self.w_down.t()?)?;

        // Track output magnitude
        let mag = output.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
        self.total_output_magnitude += mag * n_tokens as f32;

        Ok(output)
    }

    /// Get running statistics
    pub fn stats(&self, name: &str) -> ExpertStats {
        let n = self.tokens_processed.max(1) as f32;
        ExpertStats {
            name: name.to_string(),
            tokens_processed: self.tokens_processed,
            avg_output_magnitude: self.total_output_magnitude / n,
            avg_activation_sparsity: self.total_activation_sparsity / n,
            specialization_score: 0.0, // Computed at pool level
        }
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.tokens_processed = 0;
        self.total_output_magnitude = 0.0;
        self.total_activation_sparsity = 0.0;
    }
}

/// Pool of experts managed by the Thalamus router
pub struct ExpertPool {
    config: ExpertPoolConfig,
    /// The experts themselves
    experts: Vec<Expert>,
    /// Expert names
    expert_names: Vec<String>,
    /// Last introspection
    last_introspection: Option<ExpertPoolIntrospection>,
}

impl ExpertPool {
    pub fn new(config: ExpertPoolConfig) -> Result<Self> {
        let mut experts = Vec::with_capacity(config.n_experts);
        let mut names = Vec::with_capacity(config.n_experts);

        for i in 0..config.n_experts {
            experts.push(Expert::new(config.d_model, config.d_expert, &config.device)?);
            names.push(format!("expert_{i}"));
        }

        Ok(Self {
            experts,
            expert_names: names,
            last_introspection: None,
            config,
        })
    }

    /// Set expert names
    pub fn set_expert_names(&mut self, names: Vec<String>) {
        self.expert_names = names;
    }

    /// Forward pass — route tokens to selected experts and combine
    ///
    /// x: (batch, seq_len, d_model)
    /// routing_weights: (batch, seq_len, top_k)
    /// expert_indices: [batch][seq][top_k] — which experts to use
    ///
    /// Output: (batch, seq_len, d_model)
    pub fn forward(
        &mut self,
        x: &Tensor,
        routing_weights: &Tensor,
        expert_indices: &[Vec<Vec<usize>>],
    ) -> Result<Tensor> {
        let (batch, seq_len, d_model) = x.dims3()?;
        let dev = &self.config.device;

        let mut output = Tensor::zeros(&[batch, seq_len, d_model], DType::F32, dev)?;

        // Process each expert's assigned tokens
        // (In production, this would be batched more efficiently)
        for b in 0..batch {
            for t in 0..seq_len {
                let x_t = x.narrow(0, b, 1)?.narrow(1, t, 1)?
                    .reshape(&[1, d_model])?; // (1, d_model)

                let mut combined = Tensor::zeros(&[1, d_model], DType::F32, dev)?;

                for (k, &expert_idx) in expert_indices[b][t].iter().enumerate() {
                    if expert_idx >= self.experts.len() {
                        continue;
                    }

                    // Get routing weight for this expert
                    let weight = routing_weights
                        .narrow(0, b, 1)?
                        .narrow(1, t, 1)?
                        .narrow(2, k, 1)?
                        .squeeze(0)?.squeeze(0)?.squeeze(0)?
                        .to_scalar::<f32>()?;

                    // Run through expert
                    let expert_out = self.experts[expert_idx].forward(&x_t)?;

                    // Weight and accumulate
                    let weight_t = Tensor::new(&[weight], dev)?;
                    let weighted = expert_out.broadcast_mul(&weight_t)?;
                    combined = (&combined + &weighted)?;
                }

                // Place combined output
                let combined_3d = combined.unsqueeze(0)?; // (1, 1, d_model)
                output = output.slice_assign(
                    &[b..b+1, t..t+1, 0..d_model],
                    &combined_3d,
                )?;
            }
        }

        // Build introspection
        let expert_stats: Vec<ExpertStats> = self.experts.iter()
            .enumerate()
            .map(|(i, e)| e.stats(self.expert_names.get(i).map(|s| s.as_str()).unwrap_or("?")))
            .collect();

        let mut top_contributors: Vec<(usize, f32)> = expert_stats.iter()
            .enumerate()
            .map(|(i, s)| (i, s.avg_output_magnitude))
            .collect();
        top_contributors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let active_experts: usize = expert_stats.iter()
            .filter(|s| s.tokens_processed > 0)
            .count();
        let activation_sparsity = 1.0 - (active_experts as f32 / self.config.n_experts as f32);

        self.last_introspection = Some(ExpertPoolIntrospection {
            expert_stats,
            top_contributors,
            total_tokens: batch * seq_len,
            activation_sparsity,
        });

        Ok(output)
    }

    /// Get introspection data
    pub fn introspect(&self) -> Option<&ExpertPoolIntrospection> {
        self.last_introspection.as_ref()
    }

    /// Reset all expert statistics
    pub fn reset_stats(&mut self) {
        for expert in &mut self.experts {
            expert.reset_stats();
        }
    }

    /// Number of experts
    pub fn n_experts(&self) -> usize {
        self.experts.len()
    }
}

/// Compute activation sparsity (fraction of values near zero)
fn compute_sparsity(x: &Tensor) -> Result<f32> {
    let abs = x.abs()?;
    let threshold = Tensor::new(&[0.01f32], x.device())?.broadcast_as(abs.shape())?;
    let near_zero = abs.lt(&threshold)?;
    let total = x.elem_count() as f32;
    let sparse_count = near_zero.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?;
    Ok(sparse_count / total)
}

/// SiLU activation
fn silu(x: &Tensor) -> Result<Tensor> {
    let sigmoid = candle_nn::ops::sigmoid(x)?;
    x.mul(&sigmoid).map_err(|e| anyhow::anyhow!("{e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expert_creation() {
        let expert = Expert::new(64, 256, &Device::Cpu);
        assert!(expert.is_ok());
    }

    #[test]
    fn test_expert_forward() {
        let mut expert = Expert::new(64, 256, &Device::Cpu).unwrap();
        let x = Tensor::randn(0f32, 1.0, &[4, 64], &Device::Cpu).unwrap();
        let out = expert.forward(&x).unwrap();
        assert_eq!(out.dims(), &[4, 64]);
    }

    #[test]
    fn test_expert_pool() {
        let config = ExpertPoolConfig {
            d_model: 64,
            d_expert: 256,
            n_experts: 4,
            device: Device::Cpu,
        };
        let pool = ExpertPool::new(config);
        assert!(pool.is_ok());
        assert_eq!(pool.unwrap().n_experts(), 4);
    }

    #[test]
    fn test_expert_pool_forward() {
        let config = ExpertPoolConfig {
            d_model: 32,
            d_expert: 128,
            n_experts: 4,
            device: Device::Cpu,
        };
        let mut pool = ExpertPool::new(config).unwrap();

        let x = Tensor::randn(0f32, 1.0, &[1, 4, 32], &Device::Cpu).unwrap();
        let weights = Tensor::new(
            &[0.6f32, 0.4, 0.5, 0.5, 0.7, 0.3, 0.55, 0.45],
            &Device::Cpu,
        ).unwrap().reshape(&[1, 4, 2]).unwrap();

        let indices = vec![vec![
            vec![0, 1], vec![1, 2], vec![0, 3], vec![2, 3],
        ]];

        let out = pool.forward(&x, &weights, &indices).unwrap();
        assert_eq!(out.dims(), &[1, 4, 32]);
    }

    #[test]
    fn test_expert_introspection() {
        let config = ExpertPoolConfig {
            d_model: 32,
            d_expert: 128,
            n_experts: 4,
            device: Device::Cpu,
        };
        let mut pool = ExpertPool::new(config).unwrap();

        let x = Tensor::randn(0f32, 1.0, &[1, 4, 32], &Device::Cpu).unwrap();
        let weights = Tensor::new(
            &[0.6f32, 0.4, 0.5, 0.5, 0.7, 0.3, 0.55, 0.45],
            &Device::Cpu,
        ).unwrap().reshape(&[1, 4, 2]).unwrap();
        let indices = vec![vec![
            vec![0, 1], vec![1, 2], vec![0, 3], vec![2, 3],
        ]];

        let _ = pool.forward(&x, &weights, &indices).unwrap();
        let intro = pool.introspect().unwrap();
        assert_eq!(intro.expert_stats.len(), 4);
        assert_eq!(intro.total_tokens, 4);
    }
}
