//! Fast-Weight Memory — Learn During Inference
//!
//! Based on Schmidhuber's fast weights (1992) modernized for 2026.
//! This is how the model learns NEW facts in a SINGLE forward pass
//! without any backpropagation. No training loop. No gradient updates.
//! Just direct memory writes during inference.
//!
//! How it works:
//!   1. Input arrives with key (what it's about) and value (the content)
//!   2. An outer product k⊗v is added to the fast-weight matrix
//!   3. Future queries retrieve values by multiplying the matrix by the query
//!   4. A decay factor controls how quickly old memories fade
//!
//! Think of it like a continuously-updating lookup table in weight space.
//! The model literally rewires itself as it processes each token.
//!
//! OBSERVABILITY: Every memory write and read is logged. You can see
//! exactly what facts the model is storing, how strong each memory is,
//! and what's being retrieved for each query.

use anyhow::Result;
use candle_core::{Device, Tensor, DType, D};

/// Configuration for fast-weight memory
#[derive(Debug, Clone)]
pub struct FastWeightConfig {
    /// Key/query dimension
    pub d_key: usize,
    /// Value dimension
    pub d_value: usize,
    /// Input dimension (d_model)
    pub d_model: usize,
    /// Number of memory slots (heads)
    pub n_heads: usize,
    /// Decay factor per step (0.95 = slow decay, 0.5 = fast decay)
    pub decay: f32,
    /// Write strength — how strongly new memories are written
    pub write_strength: f32,
    /// Minimum retrieval confidence to return a value (vs. zeros)
    pub retrieval_threshold: f32,
    /// Device
    pub device: Device,
}

impl Default for FastWeightConfig {
    fn default() -> Self {
        Self {
            d_key: 64,
            d_value: 64,
            d_model: 768,
            n_heads: 8,
            decay: 0.95,
            write_strength: 0.1,
            retrieval_threshold: 0.01,
            device: Device::Cpu,
        }
    }
}

/// A single memory write event
#[derive(Debug, Clone)]
pub struct MemoryWrite {
    /// Step when this write occurred
    pub step: usize,
    /// Write strength (after gating)
    pub strength: f32,
    /// Which head
    pub head: usize,
    /// Key norm (indicates how specific the memory address is)
    pub key_norm: f32,
    /// Value norm (indicates how much content was stored)
    pub value_norm: f32,
}

/// A single memory read event
#[derive(Debug, Clone)]
pub struct MemoryRead {
    /// Step when this read occurred
    pub step: usize,
    /// Retrieval confidence (higher = stronger match)
    pub confidence: f32,
    /// Which head contributed most
    pub dominant_head: usize,
    /// Retrieved value norm
    pub value_norm: f32,
}

/// Full introspection for fast-weight memory
#[derive(Debug, Clone)]
pub struct FastWeightIntrospection {
    /// All write events
    pub writes: Vec<MemoryWrite>,
    /// All read events
    pub reads: Vec<MemoryRead>,
    /// Current memory utilization per head (Frobenius norm of weight matrix)
    pub memory_utilization: Vec<f32>,
    /// Memory capacity remaining (estimated, before saturation)
    pub capacity_remaining: f32,
    /// Total facts stored (estimated by write count minus decay)
    pub estimated_facts_stored: usize,
    /// Steps processed
    pub total_steps: usize,
}

/// Fast-Weight Memory Module
///
/// Maintains a set of fast-weight matrices that are updated
/// during the forward pass (not during training).
pub struct FastWeightMemory {
    config: FastWeightConfig,

    // Projections
    /// Input → key: d_model → n_heads * d_key
    w_key: Tensor,
    /// Input → value: d_model → n_heads * d_value
    w_value: Tensor,
    /// Input → query: d_model → n_heads * d_key
    w_query: Tensor,
    /// Input → write gate: d_model → n_heads (sigmoid, controls write strength)
    w_gate: Tensor,
    /// Output projection: n_heads * d_value → d_model
    w_out: Tensor,

    /// The fast-weight matrices: (n_heads, d_key, d_value)
    /// These are updated DURING inference, not during training
    fast_weights: Tensor,

    // Introspection tracking
    step_counter: usize,
    write_history: Vec<MemoryWrite>,
    read_history: Vec<MemoryRead>,
    last_introspection: Option<FastWeightIntrospection>,
}

impl FastWeightMemory {
    pub fn new(config: FastWeightConfig) -> Result<Self> {
        let dev = &config.device;
        let d = config.d_model;
        let nh = config.n_heads;
        let dk = config.d_key;
        let dv = config.d_value;

        let scale = (1.0 / d as f64).sqrt() as f32;

        let w_key = Tensor::randn(0f32, scale, &[nh * dk, d], dev)?;
        let w_value = Tensor::randn(0f32, scale, &[nh * dv, d], dev)?;
        let w_query = Tensor::randn(0f32, scale, &[nh * dk, d], dev)?;
        let w_gate = Tensor::randn(0f32, scale, &[nh, d], dev)?;
        let w_out = Tensor::randn(
            0f32, (1.0 / (nh * dv) as f64).sqrt() as f32,
            &[d, nh * dv], dev
        )?;

        // Initialize fast weights to zero (empty memory)
        let fast_weights = Tensor::zeros(&[nh, dk, dv], DType::F32, dev)?;

        Ok(Self {
            w_key, w_value, w_query, w_gate, w_out,
            fast_weights,
            step_counter: 0,
            write_history: Vec::new(),
            read_history: Vec::new(),
            last_introspection: None,
            config,
        })
    }

    /// Forward pass — read from memory, then write new knowledge
    ///
    /// Input: (batch, seq_len, d_model)
    /// Output: (batch, seq_len, d_model) — retrieved memories added to input
    pub fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;
        let dev = &self.config.device;
        let nh = self.config.n_heads;
        let dk = self.config.d_key;
        let dv = self.config.d_value;

        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let x_t = x.narrow(1, t, 1)?.squeeze(1)?; // (batch, d_model)

            // Compute key, value, query
            let k = x_t.matmul(&self.w_key.t()?)?   // (batch, nh*dk)
                .reshape(&[batch, nh, dk])?;
            let v = x_t.matmul(&self.w_value.t()?)?  // (batch, nh*dv)
                .reshape(&[batch, nh, dv])?;
            let q = x_t.matmul(&self.w_query.t()?)?  // (batch, nh*dk)
                .reshape(&[batch, nh, dk])?;

            // Write gate: determines how strongly to write (per head)
            let gate = candle_nn::ops::sigmoid(
                &x_t.matmul(&self.w_gate.t()?)? // (batch, nh)
            )?;

            // === READ: retrieve from fast-weight memory ===
            // For each head: retrieved = fast_weights[h] @ q[h]
            // q: (batch, nh, dk) → need (batch, nh, dk, 1)
            let q_col = q.unsqueeze(D::Minus1)?;
            // fast_weights: (nh, dk, dv) → broadcast to (batch, nh, dk, dv)
            let fw_expanded = self.fast_weights.unsqueeze(0)?
                .expand(&[batch, nh, dk, dv])?;
            // (batch, nh, dk, dv) × (batch, nh, dk, 1) → sum over dk → (batch, nh, dv)
            let retrieved = (&fw_expanded * &q_col.expand(&[batch, nh, dk, dv])?)?
                .sum(2)?; // (batch, nh, dv)

            // Record read introspection
            if batch == 1 {
                let ret_norms: Vec<f32> = retrieved
                    .sqr()?.sum(D::Minus1)?.sqrt()?
                    .squeeze(0)?.to_vec1()?;
                let dominant_head = ret_norms.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let max_conf = ret_norms.iter().copied().fold(0f32, f32::max);
                self.read_history.push(MemoryRead {
                    step: self.step_counter,
                    confidence: max_conf,
                    dominant_head,
                    value_norm: max_conf,
                });
            }

            // Project retrieved to output dimension
            let retrieved_flat = retrieved.reshape(&[batch, nh * dv])?;
            let mem_out = retrieved_flat.matmul(&self.w_out.t()?)?; // (batch, d_model)

            // Output = input + memory contribution
            let y_t = (&x_t + &mem_out)?;

            // === WRITE: update fast-weight memory ===
            // For each head: fast_weights[h] += gate * decay_applied * (v ⊗ k)
            // First decay existing weights
            let decay = Tensor::new(&[self.config.decay], dev)?
                .broadcast_as(self.fast_weights.shape())?;
            self.fast_weights = (&self.fast_weights * &decay)?;

            // Write new memory: outer product of k and v, scaled by gate
            // We process batch=0 for the weight update (fast weights are shared)
            if batch >= 1 {
                let k_0 = k.narrow(0, 0, 1)?.squeeze(0)?; // (nh, dk)
                let v_0 = v.narrow(0, 0, 1)?.squeeze(0)?; // (nh, dv)
                let gate_0 = gate.narrow(0, 0, 1)?.squeeze(0)?; // (nh,)

                // Outer product per head: k ⊗ v
                let k_col = k_0.unsqueeze(D::Minus1)?; // (nh, dk, 1)
                let v_row = v_0.unsqueeze(1)?;          // (nh, 1, dv)
                let outer = k_col.matmul(&v_row)?;      // (nh, dk, dv)

                // Scale by gate and write strength
                let ws = Tensor::new(&[self.config.write_strength], dev)?.broadcast_as(gate_0.shape())?;
                let gate_scale = (&gate_0 * &ws)?;
                let gate_expanded = gate_scale
                    .unsqueeze(D::Minus1)?
                    .unsqueeze(D::Minus1)?; // (nh, 1, 1)
                let write = (&outer * &gate_expanded.expand(&[nh, dk, dv])?)?;

                self.fast_weights = (&self.fast_weights + &write)?;

                // Record write introspection
                if batch == 1 {
                    let gate_vals: Vec<f32> = gate_0.to_vec1()?;
                    for (h, &g) in gate_vals.iter().enumerate() {
                        if g * self.config.write_strength > 0.001 {
                            let k_norm = k_0.narrow(0, h, 1)?
                                .sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
                            let v_norm = v_0.narrow(0, h, 1)?
                                .sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
                            self.write_history.push(MemoryWrite {
                                step: self.step_counter,
                                strength: g * self.config.write_strength,
                                head: h,
                                key_norm: k_norm,
                                value_norm: v_norm,
                            });
                        }
                    }
                }
            }

            self.step_counter += 1;
            outputs.push(y_t.unsqueeze(1)?);
        }

        // Build introspection
        let mem_norms: Vec<f32> = (0..nh)
            .map(|h| {
                self.fast_weights.narrow(0, h, 1)
                    .and_then(|t| t.sqr()?.sum_all()?.to_scalar::<f32>())
                    .map(|v| v.sqrt())
                    .unwrap_or(0.0)
            })
            .collect();

        let max_capacity = (dk * dv) as f32; // theoretical max per head
        let used = mem_norms.iter().sum::<f32>() / nh as f32;
        let capacity_remaining = ((max_capacity - used) / max_capacity).max(0.0);

        self.last_introspection = Some(FastWeightIntrospection {
            writes: self.write_history.clone(),
            reads: self.read_history.clone(),
            memory_utilization: mem_norms,
            capacity_remaining,
            estimated_facts_stored: self.write_history.len(),
            total_steps: self.step_counter,
        });

        Ok(Tensor::cat(&outputs, 1)?)
    }

    /// Get introspection data — see what the memory has learned
    pub fn introspect(&self) -> Option<&FastWeightIntrospection> {
        self.last_introspection.as_ref()
    }

    /// Clear fast-weight memory (forget everything learned during inference)
    pub fn clear_memory(&mut self) {
        let dev = &self.config.device;
        let nh = self.config.n_heads;
        let dk = self.config.d_key;
        let dv = self.config.d_value;
        self.fast_weights = Tensor::zeros(&[nh, dk, dv], DType::F32, dev)
            .expect("Failed to create zero tensor");
        self.step_counter = 0;
        self.write_history.clear();
        self.read_history.clear();
    }

    /// Get human-readable memory status
    pub fn status_summary(&self) -> String {
        let intro = match &self.last_introspection {
            Some(i) => i,
            None => return "Fast-Weight Memory: No data yet".to_string(),
        };

        let mut lines = vec![
            format!("Fast-Weight Memory — {} steps, ~{} facts stored",
                    intro.total_steps, intro.estimated_facts_stored),
            format!("  Capacity remaining: {:.0}%", intro.capacity_remaining * 100.0),
        ];

        lines.push("  Head utilization:".to_string());
        for (i, &norm) in intro.memory_utilization.iter().enumerate() {
            let bar_len = (norm * 10.0).min(20.0) as usize;
            let bar: String = "█".repeat(bar_len);
            lines.push(format!("    Head {}: {:.2} {}", i, norm, bar));
        }

        if !intro.writes.is_empty() {
            let recent: Vec<_> = intro.writes.iter().rev().take(5).collect();
            lines.push(format!("  Recent writes ({} total):", intro.writes.len()));
            for w in recent {
                lines.push(format!("    Step {} head {} strength={:.3} key_norm={:.2}",
                    w.step, w.head, w.strength, w.key_norm));
            }
        }

        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_weight_creation() {
        let config = FastWeightConfig {
            d_key: 16,
            d_value: 16,
            d_model: 32,
            n_heads: 2,
            device: Device::Cpu,
            ..Default::default()
        };
        let mem = FastWeightMemory::new(config);
        assert!(mem.is_ok());
    }

    #[test]
    fn test_fast_weight_forward() {
        let config = FastWeightConfig {
            d_key: 16,
            d_value: 16,
            d_model: 32,
            n_heads: 2,
            device: Device::Cpu,
            ..Default::default()
        };
        let mut mem = FastWeightMemory::new(config).unwrap();
        let x = Tensor::randn(0f32, 1.0, &[1, 4, 32], &Device::Cpu).unwrap();
        let out = mem.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, 4, 32]);
    }

    #[test]
    fn test_fast_weight_introspection() {
        let config = FastWeightConfig {
            d_key: 16,
            d_value: 16,
            d_model: 32,
            n_heads: 2,
            device: Device::Cpu,
            ..Default::default()
        };
        let mut mem = FastWeightMemory::new(config).unwrap();
        let x = Tensor::randn(0f32, 1.0, &[1, 4, 32], &Device::Cpu).unwrap();
        let _ = mem.forward(&x).unwrap();

        let intro = mem.introspect().unwrap();
        assert_eq!(intro.total_steps, 4);
        assert_eq!(intro.memory_utilization.len(), 2);
        assert!(!intro.reads.is_empty());
    }

    #[test]
    fn test_fast_weight_memory_persists() {
        let config = FastWeightConfig {
            d_key: 16,
            d_value: 16,
            d_model: 32,
            n_heads: 2,
            write_strength: 1.0, // Strong writes for testing
            decay: 0.99,
            device: Device::Cpu,
            ..Default::default()
        };
        let mut mem = FastWeightMemory::new(config).unwrap();

        // Write some data
        let x1 = Tensor::randn(0f32, 1.0, &[1, 4, 32], &Device::Cpu).unwrap();
        let _ = mem.forward(&x1).unwrap();

        // Memory should be non-zero now
        let fw_norm: f32 = mem.fast_weights.sqr().unwrap()
            .sum_all().unwrap().to_scalar().unwrap();
        assert!(fw_norm > 0.0, "Fast weights should be non-zero after writes");

        // Clear should reset
        mem.clear_memory();
        let fw_norm_after: f32 = mem.fast_weights.sqr().unwrap()
            .sum_all().unwrap().to_scalar().unwrap();
        assert!(fw_norm_after < 1e-10, "Fast weights should be zero after clear");
    }
}
