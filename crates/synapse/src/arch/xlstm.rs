//! xLSTM — Extended Long Short-Term Memory
//!
//! Replaces transformer blocks with recurrent processing that has O(n) complexity.
//! Based on "xLSTM: Extended Long Short-Term Memory" (Beck et al., 2024)
//! but implemented from scratch in Rust + candle.
//!
//! Key innovations over classic LSTM:
//!   - Exponential gating: gates use exp() instead of sigmoid, giving
//!     much larger dynamic range for remembering/forgetting
//!   - Matrix memory: state is a matrix (not vector), storing richer representations
//!   - Normalizer state: prevents numerical overflow from exponential gates
//!
//! Used in Synapse as the Language Module — handles syntax, grammar, fluency.

use anyhow::Result;
use candle_core::{Device, Tensor, DType, D};

/// Configuration for an xLSTM layer
#[derive(Debug, Clone)]
pub struct XLSTMConfig {
    /// Input/output dimension
    pub d_model: usize,
    /// Hidden state dimension (typically 2-4x d_model)
    pub d_hidden: usize,
    /// Number of memory heads (matrix memory is multi-headed)
    pub n_heads: usize,
    /// Head dimension (d_hidden / n_heads)
    pub d_head: usize,
    /// Device to run on
    pub device: Device,
}

impl Default for XLSTMConfig {
    fn default() -> Self {
        let d_hidden = 1536;
        let n_heads = 4;
        Self {
            d_model: 768,
            d_hidden,
            n_heads,
            d_head: d_hidden / n_heads,
            device: Device::Cpu,
        }
    }
}

/// Introspection data from an xLSTM forward pass — see inside the "brain"
#[derive(Debug, Clone)]
pub struct XLSTMIntrospection {
    /// Input gate values per timestep — what new info is being written (0-∞, exponential)
    pub input_gate_values: Vec<Vec<f32>>,
    /// Forget gate values per timestep — how much old state is retained (0-∞, exponential)
    pub forget_gate_values: Vec<Vec<f32>>,
    /// Output gate values per timestep — what gets read out (0-1, sigmoid)
    pub output_gate_values: Vec<Vec<f32>>,
    /// Memory utilization — L2 norm of cell state per head per timestep
    pub memory_norms: Vec<Vec<f32>>,
    /// Effective memory age — how many steps since significant write (per head)
    pub memory_age: Vec<usize>,
    /// Sequence length processed
    pub seq_len: usize,
}

/// Extended LSTM (mLSTM variant — matrix memory)
///
/// The key insight: classic LSTM uses a vector cell state.
/// mLSTM uses a MATRIX cell state with multi-head structure,
/// giving it associative memory properties similar to attention
/// but with O(n) complexity instead of O(n²).
pub struct XLSTMLayer {
    config: XLSTMConfig,

    // Gate projections (all from input x)
    /// Input gate: x → i (exponential gating)
    w_i: Tensor,
    b_i: Tensor,
    /// Forget gate: x → f (exponential gating)
    w_f: Tensor,
    b_f: Tensor,
    /// Output gate: x → o (sigmoid gating)
    w_o: Tensor,
    b_o: Tensor,

    // Memory projections
    /// Key projection: x → k (what to write as)
    w_k: Tensor,
    /// Value projection: x → v (what to write)
    w_v: Tensor,
    /// Query projection: x → q (what to read)
    w_q: Tensor,

    // Output projection
    /// Hidden → output: d_hidden → d_model
    w_out: Tensor,

    // Running state for inference
    /// Cell state: (n_heads, d_head, d_head) — matrix memory per head
    cell_state: Option<Tensor>,
    /// Normalizer state: (n_heads, d_head) — prevents overflow
    normalizer_state: Option<Tensor>,
    /// Maximum forget gate value seen (for numerical stability)
    max_forget: Option<Tensor>,

    /// Last introspection data
    last_introspection: Option<XLSTMIntrospection>,
}

impl XLSTMLayer {
    pub fn new(config: XLSTMConfig) -> Result<Self> {
        let dev = &config.device;
        let d = config.d_model;
        let h = config.d_hidden;

        let scale_in = (1.0 / d as f64).sqrt() as f32;
        let scale_h = (1.0 / h as f64).sqrt() as f32;

        // Gate projections
        let w_i = Tensor::randn(0f32, scale_in, &[h, d], dev)?;
        let b_i = Tensor::zeros(&[h], DType::F32, dev)?;
        let w_f = Tensor::randn(0f32, scale_in, &[h, d], dev)?;
        let b_f = Tensor::ones(&[h], DType::F32, dev)?; // Bias toward remembering
        let w_o = Tensor::randn(0f32, scale_in, &[h, d], dev)?;
        let b_o = Tensor::zeros(&[h], DType::F32, dev)?;

        // Memory projections
        let w_k = Tensor::randn(0f32, scale_in, &[h, d], dev)?;
        let w_v = Tensor::randn(0f32, scale_in, &[h, d], dev)?;
        let w_q = Tensor::randn(0f32, scale_in, &[h, d], dev)?;

        // Output projection
        let w_out = Tensor::randn(0f32, scale_h, &[d, h], dev)?;

        Ok(Self {
            w_i, b_i, w_f, b_f, w_o, b_o,
            w_k, w_v, w_q, w_out,
            cell_state: None,
            normalizer_state: None,
            max_forget: None,
            last_introspection: None,
            config,
        })
    }

    /// Forward pass — process sequence through xLSTM
    ///
    /// Input: (batch, seq_len, d_model)
    /// Output: (batch, seq_len, d_model)
    pub fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;
        let dev = &self.config.device;
        let n_heads = self.config.n_heads;
        let d_head = self.config.d_head;

        // Introspection collectors
        let mut intro_input_gates = Vec::with_capacity(seq_len);
        let mut intro_forget_gates = Vec::with_capacity(seq_len);
        let mut intro_output_gates = Vec::with_capacity(seq_len);
        let mut intro_memory_norms = Vec::with_capacity(seq_len);

        // Initialize states if needed
        let mut c = if let Some(ref s) = self.cell_state {
            s.clone()
        } else {
            // (batch, n_heads, d_head, d_head) — matrix memory
            Tensor::zeros(&[batch, n_heads, d_head, d_head], DType::F32, dev)?
        };
        let mut n = if let Some(ref s) = self.normalizer_state {
            s.clone()
        } else {
            // (batch, n_heads, d_head)
            Tensor::ones(&[batch, n_heads, d_head], DType::F32, dev)?
        };

        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let x_t = x.narrow(1, t, 1)?.squeeze(1)?; // (batch, d_model)

            // Compute gates
            let i_pre = x_t.matmul(&self.w_i.t()?)?.broadcast_add(&self.b_i)?;
            let f_pre = x_t.matmul(&self.w_f.t()?)?.broadcast_add(&self.b_f)?;
            let o_pre = x_t.matmul(&self.w_o.t()?)?.broadcast_add(&self.b_o)?;

            // Exponential gating for input and forget (key xLSTM innovation)
            // Clamp to prevent overflow: exp(20) ≈ 500M, plenty of range
            let i_gate = clamp(&i_pre, -20.0, 20.0)?.exp()?;
            let f_gate = clamp(&f_pre, -20.0, 20.0)?.exp()?;
            // Output gate uses sigmoid (bounded 0-1)
            let o_gate = candle_nn::ops::sigmoid(&o_pre)?;

            // Record gate values for introspection
            if batch == 1 {
                let ig: Vec<f32> = i_gate.flatten_all()?.to_vec1()?;
                let fg: Vec<f32> = f_gate.flatten_all()?.to_vec1()?;
                let og: Vec<f32> = o_gate.flatten_all()?.to_vec1()?;
                intro_input_gates.push(ig[..n_heads.min(8)].to_vec());
                intro_forget_gates.push(fg[..n_heads.min(8)].to_vec());
                intro_output_gates.push(og[..n_heads.min(8)].to_vec());
            }

            // Compute key, value, query
            let k = x_t.matmul(&self.w_k.t()?)?; // (batch, d_hidden)
            let v = x_t.matmul(&self.w_v.t()?)?;
            let q = x_t.matmul(&self.w_q.t()?)?;

            // Reshape to multi-head: (batch, n_heads, d_head)
            let k = k.reshape(&[batch, n_heads, d_head])?;
            let v = v.reshape(&[batch, n_heads, d_head])?;
            let q = q.reshape(&[batch, n_heads, d_head])?;
            let i_g = i_gate.reshape(&[batch, n_heads, d_head])?;
            let f_g = f_gate.reshape(&[batch, n_heads, d_head])?;

            // Matrix memory update: C = f * C + i * (v ⊗ k)
            // v ⊗ k is outer product per head: (batch, n_heads, d_head, 1) * (batch, n_heads, 1, d_head)
            let v_col = v.unsqueeze(D::Minus1)?;  // (batch, n_heads, d_head, 1)
            let k_row = k.unsqueeze(2)?;           // (batch, n_heads, 1, d_head)
            let outer = v_col.matmul(&k_row)?;     // (batch, n_heads, d_head, d_head)

            // Scale outer product by input gate
            let i_scaled = i_g.unsqueeze(D::Minus1)?; // (batch, n_heads, d_head, 1)
            let write = outer.broadcast_mul(&i_scaled)?;

            // Scale cell state by forget gate
            let f_scaled = f_g.unsqueeze(D::Minus1)?; // (batch, n_heads, d_head, 1)
            c = c.broadcast_mul(&f_scaled)?.add(&write)?;

            // Update normalizer: n = f * n + i * k
            let n_update = (&f_g * &n)?.add(&(&i_g * &k)?)?;
            n = n_update;

            // Read from memory: h = C * q / (max(|n * q|, 1))
            let q_col = q.unsqueeze(D::Minus1)?;  // (batch, n_heads, d_head, 1)
            let read = c.matmul(&q_col)?.squeeze(D::Minus1)?; // (batch, n_heads, d_head)

            // Normalize to prevent explosion
            let nq = (&n * &q)?.sum(D::Minus1)?; // (batch, n_heads)
            let nq_abs = nq.abs()?;
            let ones = Tensor::ones(nq_abs.shape(), DType::F32, dev)?;
            let normalizer = nq_abs.maximum(&ones)?; // max(|n·q|, 1)
            let normalizer = normalizer.unsqueeze(D::Minus1)?; // (batch, n_heads, 1)
            let h_read = read.broadcast_div(&normalizer)?;

            // Apply output gate
            let o_g = o_gate.reshape(&[batch, n_heads, d_head])?;
            let h_gated = (&h_read * &o_g)?;

            // Record memory norms for introspection
            if batch == 1 {
                let norms: Vec<f32> = h_read
                    .sqr()?.sum(D::Minus1)?.sqrt()?
                    .flatten_all()?.to_vec1()?;
                intro_memory_norms.push(norms);
            }

            // Flatten heads and project to output: (batch, d_hidden) → (batch, d_model)
            let h_flat = h_gated.reshape(&[batch, self.config.d_hidden])?;
            let y_t = h_flat.matmul(&self.w_out.t()?)?;

            outputs.push(y_t.unsqueeze(1)?);
        }

        // Save state
        self.cell_state = Some(c);
        self.normalizer_state = Some(n);

        // Save introspection
        if batch == 1 {
            self.last_introspection = Some(XLSTMIntrospection {
                input_gate_values: intro_input_gates,
                forget_gate_values: intro_forget_gates,
                output_gate_values: intro_output_gates,
                memory_norms: intro_memory_norms,
                memory_age: vec![0; n_heads], // TODO: track actual age
                seq_len,
            });
        }

        Ok(Tensor::cat(&outputs, 1)?)
    }

    /// Get the last introspection data — see inside the xLSTM
    pub fn introspect(&self) -> Option<&XLSTMIntrospection> {
        self.last_introspection.as_ref()
    }

    /// Reset running state
    pub fn reset_state(&mut self) {
        self.cell_state = None;
        self.normalizer_state = None;
        self.max_forget = None;
    }

    /// Step function for single-token autoregressive generation
    pub fn step(&mut self, x: &Tensor) -> Result<Tensor> {
        self.forward(x)
    }
}

/// Clamp tensor values to [min, max]
fn clamp(x: &Tensor, min_val: f32, max_val: f32) -> Result<Tensor> {
    let dev = x.device();
    let min_t = Tensor::new(&[min_val], dev)?.broadcast_as(x.shape())?;
    let max_t = Tensor::new(&[max_val], dev)?.broadcast_as(x.shape())?;
    let clamped = x.maximum(&min_t)?.minimum(&max_t)?;
    Ok(clamped)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xlstm_creation() {
        let config = XLSTMConfig {
            d_model: 64,
            d_hidden: 128,
            n_heads: 4,
            d_head: 32,
            device: Device::Cpu,
        };
        let layer = XLSTMLayer::new(config);
        assert!(layer.is_ok());
    }

    #[test]
    fn test_xlstm_forward() {
        let config = XLSTMConfig {
            d_model: 64,
            d_hidden: 128,
            n_heads: 4,
            d_head: 32,
            device: Device::Cpu,
        };
        let mut layer = XLSTMLayer::new(config).unwrap();
        let x = Tensor::randn(0f32, 1.0, &[1, 8, 64], &Device::Cpu).unwrap();
        let out = layer.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, 8, 64]);
    }

    #[test]
    fn test_xlstm_introspection() {
        let config = XLSTMConfig {
            d_model: 32,
            d_hidden: 64,
            n_heads: 2,
            d_head: 32,
            device: Device::Cpu,
        };
        let mut layer = XLSTMLayer::new(config).unwrap();
        let x = Tensor::randn(0f32, 1.0, &[1, 4, 32], &Device::Cpu).unwrap();
        let _ = layer.forward(&x).unwrap();

        let intro = layer.introspect().unwrap();
        assert_eq!(intro.seq_len, 4);
        assert_eq!(intro.input_gate_values.len(), 4);
        assert_eq!(intro.forget_gate_values.len(), 4);
        assert_eq!(intro.memory_norms.len(), 4);
    }

    #[test]
    fn test_xlstm_state_persistence() {
        let config = XLSTMConfig {
            d_model: 32,
            d_hidden: 64,
            n_heads: 2,
            d_head: 32,
            device: Device::Cpu,
        };
        let mut layer = XLSTMLayer::new(config).unwrap();
        let x = Tensor::randn(0f32, 1.0, &[1, 4, 32], &Device::Cpu).unwrap();
        let _ = layer.forward(&x).unwrap();
        assert!(layer.cell_state.is_some());

        layer.reset_state();
        assert!(layer.cell_state.is_none());
    }
}
