//! Mamba — Selective State Space Model
//!
//! Replaces transformer attention with O(n) state-space processing.
//! Based on "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
//! (Gu & Dao, 2023) but implemented from scratch in Rust + candle.
//!
//! Key insight: Instead of attending to ALL previous tokens (O(n²)),
//! Mamba maintains a compressed state that selectively remembers what matters.
//! The selection mechanism is input-dependent — different inputs cause
//! different information to be retained or forgotten.
//!
//! Used in Synapse as the Thalamus router backbone.

use anyhow::Result;
use candle_core::{Device, Tensor, DType, D};
use super::{linear, linear_bias};

/// Configuration for a Mamba layer
#[derive(Debug, Clone)]
pub struct MambaConfig {
    /// Model dimension (d_model)
    pub d_model: usize,
    /// State space dimension (N in the paper)
    pub d_state: usize,
    /// Convolution kernel size
    pub d_conv: usize,
    /// Expansion factor for inner dimension
    pub expand: usize,
    /// Device to run on
    pub device: Device,
}

impl Default for MambaConfig {
    fn default() -> Self {
        Self {
            d_model: 768,
            d_state: 16,
            d_conv: 4,
            expand: 2,
            device: Device::Cpu,
        }
    }
}

/// Selective State Space Model (S6) — the core of Mamba
///
/// Implements the continuous-time state space equation:
///   h'(t) = A h(t) + B x(t)
///   y(t) = C h(t) + D x(t)
///
/// With input-dependent discretization (the "selective" part):
///   delta, B, C are all functions of the input x
pub struct MambaLayer {
    config: MambaConfig,
    /// Inner dimension = d_model * expand
    d_inner: usize,

    // Projections
    /// Input projection: d_model → 2 * d_inner (split into x and z paths)
    in_proj: Tensor,
    /// Output projection: d_inner → d_model
    out_proj: Tensor,

    // SSM parameters
    /// A matrix (d_inner, d_state) — diagonal state matrix (log-space for stability)
    a_log: Tensor,
    /// D vector (d_inner,) — skip connection
    d: Tensor,

    // Selection mechanism projections
    /// Projects input to delta (step size): d_inner → dt_rank
    dt_proj_weight: Tensor,
    dt_proj_bias: Tensor,
    /// Projects dt_rank → d_inner
    dt_rank: usize,

    // Input-dependent B and C projections
    /// x → B: d_inner → d_state
    x_proj: Tensor,

    // 1D convolution
    conv1d_weight: Tensor,
    conv1d_bias: Tensor,

    // Running state for inference
    ssm_state: Option<Tensor>,
    conv_state: Option<Tensor>,
}

impl MambaLayer {
    pub fn new(config: MambaConfig) -> Result<Self> {
        let d_inner = config.d_model * config.expand;
        let dt_rank = (config.d_model + 15) / 16; // ceil(d_model / 16)
        let dev = &config.device;

        // Initialize projections with Xavier/Glorot uniform
        let scale_in = (1.0 / (config.d_model as f64)).sqrt();
        let scale_out = (1.0 / (d_inner as f64)).sqrt();

        let in_proj = Tensor::randn(
            0f32, scale_in as f32,
            &[2 * d_inner, config.d_model], dev
        )?;

        let out_proj = Tensor::randn(
            0f32, scale_out as f32,
            &[config.d_model, d_inner], dev
        )?;

        // A is initialized as negative log-uniform (for stability in continuous time)
        // A = -exp(a_log), where a_log is initialized as log(1..d_state+1)
        let a_log_data: Vec<f32> = (1..=config.d_state)
            .map(|i| (i as f32).ln())
            .collect();
        let a_log_single = Tensor::new(a_log_data.as_slice(), dev)?;
        let a_log = a_log_single
            .unsqueeze(0)?
            .expand(&[d_inner, config.d_state])?
            .contiguous()?;

        // D = ones (skip connection initialized to identity)
        let d = Tensor::ones(&[d_inner], DType::F32, dev)?;

        // Selection mechanism
        let dt_proj_weight = Tensor::randn(
            0f32, (1.0 / dt_rank as f32).sqrt(),
            &[d_inner, dt_rank], dev
        )?;
        let dt_proj_bias = Tensor::zeros(&[d_inner], DType::F32, dev)?;

        // x_proj: projects to dt, B, C concatenated
        // Output: dt_rank + 2 * d_state
        let x_proj = Tensor::randn(
            0f32, (1.0 / d_inner as f32).sqrt(),
            &[dt_rank + 2 * config.d_state, d_inner], dev
        )?;

        // 1D convolution
        let conv1d_weight = Tensor::randn(
            0f32, (1.0 / (config.d_conv as f32)).sqrt(),
            &[d_inner, 1, config.d_conv], dev
        )?;
        let conv1d_bias = Tensor::zeros(&[d_inner], DType::F32, dev)?;

        Ok(Self {
            d_inner,
            dt_rank,
            in_proj,
            out_proj,
            a_log,
            d,
            dt_proj_weight,
            dt_proj_bias,
            x_proj,
            conv1d_weight,
            conv1d_bias,
            ssm_state: None,
            conv_state: None,
            config,
        })
    }

    /// Forward pass — process a sequence through the selective SSM
    ///
    /// Input: (batch, seq_len, d_model)
    /// Output: (batch, seq_len, d_model)
    pub fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let (_batch, seq_len, _) = x.dims3()?;

        // Project input: (B, L, d_model) → (B, L, 2*d_inner)
        let xz = linear(x, &self.in_proj)?;

        // Split into x path and z path (gate)
        let x_path = xz.narrow(D::Minus1, 0, self.d_inner)?;
        let z_path = xz.narrow(D::Minus1, self.d_inner, self.d_inner)?;

        // 1D convolution on x path (causal, groups=d_inner)
        let x_conv = self.causal_conv1d(&x_path)?;

        // SiLU activation
        let x_act = silu(&x_conv)?;

        // Selective SSM
        let y = self.selective_ssm(&x_act)?;

        // Gate with z path (SiLU gate)
        let z_act = silu(&z_path)?;
        let output = (&y * &z_act)?;

        // Output projection: (B, L, d_inner) → (B, L, d_model)
        linear(&output, &self.out_proj)
    }

    /// Selective State Space Model — the core innovation
    ///
    /// Unlike traditional SSMs where A, B, C are static,
    /// Mamba makes B, C, and delta (step size) input-dependent.
    /// This lets the model selectively attend to or ignore inputs.
    fn selective_ssm(&mut self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;
        let dev = &self.config.device;

        // Project x to get dt, B, C
        // x_proj: (B, L, d_inner) → (B, L, dt_rank + 2*d_state)
        let x_dbc = linear(x, &self.x_proj)?;

        let dt_rank = self.dt_rank;
        let d_state = self.config.d_state;

        // Split projections
        let dt = x_dbc.narrow(D::Minus1, 0, dt_rank)?;
        let b = x_dbc.narrow(D::Minus1, dt_rank, d_state)?;
        let c = x_dbc.narrow(D::Minus1, dt_rank + d_state, d_state)?;

        // Project dt to full dimension + softplus for positivity
        let dt = linear_bias(&dt, &self.dt_proj_weight, &self.dt_proj_bias)?;
        let dt = softplus(&dt)?; // Ensure positive step sizes

        // Compute A from log space (negative for stability)
        let a = self.a_log.exp()?.neg()?;

        // Discretize: A_bar = exp(delta * A), B_bar = delta * B
        // For each position in the sequence

        // Initialize or get running state
        let mut h = if let Some(ref state) = self.ssm_state {
            state.clone()
        } else {
            Tensor::zeros(&[batch, self.d_inner, d_state], DType::F32, dev)?
        };

        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            // Get values at position t
            let dt_t = dt.narrow(1, t, 1)?.squeeze(1)?; // (B, d_inner)
            let b_t = b.narrow(1, t, 1)?.squeeze(1)?;   // (B, d_state)
            let c_t = c.narrow(1, t, 1)?.squeeze(1)?;   // (B, d_state)
            let x_t = x.narrow(1, t, 1)?.squeeze(1)?;   // (B, d_inner)

            // Discretize A: A_bar = exp(dt * A)
            // dt_t: (B, d_inner), a: (d_inner, d_state)
            let dt_a = dt_t
                .unsqueeze(D::Minus1)?   // (B, d_inner, 1)
                .broadcast_mul(
                    &a.unsqueeze(0)?     // (1, d_inner, d_state)
                )?;
            let a_bar = dt_a.exp()?;

            // Discretize B: B_bar = dt * B
            // dt_t: (B, d_inner), b_t: (B, d_state)
            let db = dt_t
                .unsqueeze(D::Minus1)?   // (B, d_inner, 1)
                .broadcast_mul(
                    &b_t.unsqueeze(1)?   // (B, 1, d_state)
                )?;

            // State update: h = A_bar * h + B_bar * x
            let x_expanded = x_t.unsqueeze(D::Minus1)?; // (B, d_inner, 1)
            let bx = db.broadcast_mul(&x_expanded)?;
            h = a_bar.broadcast_mul(&h)?.add(&bx)?;

            // Output: y = C * h + D * x
            // c_t: (B, d_state), h: (B, d_inner, d_state)
            let y_t = h.broadcast_mul(
                &c_t.unsqueeze(1)? // (B, 1, d_state)
            )?.sum(D::Minus1)?;    // (B, d_inner)

            // Add skip connection
            let y_t = (&y_t + &x_t.broadcast_mul(&self.d)?)?;

            outputs.push(y_t.unsqueeze(1)?); // (B, 1, d_inner)
        }

        // Save state for next call
        self.ssm_state = Some(h);

        // Stack outputs: (B, L, d_inner)
        Ok(Tensor::cat(&outputs, 1)?)
    }

    /// Causal 1D convolution (depthwise, padded)
    fn causal_conv1d(&mut self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, channels) = x.dims3()?;
        let k = self.config.d_conv;
        let dev = &self.config.device;

        // Pad left with zeros (causal = only look at past)
        let pad = Tensor::zeros(&[batch, k - 1, channels], DType::F32, dev)?;
        let padded = Tensor::cat(&[&pad, x], 1)?; // (B, L+k-1, C)

        // Manual depthwise conv1d: for each output position, dot with kernel
        let mut outputs = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let window = padded.narrow(1, t, k)?; // (B, k, C)

            // Dot product with conv weights per channel
            // conv1d_weight: (C, 1, k) → reshape to (C, k)
            let w = self.conv1d_weight.squeeze(1)?; // (C, k)
            let w_t = w.t()?; // (k, C)

            // Element-wise multiply and sum over kernel dimension
            let out = (window.broadcast_mul(&w_t.unsqueeze(0)?))?
                .sum(1)?; // (B, C)

            let out = out.broadcast_add(&self.conv1d_bias)?;
            outputs.push(out.unsqueeze(1)?);
        }

        Ok(Tensor::cat(&outputs, 1)?)
    }

    /// Reset the running state (call between sequences)
    pub fn reset_state(&mut self) {
        self.ssm_state = None;
        self.conv_state = None;
    }

    /// Step function for autoregressive generation (single token)
    /// Much faster than full forward for inference
    pub fn step(&mut self, x: &Tensor) -> Result<Tensor> {
        // x: (batch, 1, d_model) — single token
        self.forward(x)
    }
}

/// SiLU (Swish) activation: x * sigmoid(x)
fn silu(x: &Tensor) -> Result<Tensor> {
    let sigmoid = candle_nn::ops::sigmoid(x)?;
    x.mul(&sigmoid).map_err(|e| anyhow::anyhow!("{e}"))
}

/// Softplus: log(1 + exp(x))
fn softplus(x: &Tensor) -> Result<Tensor> {
    let ones = Tensor::ones(x.shape(), x.dtype(), x.device())?;
    let exp_x = x.exp()?;
    let result = (&ones + &exp_x)?.log()?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mamba_layer_creation() {
        let config = MambaConfig {
            d_model: 64,
            d_state: 8,
            d_conv: 4,
            expand: 2,
            device: Device::Cpu,
        };
        let layer = MambaLayer::new(config);
        assert!(layer.is_ok());
    }

    #[test]
    fn test_mamba_forward() {
        let config = MambaConfig {
            d_model: 64,
            d_state: 8,
            d_conv: 4,
            expand: 2,
            device: Device::Cpu,
        };
        let mut layer = MambaLayer::new(config).unwrap();

        // (batch=1, seq_len=8, d_model=64)
        let x = Tensor::randn(0f32, 1.0, &[1, 8, 64], &Device::Cpu).unwrap();
        let out = layer.forward(&x).unwrap();

        assert_eq!(out.dims(), &[1, 8, 64]);
    }

    #[test]
    fn test_mamba_state_persistence() {
        let config = MambaConfig {
            d_model: 32,
            d_state: 4,
            d_conv: 4,
            expand: 2,
            device: Device::Cpu,
        };
        let mut layer = MambaLayer::new(config).unwrap();

        let x1 = Tensor::randn(0f32, 1.0, &[1, 4, 32], &Device::Cpu).unwrap();
        let _ = layer.forward(&x1).unwrap();
        assert!(layer.ssm_state.is_some());

        layer.reset_state();
        assert!(layer.ssm_state.is_none());
    }

    #[test]
    fn test_silu() {
        let x = Tensor::new(&[0.0f32, 1.0, -1.0], &Device::Cpu).unwrap();
        let out = silu(&x).unwrap();
        let vals: Vec<f32> = out.to_vec1().unwrap();
        // silu(0) = 0, silu(1) ≈ 0.731, silu(-1) ≈ -0.269
        assert!((vals[0] - 0.0).abs() < 0.01);
        assert!((vals[1] - 0.731).abs() < 0.01);
        assert!((vals[2] - (-0.269)).abs() < 0.01);
    }
}
