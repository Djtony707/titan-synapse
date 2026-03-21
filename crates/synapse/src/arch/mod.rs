//! The Synapse Architecture — Beyond Transformers
//!
//! Brain-inspired modular architecture where specialized modules
//! communicate through a learned routing system (Thalamus).
//!
//! Key principles:
//! - No O(n²) attention anywhere — Mamba (state space) + xLSTM (recurrent)
//! - Sparse activation — only 2-3 of 8+ modules fire per token
//! - Fast-weight memory — learn new facts in ONE forward pass
//! - Modular training — update individual modules without catastrophic forgetting
//!
//! Total active params per token: ~800M even with 3B+ total system params.

pub mod mamba;
pub mod xlstm;
pub mod thalamus;
pub mod expert;
pub mod fast_weights;
pub mod synapse_model;

pub use synapse_model::SynapseModel;
pub use thalamus::Thalamus;

/// Linear projection that handles 2D and 3D tensors
/// x: (*, d_in), weight: (d_out, d_in) → (*, d_out)
pub(crate) fn linear(x: &candle_core::Tensor, weight: &candle_core::Tensor) -> anyhow::Result<candle_core::Tensor> {
    let w_t = weight.t()?;
    let dims = x.dims();
    if dims.len() == 2 {
        Ok(x.matmul(&w_t)?)
    } else if dims.len() == 3 {
        let (b, l, d) = (dims[0], dims[1], dims[2]);
        let flat = x.reshape(&[b * l, d])?;
        let out = flat.matmul(&w_t)?;
        let d_out = out.dims()[1];
        Ok(out.reshape(&[b, l, d_out])?)
    } else {
        anyhow::bail!("linear: expected 2D or 3D tensor, got {}D", dims.len())
    }
}

/// Linear projection with bias
pub(crate) fn linear_bias(
    x: &candle_core::Tensor,
    weight: &candle_core::Tensor,
    bias: &candle_core::Tensor,
) -> anyhow::Result<candle_core::Tensor> {
    let w_t = weight.t()?;
    let dims = x.dims();
    if dims.len() == 2 {
        let out = x.matmul(&w_t)?;
        Ok(out.broadcast_add(bias)?)
    } else if dims.len() == 3 {
        let (b, l, d) = (dims[0], dims[1], dims[2]);
        let flat = x.reshape(&[b * l, d])?;
        let out = flat.matmul(&w_t)?;
        let out = out.broadcast_add(bias)?;
        let d_out = weight.dims()[0];
        Ok(out.reshape(&[b, l, d_out])?)
    } else {
        anyhow::bail!("linear_bias: expected 2D or 3D tensor, got {}D", dims.len())
    }
}
