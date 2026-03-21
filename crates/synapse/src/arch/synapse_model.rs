//! SynapseModel — The Full Architecture
//!
//! Combines all components into a single model:
//!   - Thalamus (Mamba-based router) decides which modules to activate
//!   - xLSTM language module handles syntax and grammar
//!   - Sparse MoE expert pool provides specialized knowledge
//!   - Fast-weight memory learns new facts during inference
//!   - RMSNorm + residual connections for stable training
//!
//! The brain isn't one neural network — it's a modular system with
//! specialized regions connected by learned routing. This is the same.
//!
//! OBSERVABILITY: The `SynapseIntrospection` struct gives you a complete
//! view into the model's decision-making at every layer. No black box.
//! See which modules fired, what was remembered, what was forgotten,
//! and how confident the model is in its output.

use anyhow::Result;
use candle_core::{Device, Tensor, DType, D};

use super::mamba::MambaConfig;
use super::xlstm::{XLSTMConfig, XLSTMLayer, XLSTMIntrospection};
use super::thalamus::{ThalamusConfig, Thalamus, ThalamusIntrospection};
use super::expert::{ExpertPoolConfig, ExpertPool, ExpertPoolIntrospection};
use super::fast_weights::{FastWeightConfig, FastWeightMemory, FastWeightIntrospection};
use super::linear;

/// Full model configuration
#[derive(Debug, Clone)]
pub struct SynapseModelConfig {
    /// Token embedding dimension
    pub d_model: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Number of Synapse layers (each layer = thalamus + xlstm + experts + memory)
    pub n_layers: usize,
    /// Number of experts per layer
    pub n_experts: usize,
    /// Top-k experts to activate per token
    pub top_k: usize,
    /// Expert FFN hidden dimension
    pub d_expert: usize,
    /// xLSTM hidden dimension
    pub d_xlstm_hidden: usize,
    /// xLSTM heads
    pub n_xlstm_heads: usize,
    /// Fast-weight key/value dimension
    pub d_memory: usize,
    /// Fast-weight heads
    pub n_memory_heads: usize,
    /// Mamba state dimension (for Thalamus)
    pub d_state: usize,
    /// Enable Hebbian routing
    pub hebbian_learning: bool,
    /// Enable fast-weight memory
    pub use_fast_weights: bool,
    /// Device
    pub device: Device,
}

impl Default for SynapseModelConfig {
    fn default() -> Self {
        Self {
            d_model: 768,
            vocab_size: 151936, // Qwen2.5 vocab size
            n_layers: 12,
            n_experts: 8,
            top_k: 2,
            d_expert: 3072,
            d_xlstm_hidden: 1536,
            n_xlstm_heads: 4,
            d_memory: 64,
            n_memory_heads: 8,
            d_state: 16,
            hebbian_learning: true,
            use_fast_weights: true,
            device: Device::Cpu,
        }
    }
}

impl SynapseModelConfig {
    /// Estimated total parameters
    pub fn total_params(&self) -> usize {
        let embedding = self.vocab_size * self.d_model * 2; // in + out
        let per_layer =
            // Thalamus (Mamba + router head)
            self.d_model * self.d_model * 2 * 2 + // Mamba in/out proj
            self.d_model * self.n_experts +        // router head
            // xLSTM
            self.d_model * self.d_xlstm_hidden * 6 + // gates + kqv
            self.d_xlstm_hidden * self.d_model +     // output proj
            // Experts
            self.n_experts * (self.d_model * self.d_expert * 3) + // up/gate/down per expert
            // Fast weights (projections only — fast weights themselves are dynamic)
            self.d_model * self.n_memory_heads * self.d_memory * 4; // key/val/query/gate
        embedding + per_layer * self.n_layers
    }

    /// Estimated active parameters per token (with top-k routing)
    pub fn active_params_per_token(&self) -> usize {
        let embedding = self.vocab_size * self.d_model * 2;
        let per_layer =
            self.d_model * self.d_model * 2 * 2 + // Thalamus (always active)
            self.d_model * self.d_xlstm_hidden * 6 + self.d_xlstm_hidden * self.d_model + // xLSTM (always active)
            self.top_k * (self.d_model * self.d_expert * 3) + // Only top-k experts active
            self.d_model * self.n_memory_heads * self.d_memory * 4; // Fast weights (always active)
        embedding + per_layer * self.n_layers
    }

    /// Estimated VRAM in MB (FP32)
    pub fn estimated_vram_mb(&self) -> f64 {
        (self.total_params() * 4) as f64 / (1024.0 * 1024.0)
    }
}

/// A single Synapse layer: Thalamus → xLSTM → Experts → Memory → RMSNorm
struct SynapseLayer {
    thalamus: Thalamus,
    xlstm: XLSTMLayer,
    experts: ExpertPool,
    memory: Option<FastWeightMemory>,
    // RMSNorm parameters
    norm1_weight: Tensor,
    norm2_weight: Tensor,
    norm3_weight: Tensor,
    d_model: usize,
}

impl SynapseLayer {
    fn new(config: &SynapseModelConfig) -> Result<Self> {
        let dev = &config.device;

        let thalamus = Thalamus::new(ThalamusConfig {
            d_model: config.d_model,
            n_experts: config.n_experts,
            top_k: config.top_k,
            d_state: config.d_state,
            hebbian_learning: config.hebbian_learning,
            device: dev.clone(),
            ..Default::default()
        })?;

        let xlstm = XLSTMLayer::new(XLSTMConfig {
            d_model: config.d_model,
            d_hidden: config.d_xlstm_hidden,
            n_heads: config.n_xlstm_heads,
            d_head: config.d_xlstm_hidden / config.n_xlstm_heads,
            device: dev.clone(),
        })?;

        let experts = ExpertPool::new(ExpertPoolConfig {
            d_model: config.d_model,
            d_expert: config.d_expert,
            n_experts: config.n_experts,
            device: dev.clone(),
        })?;

        let memory = if config.use_fast_weights {
            Some(FastWeightMemory::new(FastWeightConfig {
                d_key: config.d_memory,
                d_value: config.d_memory,
                d_model: config.d_model,
                n_heads: config.n_memory_heads,
                device: dev.clone(),
                ..Default::default()
            })?)
        } else {
            None
        };

        let norm1_weight = Tensor::ones(&[config.d_model], DType::F32, dev)?;
        let norm2_weight = Tensor::ones(&[config.d_model], DType::F32, dev)?;
        let norm3_weight = Tensor::ones(&[config.d_model], DType::F32, dev)?;

        Ok(Self {
            thalamus,
            xlstm,
            experts,
            memory,
            norm1_weight,
            norm2_weight,
            norm3_weight,
            d_model: config.d_model,
        })
    }

    fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        // 1. xLSTM language processing with residual
        let normed = rms_norm(x, &self.norm1_weight)?;
        let xlstm_out = self.xlstm.forward(&normed)?;
        let x = (x + &xlstm_out)?;

        // 2. Thalamus routing → Expert processing with residual
        let normed = rms_norm(&x, &self.norm2_weight)?;
        let (routing_weights, expert_indices) = self.thalamus.forward(&normed)?;
        let expert_out = self.experts.forward(&normed, &routing_weights, &expert_indices)?;
        let x = (&x + &expert_out)?;

        // 3. Fast-weight memory (optional) with residual
        if let Some(ref mut memory) = self.memory {
            let normed = rms_norm(&x, &self.norm3_weight)?;
            let mem_out = memory.forward(&normed)?;
            let x = (&x + &mem_out)?;
            Ok(x)
        } else {
            Ok(x)
        }
    }

    fn reset_state(&mut self) {
        self.thalamus.reset_state();
        self.xlstm.reset_state();
        if let Some(ref mut memory) = self.memory {
            memory.clear_memory();
        }
    }
}

/// Per-layer introspection
#[derive(Debug, Clone)]
pub struct LayerIntrospection {
    pub layer_idx: usize,
    pub thalamus: Option<ThalamusIntrospection>,
    pub xlstm: Option<XLSTMIntrospection>,
    pub experts: Option<ExpertPoolIntrospection>,
    pub memory: Option<FastWeightIntrospection>,
}

/// Complete model introspection — the anti-black-box
#[derive(Debug, Clone)]
pub struct SynapseIntrospection {
    /// Per-layer introspection data
    pub layers: Vec<LayerIntrospection>,
    /// Total parameters
    pub total_params: usize,
    /// Active parameters per token
    pub active_params: usize,
    /// Activation sparsity (what % of params are dormant)
    pub sparsity: f32,
    /// Overall routing confidence
    pub routing_confidence: f32,
    /// Top Hebbian pathways across all layers
    pub global_pathways: Vec<(String, f32)>,
    /// Memory utilization summary
    pub memory_status: String,
    /// Human-readable summary
    pub summary: String,
}

/// The SYNAPSE Model — Beyond Transformers
///
/// A modular, brain-inspired architecture that replaces monolithic
/// transformer blocks with specialized modules + learned routing.
///
/// Key properties:
/// - O(n) complexity (no quadratic attention)
/// - Sparse activation (only top-k experts per token)
/// - In-context learning via fast weights
/// - Full observability — every decision is transparent
pub struct SynapseModel {
    config: SynapseModelConfig,

    /// Token embedding
    embedding: Tensor,
    /// Position encoding (learned)
    pos_embedding: Option<Tensor>,
    /// The Synapse layers
    layers: Vec<SynapseLayer>,
    /// Final normalization
    final_norm: Tensor,
    /// Language model head: d_model → vocab_size
    lm_head: Tensor,

    /// Last introspection
    last_introspection: Option<SynapseIntrospection>,
}

impl SynapseModel {
    /// Create a new Synapse model
    pub fn new(config: SynapseModelConfig) -> Result<Self> {
        let dev = &config.device;

        // Token embedding
        let scale = (1.0 / config.d_model as f64).sqrt() as f32;
        let embedding = Tensor::randn(
            0f32, scale,
            &[config.vocab_size, config.d_model], dev
        )?;

        // Position embedding (optional, for short sequences)
        let pos_embedding = None; // Using RoPE-style or none initially

        // Build layers
        let mut layers = Vec::with_capacity(config.n_layers);
        for _ in 0..config.n_layers {
            layers.push(SynapseLayer::new(&config)?);
        }

        // Final norm
        let final_norm = Tensor::ones(&[config.d_model], DType::F32, dev)?;

        // LM head (tied with embedding transpose)
        let lm_head = Tensor::randn(
            0f32, scale,
            &[config.vocab_size, config.d_model], dev
        )?;

        tracing::info!(
            "SynapseModel created: {} total params ({:.0}M), {} active per token ({:.0}M), {} layers",
            config.total_params(),
            config.total_params() as f64 / 1e6,
            config.active_params_per_token(),
            config.active_params_per_token() as f64 / 1e6,
            config.n_layers,
        );

        Ok(Self {
            layers,
            embedding,
            pos_embedding,
            final_norm,
            lm_head,
            last_introspection: None,
            config,
        })
    }

    /// Forward pass — process token IDs through the full model
    ///
    /// Input: (batch, seq_len) — token IDs
    /// Output: (batch, seq_len, vocab_size) — logits
    pub fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        let (batch, seq_len) = input_ids.dims2()?;

        // Embed tokens: flatten input_ids to 1D for index_select, then reshape
        let flat_ids = input_ids.reshape(&[batch * seq_len])?;
        let x = self.embedding.index_select(&flat_ids, 0)?;
        let mut x = x.reshape(&[batch, seq_len, self.config.d_model])?;

        // Process through layers
        for layer in &mut self.layers {
            x = layer.forward(&x)?;
        }

        // Final norm
        x = rms_norm(&x, &self.final_norm)?;

        // LM head → logits
        let logits = linear(&x, &self.lm_head)?;

        // Build introspection
        self.build_introspection();

        Ok(logits)
    }

    /// Build full introspection from all layers
    fn build_introspection(&mut self) {
        let mut layer_intros = Vec::with_capacity(self.layers.len());
        let mut all_pathways: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
        let mut total_confidence = 0f32;
        let mut confidence_count = 0;

        for (i, layer) in self.layers.iter().enumerate() {
            let thalamus_intro = layer.thalamus.introspect().cloned();
            let xlstm_intro = layer.xlstm.introspect().cloned();
            let expert_intro = layer.experts.introspect().cloned();
            let memory_intro = layer.memory.as_ref().and_then(|m| m.introspect().cloned());

            // Aggregate pathways
            if let Some(ref ti) = thalamus_intro {
                total_confidence += ti.avg_confidence;
                confidence_count += 1;
                for (path, strength) in &ti.top_pathways {
                    *all_pathways.entry(path.clone()).or_insert(0.0) += strength;
                }
            }

            layer_intros.push(LayerIntrospection {
                layer_idx: i,
                thalamus: thalamus_intro,
                xlstm: xlstm_intro,
                experts: expert_intro,
                memory: memory_intro,
            });
        }

        let avg_confidence = if confidence_count > 0 {
            total_confidence / confidence_count as f32
        } else {
            0.0
        };

        let mut global_pathways: Vec<(String, f32)> = all_pathways.into_iter().collect();
        global_pathways.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        global_pathways.truncate(10);

        let total_params = self.config.total_params();
        let active_params = self.config.active_params_per_token();
        let sparsity = 1.0 - (active_params as f32 / total_params as f32);

        // Memory status
        let memory_status = self.layers.iter()
            .filter_map(|l| l.memory.as_ref())
            .filter_map(|m| m.introspect())
            .map(|i| format!("{}steps/~{}facts", i.total_steps, i.estimated_facts_stored))
            .collect::<Vec<_>>()
            .join(", ");

        let summary = format!(
            "Synapse Model: {:.0}M total / {:.0}M active ({:.0}% sparse)\n\
             Routing confidence: {:.1}\n\
             Top pathway: {}\n\
             Memory: {}",
            total_params as f64 / 1e6,
            active_params as f64 / 1e6,
            sparsity * 100.0,
            avg_confidence,
            global_pathways.first().map(|(p, s)| format!("{} ({:.3})", p, s)).unwrap_or("none".into()),
            if memory_status.is_empty() { "disabled".into() } else { memory_status.clone() },
        );

        self.last_introspection = Some(SynapseIntrospection {
            layers: layer_intros,
            total_params,
            active_params,
            sparsity,
            routing_confidence: avg_confidence,
            global_pathways,
            memory_status,
            summary,
        });
    }

    /// Get full model introspection — the anti-black-box
    pub fn introspect(&self) -> Option<&SynapseIntrospection> {
        self.last_introspection.as_ref()
    }

    /// Reset all running state (between sequences)
    pub fn reset_state(&mut self) {
        for layer in &mut self.layers {
            layer.reset_state();
        }
    }

    /// Get model config
    pub fn config(&self) -> &SynapseModelConfig {
        &self.config
    }

    /// Human-readable model summary
    pub fn summary(&self) -> String {
        format!(
            "╔══════════════════════════════════════════════════╗\n\
             ║          SYNAPSE MODEL — Beyond Transformers      ║\n\
             ╠══════════════════════════════════════════════════╣\n\
             ║  Layers:          {:>4}                            ║\n\
             ║  Total params:    {:>8.1}M                        ║\n\
             ║  Active/token:    {:>8.1}M                        ║\n\
             ║  Sparsity:        {:>7.1}%                        ║\n\
             ║  Experts:         {:>4} (top-{} active)           ║\n\
             ║  xLSTM heads:     {:>4}                            ║\n\
             ║  Memory heads:    {:>4}                            ║\n\
             ║  Vocab size:      {:>6}                            ║\n\
             ║  Hebbian routing: {:>5}                            ║\n\
             ║  Fast weights:    {:>5}                            ║\n\
             ║  VRAM estimate:   {:>8.0}MB (FP32)               ║\n\
             ╚══════════════════════════════════════════════════╝",
            self.config.n_layers,
            self.config.total_params() as f64 / 1e6,
            self.config.active_params_per_token() as f64 / 1e6,
            (1.0 - self.config.active_params_per_token() as f64 / self.config.total_params() as f64) * 100.0,
            self.config.n_experts, self.config.top_k,
            self.config.n_xlstm_heads,
            self.config.n_memory_heads,
            self.config.vocab_size,
            if self.config.hebbian_learning { "ON" } else { "OFF" },
            if self.config.use_fast_weights { "ON" } else { "OFF" },
            self.config.estimated_vram_mb(),
        )
    }
}

/// RMSNorm: x * weight / sqrt(mean(x²) + eps)
fn rms_norm(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let eps = 1e-6;
    let variance = x.sqr()?.mean(D::Minus1)?; // (batch, seq_len)
    let eps_t = Tensor::new(&[eps as f32], x.device())?.broadcast_as(variance.shape())?;
    let inv_rms = (&variance + &eps_t)?.sqrt()?.recip()?;
    let inv_rms = inv_rms.unsqueeze(D::Minus1)?; // (batch, seq_len, 1)
    let normed = x.broadcast_mul(&inv_rms)?;
    normed.broadcast_mul(&weight.unsqueeze(0)?.unsqueeze(0)?)
        .map_err(|e| anyhow::anyhow!("{e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> SynapseModelConfig {
        SynapseModelConfig {
            d_model: 32,
            vocab_size: 100,
            n_layers: 2,
            n_experts: 4,
            top_k: 2,
            d_expert: 128,
            d_xlstm_hidden: 64,
            n_xlstm_heads: 2,
            d_memory: 16,
            n_memory_heads: 2,
            d_state: 4,
            hebbian_learning: true,
            use_fast_weights: true,
            device: Device::Cpu,
        }
    }

    #[test]
    fn test_model_creation() {
        let config = small_config();
        let model = SynapseModel::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_param_counting() {
        let config = small_config();
        assert!(config.total_params() > 0);
        assert!(config.active_params_per_token() > 0);
        assert!(config.active_params_per_token() <= config.total_params());
    }

    #[test]
    fn test_model_forward() {
        let config = small_config();
        let mut model = SynapseModel::new(config).unwrap();

        // Token IDs: (batch=1, seq_len=4)
        let input_ids = Tensor::new(&[1u32, 5, 10, 50], &Device::Cpu).unwrap()
            .reshape(&[1, 4]).unwrap();

        let logits = model.forward(&input_ids).unwrap();
        assert_eq!(logits.dims(), &[1, 4, 100]); // (batch, seq, vocab)
    }

    #[test]
    fn test_model_introspection() {
        let config = small_config();
        let mut model = SynapseModel::new(config).unwrap();

        let input_ids = Tensor::new(&[1u32, 5, 10, 50], &Device::Cpu).unwrap()
            .reshape(&[1, 4]).unwrap();
        let _ = model.forward(&input_ids).unwrap();

        let intro = model.introspect().unwrap();
        assert_eq!(intro.layers.len(), 2);
        assert!(intro.total_params > 0);
        assert!(intro.active_params > 0);
        assert!(intro.sparsity > 0.0);
        assert!(!intro.summary.is_empty());

        // Each layer should have thalamus introspection
        for layer in &intro.layers {
            assert!(layer.thalamus.is_some());
        }
    }

    #[test]
    fn test_model_summary() {
        let config = small_config();
        let model = SynapseModel::new(config).unwrap();
        let summary = model.summary();
        assert!(summary.contains("SYNAPSE MODEL"));
        assert!(summary.contains("Beyond Transformers"));
    }

    #[test]
    fn test_model_reset() {
        let config = small_config();
        let mut model = SynapseModel::new(config).unwrap();

        let input_ids = Tensor::new(&[1u32, 5, 10], &Device::Cpu).unwrap()
            .reshape(&[1, 3]).unwrap();
        let _ = model.forward(&input_ids).unwrap();

        // Reset should not panic
        model.reset_state();
    }
}
