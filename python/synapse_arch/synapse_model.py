"""SynapseModel — The Full Architecture.

Faithful PyTorch mirror of crates/synapse/src/arch/synapse_model.rs.
Embedding -> N layers of (xLSTM -> Thalamus+Experts -> FastWeights) -> LM head.
"""

import torch
import torch.nn as nn

from .config import SynapseModelConfig
from .xlstm import XLSTMLayer, XLSTMConfig
from .thalamus import Thalamus, ThalamusConfig
from .expert import ExpertPool, ExpertPoolConfig
from .fast_weights import FastWeightMemory, FastWeightConfig


class RMSNorm(nn.Module):
    """RMS normalization — mirrors rms_norm() in synapse_model.rs."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return x_normed * self.weight


class SynapseLayer(nn.Module):
    """Single Synapse layer: xLSTM -> Thalamus+Experts -> FastWeights.

    Mirrors SynapseLayer in synapse_model.rs.
    """

    def __init__(self, config: SynapseModelConfig):
        super().__init__()

        self.xlstm = XLSTMLayer(XLSTMConfig(
            d_model=config.d_model,
            d_hidden=config.d_xlstm_hidden,
            n_heads=config.n_xlstm_heads,
            d_head=config.d_xlstm_hidden // config.n_xlstm_heads,
        ))

        self.thalamus = Thalamus(ThalamusConfig(
            d_model=config.d_model,
            n_experts=config.n_experts,
            top_k=config.top_k,
            d_state=config.d_state,
            hebbian_learning=config.hebbian_learning,
        ))

        self.experts = ExpertPool(ExpertPoolConfig(
            d_model=config.d_model,
            d_expert=config.d_expert,
            n_experts=config.n_experts,
        ))

        self.memory = None
        if config.use_fast_weights:
            self.memory = FastWeightMemory(FastWeightConfig(
                d_key=config.d_memory,
                d_value=config.d_memory,
                d_model=config.d_model,
                n_heads=config.n_memory_heads,
            ))

        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)
        self.norm3 = RMSNorm(config.d_model)

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        """Forward pass with residual connections.

        Order matches synapse_model.rs: xLSTM -> Thalamus+Experts -> Memory.
        """
        # 1. xLSTM language processing + residual
        normed = self.norm1(x)
        xlstm_out = self.xlstm(normed)
        x = x + xlstm_out

        # 2. Thalamus routing -> Expert processing + residual
        normed = self.norm2(x)
        routing_weights, expert_indices = self.thalamus(normed, training=training)
        expert_out = self.experts(normed, routing_weights, expert_indices)
        x = x + expert_out

        # 3. Fast-weight memory (optional) + residual
        if self.memory is not None:
            normed = self.norm3(x)
            mem_out = self.memory(normed)
            x = x + mem_out

        return x

    def reset_state(self):
        self.xlstm.reset_state()
        self.thalamus.reset_state()
        if self.memory is not None:
            self.memory.clear_memory()


class SynapseModel(nn.Module):
    """The full Synapse model — mirrors SynapseModel in synapse_model.rs.

    Embedding -> N x SynapseLayer -> RMSNorm -> LM Head.
    """

    def __init__(self, config: SynapseModelConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Synapse layers
        self.layers = nn.ModuleList([
            SynapseLayer(config) for _ in range(config.n_layers)
        ])

        # Final norm
        self.final_norm = RMSNorm(config.d_model)

        # LM head (separate from embedding, not tied)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

        print(
            f"SynapseModel: {config.total_params() / 1e6:.0f}M total, "
            f"{config.active_params_per_token() / 1e6:.0f}M active/token, "
            f"{config.n_layers} layers"
        )

    def _init_weights(self, module: nn.Module):
        """Initialize weights — matches Rust Xavier/Glorot init."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        training: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: (batch, seq_len) token IDs
            labels: (batch, seq_len) target IDs for loss computation
            training: whether we're training (affects routing noise)

        Returns:
            dict with 'logits' and optionally 'loss'
        """
        # Embed tokens
        x = self.embedding(input_ids)  # (batch, seq_len, d_model)

        # Process through Synapse layers
        for layer in self.layers:
            x = layer(x, training=training)

        # Final norm + LM head
        x = self.final_norm(x)
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)

        result = {"logits": logits}

        # Compute loss if labels provided
        if labels is not None:
            # Shift: predict next token
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss

        return result

    def reset_state(self):
        """Reset all running state (between sequences)."""
        for layer in self.layers:
            layer.reset_state()

    def param_count(self) -> dict[str, int]:
        """Count parameters by component."""
        counts = {}
        total = 0
        for name, param in self.named_parameters():
            component = name.split(".")[0]
            if component == "layers":
                # Get subcomponent: layers.0.xlstm, layers.0.thalamus, etc.
                parts = name.split(".")
                if len(parts) >= 3:
                    component = f"layers.{parts[2]}"
            counts[component] = counts.get(component, 0) + param.numel()
            total += param.numel()
        counts["total"] = total
        return counts
