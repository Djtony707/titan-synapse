"""Thalamus — Brain-Inspired Router with Hebbian Learning.

Faithful PyTorch mirror of crates/synapse/src/arch/thalamus.rs.
Mamba backbone -> linear router -> softmax -> top-k selection.
"""

from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba import MambaLayer, MambaConfig


@dataclass
class ThalamusConfig:
    d_model: int = 768
    n_experts: int = 8
    top_k: int = 2
    d_state: int = 16
    noise_std: float = 0.1
    hebbian_learning: bool = True
    hebbian_lr: float = 0.01


class Thalamus(nn.Module):
    """Mamba-based router with Hebbian pathway learning.

    Mirrors thalamus.rs Thalamus exactly.
    """

    def __init__(self, config: ThalamusConfig):
        super().__init__()
        self.config = config

        # Mamba backbone for context-aware routing
        self.mamba = MambaLayer(MambaConfig(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=4,
            expand=2,
        ))

        # Router head: d_model -> n_experts
        self.router_proj = nn.Linear(config.d_model, config.n_experts, bias=True)
        nn.init.zeros_(self.router_proj.bias)

        # Expert names (for introspection, not part of gradient graph)
        self.expert_names = [
            "language", "reasoning", "code", "math",
            "memory", "creative", "analysis", "planning",
        ][:config.n_experts]
        while len(self.expert_names) < config.n_experts:
            self.expert_names.append(f"expert_{len(self.expert_names)}")

        # Hebbian strengths (not part of gradient graph)
        self.hebbian_strengths: dict[str, float] = defaultdict(float)

    def forward(
        self, x: torch.Tensor, training: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass — route tokens to experts.

        Args:
            x: (batch, seq_len, d_model)
            training: if True, add noise for load balancing

        Returns:
            routing_weights: (batch, seq_len, top_k) — normalized weights
            expert_indices: (batch, seq_len, top_k) — selected expert indices
        """
        batch, seq_len, _ = x.shape

        # Context-aware routing via Mamba backbone
        context = self.mamba(x)  # (batch, seq_len, d_model)

        # Routing logits
        logits = self.router_proj(context)  # (batch, seq_len, n_experts)

        # Add noise during training for load balancing
        if training and self.config.noise_std > 0:
            noise = torch.randn_like(logits) * self.config.noise_std
            logits = logits + noise

        # Softmax over experts
        routing_probs = F.softmax(logits, dim=-1)  # (batch, seq_len, n_experts)

        # Top-k selection
        top_k_weights, top_k_indices = torch.topk(
            routing_probs, self.config.top_k, dim=-1
        )  # both (batch, seq_len, top_k)

        # Normalize weights to sum to 1
        routing_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # Hebbian learning (inference only, not part of gradient graph)
        if self.config.hebbian_learning and not training:
            self._update_hebbian(top_k_indices)

        return routing_weights, top_k_indices

    def _update_hebbian(self, indices: torch.Tensor):
        """Update Hebbian pathway strengths — fire together, wire together."""
        # indices: (batch, seq_len, top_k)
        for b in range(indices.shape[0]):
            for t in range(indices.shape[1]):
                experts = sorted(indices[b, t].tolist())
                pathway = "+".join(
                    self.expert_names[i] if i < len(self.expert_names) else f"expert_{i}"
                    for i in experts
                )
                self.hebbian_strengths[pathway] += self.config.hebbian_lr
                # Decay all
                for key in self.hebbian_strengths:
                    self.hebbian_strengths[key] *= 0.999

    def reset_state(self):
        self.mamba.reset_state()

    def get_load_balance_loss(
        self, routing_probs: torch.Tensor, expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """Auxiliary load balancing loss for training.

        Encourages even distribution of tokens across experts.
        Based on Switch Transformer load balance loss.
        """
        n_experts = self.config.n_experts
        # Fraction of tokens routed to each expert
        # expert_indices: (batch, seq_len, top_k) — one-hot and sum
        one_hot = F.one_hot(expert_indices, n_experts).float()  # (B, L, K, E)
        tokens_per_expert = one_hot.sum(dim=(0, 1, 2))  # (E,)
        fraction = tokens_per_expert / tokens_per_expert.sum()

        # Average routing probability per expert
        avg_prob = routing_probs.mean(dim=(0, 1))  # (E,)

        # Loss = n_experts * sum(fraction * avg_prob) — want this to be 1.0
        return n_experts * (fraction * avg_prob).sum()
