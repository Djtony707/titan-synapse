"""Expert — Sparse Mixture of Experts with SwiGLU.

Faithful PyTorch mirror of crates/synapse/src/arch/expert.rs.
Each expert is a gated FFN. Only top-k activate per token.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ExpertPoolConfig:
    d_model: int = 768
    d_expert: int = 3072
    n_experts: int = 8


class Expert(nn.Module):
    """Single expert: SwiGLU feed-forward network.

    Mirrors expert.rs Expert: down(silu(gate(x)) * up(x))
    """

    def __init__(self, d_model: int, d_expert: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_expert, bias=False)
        self.w_up = nn.Linear(d_model, d_expert, bias=False)
        self.w_down = nn.Linear(d_expert, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: (tokens, d_model) -> (tokens, d_model)."""
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class ExpertPool(nn.Module):
    """Pool of sparse experts — mirrors expert.rs ExpertPool.

    Takes routing weights and indices from Thalamus,
    runs only selected experts per token, combines outputs.
    """

    def __init__(self, config: ExpertPoolConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            Expert(config.d_model, config.d_expert)
            for _ in range(config.n_experts)
        ])

    def forward(
        self,
        x: torch.Tensor,
        routing_weights: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with sparse routing.

        Args:
            x: (batch, seq_len, d_model)
            routing_weights: (batch, seq_len, top_k) — normalized weights
            expert_indices: (batch, seq_len, top_k) — expert indices

        Returns:
            (batch, seq_len, d_model) — weighted sum of expert outputs
        """
        batch, seq_len, d_model = x.shape
        top_k = expert_indices.shape[-1]

        # Flatten batch and seq for efficient processing
        x_flat = x.reshape(-1, d_model)  # (B*L, d_model)
        indices_flat = expert_indices.reshape(-1, top_k)  # (B*L, top_k)
        weights_flat = routing_weights.reshape(-1, top_k)  # (B*L, top_k)

        # Output accumulator
        output = torch.zeros_like(x_flat)  # (B*L, d_model)

        # Process each expert: gather tokens assigned to it, run, scatter back
        for expert_idx in range(self.config.n_experts):
            # Find which (token, k) pairs route to this expert
            mask = (indices_flat == expert_idx)  # (B*L, top_k)

            if not mask.any():
                continue

            # Get the tokens and weights for this expert
            # For each token, check if any of its top_k slots picked this expert
            token_mask = mask.any(dim=-1)  # (B*L,)
            if not token_mask.any():
                continue

            token_indices = token_mask.nonzero(as_tuple=True)[0]  # which tokens
            expert_input = x_flat[token_indices]  # (N, d_model)

            # Run expert
            expert_output = self.experts[expert_idx](expert_input)  # (N, d_model)

            # Get the weight for this expert for each token
            # Sum weights across all k positions that selected this expert
            token_weights = (weights_flat[token_indices] * mask[token_indices].float()).sum(-1)
            weighted_output = expert_output * token_weights.unsqueeze(-1)

            # Scatter back
            output.index_add_(0, token_indices, weighted_output)

        return output.reshape(batch, seq_len, d_model)
