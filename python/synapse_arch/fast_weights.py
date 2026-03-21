"""Fast-Weight Memory — Learn During Inference.

Faithful PyTorch mirror of crates/synapse/src/arch/fast_weights.rs.
Outer-product memory: W += gate * strength * (k outer v), read = W^T @ q.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class FastWeightConfig:
    d_key: int = 64
    d_value: int = 64
    d_model: int = 768
    n_heads: int = 8
    decay: float = 0.95
    write_strength: float = 0.1
    retrieval_threshold: float = 0.01


class FastWeightMemory(nn.Module):
    """Fast-weight memory with outer-product writes and gated reads.

    Mirrors fast_weights.rs FastWeightMemory exactly.

    The fast_weights tensor is updated during forward pass (inference learning).
    The projection weights (w_key, w_value, w_query, w_gate, w_out) are
    trained via backprop. The fast_weights matrix itself is NOT part of
    the gradient graph.
    """

    def __init__(self, config: FastWeightConfig):
        super().__init__()
        self.config = config
        nh, dk, dv, d = config.n_heads, config.d_key, config.d_value, config.d_model

        # Trainable projections
        self.w_key = nn.Linear(d, nh * dk, bias=False)
        self.w_value = nn.Linear(d, nh * dv, bias=False)
        self.w_query = nn.Linear(d, nh * dk, bias=False)
        self.w_gate = nn.Linear(d, nh, bias=False)
        self.w_out = nn.Linear(nh * dv, d, bias=False)

        # Fast-weight matrix: NOT a parameter (updated at inference, not training)
        self.register_buffer(
            "fast_weights",
            torch.zeros(nh, dk, dv),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: (batch, seq_len, d_model) -> (batch, seq_len, d_model).

        Reads from fast-weight memory, then writes new memories.
        """
        batch, seq_len, _ = x.shape
        nh = self.config.n_heads
        dk = self.config.d_key
        dv = self.config.d_value

        outputs = []

        for t in range(seq_len):
            x_t = x[:, t]  # (batch, d_model)

            # Project to multi-head key, value, query
            k = self.w_key(x_t).reshape(batch, nh, dk)
            v = self.w_value(x_t).reshape(batch, nh, dv)
            q = self.w_query(x_t).reshape(batch, nh, dk)
            gate = torch.sigmoid(self.w_gate(x_t))  # (batch, nh)

            # === READ: retrieve from fast-weight memory ===
            # fast_weights: (nh, dk, dv) -> expand to (batch, nh, dk, dv)
            fw = self.fast_weights.unsqueeze(0).expand(batch, -1, -1, -1)
            # q: (batch, nh, dk) -> (batch, nh, dk, 1)
            q_col = q.unsqueeze(-1)
            # retrieved = fw^T @ q but fw is (dk, dv), so we want (dv,) output
            # Actually: (nh, dk, dv) * (nh, dk, 1) -> sum over dk -> (nh, dv)
            retrieved = (fw * q_col.expand_as(fw)).sum(dim=2)  # (batch, nh, dv)

            # Project retrieved to output
            retrieved_flat = retrieved.reshape(batch, nh * dv)
            mem_out = self.w_out(retrieved_flat)  # (batch, d_model)

            # Output = input + memory
            y_t = x_t + mem_out

            # === WRITE: update fast-weight memory ===
            # Decay existing weights
            self.fast_weights = self.fast_weights * self.config.decay

            # Outer product: k outer v, scaled by gate and write_strength
            # Use batch=0 for weight update (shared memory)
            k_0 = k[0]    # (nh, dk)
            v_0 = v[0]    # (nh, dv)
            gate_0 = gate[0]  # (nh,)

            outer = k_0.unsqueeze(-1) * v_0.unsqueeze(1)  # (nh, dk, dv)
            gate_scale = (gate_0 * self.config.write_strength).unsqueeze(-1).unsqueeze(-1)
            write = outer * gate_scale

            self.fast_weights = self.fast_weights + write.detach()

            outputs.append(y_t.unsqueeze(1))

        return torch.cat(outputs, dim=1)

    def clear_memory(self):
        """Clear all fast-weight memory."""
        self.fast_weights.zero_()
