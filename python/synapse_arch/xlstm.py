"""xLSTM — Extended Long Short-Term Memory (mLSTM variant).

Faithful PyTorch mirror of crates/synapse/src/arch/xlstm.rs.
Exponential gating + matrix cell state for O(n) associative memory.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class XLSTMConfig:
    d_model: int = 768
    d_hidden: int = 1536
    n_heads: int = 4
    d_head: int = 384  # d_hidden // n_heads


class XLSTMLayer(nn.Module):
    """Extended LSTM with exponential gating and matrix memory.

    Mirrors xlstm.rs XLSTMLayer exactly.
    """

    def __init__(self, config: XLSTMConfig):
        super().__init__()
        self.config = config
        d, h = config.d_model, config.d_hidden

        # Gate projections
        self.w_i = nn.Linear(d, h, bias=True)   # input gate (exponential)
        self.w_f = nn.Linear(d, h, bias=True)   # forget gate (exponential)
        self.w_o = nn.Linear(d, h, bias=True)   # output gate (sigmoid)

        # Initialize forget gate bias to 1.0 (bias toward remembering)
        nn.init.ones_(self.w_f.bias)
        nn.init.zeros_(self.w_i.bias)
        nn.init.zeros_(self.w_o.bias)

        # Memory projections
        self.w_k = nn.Linear(d, h, bias=False)
        self.w_v = nn.Linear(d, h, bias=False)
        self.w_q = nn.Linear(d, h, bias=False)

        # Output projection: d_hidden -> d_model
        self.w_out = nn.Linear(h, d, bias=False)

        # Running state
        self.cell_state: Optional[torch.Tensor] = None
        self.normalizer_state: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (batch, seq_len, d_model) -> (batch, seq_len, d_model)."""
        batch, seq_len, _ = x.shape
        n_heads = self.config.n_heads
        d_head = self.config.d_head

        # Initialize states
        if self.cell_state is not None:
            c = self.cell_state
        else:
            c = torch.zeros(batch, n_heads, d_head, d_head, device=x.device, dtype=x.dtype)

        if self.normalizer_state is not None:
            n = self.normalizer_state
        else:
            n = torch.ones(batch, n_heads, d_head, device=x.device, dtype=x.dtype)

        outputs = []

        for t in range(seq_len):
            x_t = x[:, t]  # (batch, d_model)

            # Compute gates
            i_pre = self.w_i(x_t)  # (batch, d_hidden)
            f_pre = self.w_f(x_t)
            o_pre = self.w_o(x_t)

            # Exponential gating (clamp to prevent overflow)
            i_gate = torch.exp(torch.clamp(i_pre, -20.0, 20.0))
            f_gate = torch.exp(torch.clamp(f_pre, -20.0, 20.0))
            o_gate = torch.sigmoid(o_pre)

            # Key, value, query
            k = self.w_k(x_t).reshape(batch, n_heads, d_head)
            v = self.w_v(x_t).reshape(batch, n_heads, d_head)
            q = self.w_q(x_t).reshape(batch, n_heads, d_head)
            i_g = i_gate.reshape(batch, n_heads, d_head)
            f_g = f_gate.reshape(batch, n_heads, d_head)

            # Matrix memory update: C = f * C + i * (v outer k)
            outer = v.unsqueeze(-1) * k.unsqueeze(2)  # (batch, n_heads, d_head, d_head)
            write = outer * i_g.unsqueeze(-1)  # scale by input gate
            c = c * f_g.unsqueeze(-1) + write

            # Update normalizer: n = f * n + i * k
            n = f_g * n + i_g * k

            # Read from memory: h = C @ q / max(|n . q|, 1)
            q_col = q.unsqueeze(-1)  # (batch, n_heads, d_head, 1)
            read = torch.matmul(c, q_col).squeeze(-1)  # (batch, n_heads, d_head)

            nq = (n * q).sum(-1)  # (batch, n_heads)
            normalizer = torch.clamp(nq.abs(), min=1.0).unsqueeze(-1)  # (batch, n_heads, 1)
            h_read = read / normalizer

            # Apply output gate
            o_g = o_gate.reshape(batch, n_heads, d_head)
            h_gated = h_read * o_g

            # Flatten heads and project to output
            h_flat = h_gated.reshape(batch, self.config.d_hidden)
            y_t = self.w_out(h_flat)

            outputs.append(y_t.unsqueeze(1))

        # Save state (detach for inference; for training we don't persist)
        self.cell_state = c.detach()
        self.normalizer_state = n.detach()

        return torch.cat(outputs, dim=1)

    def reset_state(self):
        self.cell_state = None
        self.normalizer_state = None
