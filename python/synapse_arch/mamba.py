"""Mamba — Selective State Space Model.

Faithful PyTorch mirror of crates/synapse/src/arch/mamba.rs.
O(n) replacement for O(n^2) attention.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MambaConfig:
    d_model: int = 768
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2


class MambaLayer(nn.Module):
    """Selective State Space Model (S6) — mirrors mamba.rs MambaLayer."""

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.d_inner = config.d_model * config.expand
        self.dt_rank = math.ceil(config.d_model / 16)

        # Input projection: d_model -> 2 * d_inner (x path + z gate)
        self.in_proj = nn.Linear(config.d_model, 2 * self.d_inner, bias=False)

        # Output projection: d_inner -> d_model
        self.out_proj = nn.Linear(self.d_inner, config.d_model, bias=False)

        # A matrix: log-space initialization for stability
        # a_log[i] = log(i+1) for i in 0..d_state, expanded to (d_inner, d_state)
        a_log_data = torch.log(torch.arange(1, config.d_state + 1, dtype=torch.float32))
        self.a_log = nn.Parameter(a_log_data.unsqueeze(0).expand(self.d_inner, -1).clone())

        # D: skip connection (ones)
        self.d = nn.Parameter(torch.ones(self.d_inner))

        # Selection mechanism: x -> (dt_rank + 2*d_state)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * config.d_state, bias=False)

        # dt projection: dt_rank -> d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        nn.init.zeros_(self.dt_proj.bias)

        # 1D causal depthwise convolution
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, config.d_conv,
            padding=config.d_conv - 1, groups=self.d_inner, bias=True,
        )
        nn.init.zeros_(self.conv1d.bias)

        # Running state
        self.ssm_state: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (batch, seq_len, d_model) -> (batch, seq_len, d_model)."""
        batch, seq_len, _ = x.shape

        # Project: (B, L, d_model) -> (B, L, 2*d_inner)
        xz = self.in_proj(x)
        x_path, z_path = xz.chunk(2, dim=-1)

        # Causal conv1d: transpose to (B, C, L), conv, truncate, transpose back
        x_conv = x_path.transpose(1, 2)  # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # causal: truncate future
        x_conv = x_conv.transpose(1, 2)  # (B, L, d_inner)

        # SiLU activation
        x_act = F.silu(x_conv)

        # Selective SSM
        y = self._selective_ssm(x_act, batch, seq_len)

        # Gate with z path
        z_act = F.silu(z_path)
        output = y * z_act

        # Output projection
        return self.out_proj(output)

    def _selective_ssm(self, x: torch.Tensor, batch: int, seq_len: int) -> torch.Tensor:
        """Selective SSM scan — sequential over time steps."""
        d_state = self.config.d_state

        # Project x to get dt, B, C
        x_dbc = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        dt, b, c = x_dbc.split([self.dt_rank, d_state, d_state], dim=-1)

        # Project dt to full dimension + softplus
        dt = F.softplus(self.dt_proj(dt))  # (B, L, d_inner)

        # A from log space (negative for stability)
        a = -torch.exp(self.a_log)  # (d_inner, d_state)

        # Initialize state
        if self.ssm_state is not None:
            h = self.ssm_state
        else:
            h = torch.zeros(batch, self.d_inner, d_state, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(seq_len):
            dt_t = dt[:, t]  # (B, d_inner)
            b_t = b[:, t]    # (B, d_state)
            c_t = c[:, t]    # (B, d_state)
            x_t = x[:, t]    # (B, d_inner)

            # Discretize: A_bar = exp(dt * A)
            dt_a = dt_t.unsqueeze(-1) * a.unsqueeze(0)  # (B, d_inner, d_state)
            a_bar = torch.exp(dt_a)

            # Discretize B: B_bar = dt * B
            db = dt_t.unsqueeze(-1) * b_t.unsqueeze(1)  # (B, d_inner, d_state)

            # State update: h = A_bar * h + B_bar * x
            bx = db * x_t.unsqueeze(-1)
            h = a_bar * h + bx

            # Output: y = C * h + D * x
            y_t = (h * c_t.unsqueeze(1)).sum(-1)  # (B, d_inner)
            y_t = y_t + x_t * self.d

            outputs.append(y_t.unsqueeze(1))

        self.ssm_state = h.detach()
        return torch.cat(outputs, dim=1)

    def reset_state(self):
        self.ssm_state = None
