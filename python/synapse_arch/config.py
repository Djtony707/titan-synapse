"""Synapse model configuration — mirrors SynapseModelConfig in synapse_model.rs."""

from dataclasses import dataclass


@dataclass
class SynapseModelConfig:
    """Full model configuration. Matches Rust SynapseModelConfig exactly."""

    d_model: int = 768
    vocab_size: int = 151936  # Qwen2.5 vocab size
    n_layers: int = 12
    n_experts: int = 8
    top_k: int = 2
    d_expert: int = 3072
    d_xlstm_hidden: int = 1536
    n_xlstm_heads: int = 4
    d_memory: int = 64
    n_memory_heads: int = 8
    d_state: int = 16
    hebbian_learning: bool = True
    use_fast_weights: bool = True

    @property
    def d_xlstm_head(self) -> int:
        return self.d_xlstm_hidden // self.n_xlstm_heads

    def total_params(self) -> int:
        """Estimated total parameters — matches Rust implementation."""
        embedding = self.vocab_size * self.d_model * 2  # in + out
        per_layer = (
            # Thalamus (Mamba + router head)
            self.d_model * self.d_model * 2 * 2  # Mamba in/out proj
            + self.d_model * self.n_experts  # router head
            # xLSTM
            + self.d_model * self.d_xlstm_hidden * 6  # gates + kqv
            + self.d_xlstm_hidden * self.d_model  # output proj
            # Experts
            + self.n_experts * (self.d_model * self.d_expert * 3)  # up/gate/down
            # Fast weights (projections only)
            + self.d_model * self.n_memory_heads * self.d_memory * 4  # key/val/query/gate
        )
        return embedding + per_layer * self.n_layers

    def active_params_per_token(self) -> int:
        """Active params with top-k routing — matches Rust implementation."""
        embedding = self.vocab_size * self.d_model * 2
        per_layer = (
            self.d_model * self.d_model * 2 * 2  # Thalamus always active
            + self.d_model * self.d_xlstm_hidden * 6
            + self.d_xlstm_hidden * self.d_model  # xLSTM always active
            + self.top_k * (self.d_model * self.d_expert * 3)  # Only top-k experts
            + self.d_model * self.n_memory_heads * self.d_memory * 4  # Fast weights always active
        )
        return embedding + per_layer * self.n_layers

    def estimated_vram_mb(self, bytes_per_param: int = 4) -> float:
        """Estimated VRAM in MB."""
        return (self.total_params() * bytes_per_param) / (1024 * 1024)

    @staticmethod
    def small_test() -> "SynapseModelConfig":
        """Small config for testing — matches Rust small_config()."""
        return SynapseModelConfig(
            d_model=32,
            vocab_size=100,
            n_layers=2,
            n_experts=4,
            top_k=2,
            d_expert=128,
            d_xlstm_hidden=64,
            n_xlstm_heads=2,
            d_memory=16,
            n_memory_heads=2,
            d_state=4,
        )

    @staticmethod
    def validation() -> "SynapseModelConfig":
        """512D/8L validation config for quick training runs."""
        return SynapseModelConfig(
            d_model=512,
            n_layers=8,
            n_experts=8,
            top_k=2,
            d_expert=2048,
            d_xlstm_hidden=1024,
            n_xlstm_heads=4,
            d_memory=64,
            n_memory_heads=8,
            d_state=16,
        )
