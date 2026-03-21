"""Synapse Architecture — PyTorch implementation for training.

Faithful mirror of the Rust/candle implementation in crates/synapse/src/arch/.
Train here, export SafeTensors, load in Rust for inference.
"""

from .config import SynapseModelConfig
from .mamba import MambaLayer, MambaConfig
from .xlstm import XLSTMLayer, XLSTMConfig
from .thalamus import Thalamus, ThalamusConfig
from .expert import Expert, ExpertPool, ExpertPoolConfig
from .fast_weights import FastWeightMemory, FastWeightConfig
from .synapse_model import SynapseModel

__all__ = [
    "SynapseModelConfig",
    "SynapseModel",
    "MambaLayer", "MambaConfig",
    "XLSTMLayer", "XLSTMConfig",
    "Thalamus", "ThalamusConfig",
    "Expert", "ExpertPool", "ExpertPoolConfig",
    "FastWeightMemory", "FastWeightConfig",
]
