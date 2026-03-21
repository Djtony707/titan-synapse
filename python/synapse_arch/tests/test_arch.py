"""Tests for PyTorch Synapse Architecture — mirrors Rust test cases."""

import torch
import pytest
from ..config import SynapseModelConfig
from ..mamba import MambaLayer, MambaConfig
from ..xlstm import XLSTMLayer, XLSTMConfig
from ..thalamus import Thalamus, ThalamusConfig
from ..expert import Expert, ExpertPool, ExpertPoolConfig
from ..fast_weights import FastWeightMemory, FastWeightConfig
from ..synapse_model import SynapseModel


# === Mamba tests (mirrors mamba.rs tests) ===

def test_mamba_creation():
    layer = MambaLayer(MambaConfig(d_model=64, d_state=8, d_conv=4, expand=2))
    assert layer is not None

def test_mamba_forward():
    layer = MambaLayer(MambaConfig(d_model=64, d_state=8, d_conv=4, expand=2))
    x = torch.randn(1, 8, 64)
    out = layer(x)
    assert out.shape == (1, 8, 64)

def test_mamba_state_persistence():
    layer = MambaLayer(MambaConfig(d_model=32, d_state=4, d_conv=4, expand=2))
    x = torch.randn(1, 4, 32)
    _ = layer(x)
    assert layer.ssm_state is not None
    layer.reset_state()
    assert layer.ssm_state is None


# === xLSTM tests (mirrors xlstm.rs tests) ===

def test_xlstm_creation():
    layer = XLSTMLayer(XLSTMConfig(d_model=64, d_hidden=128, n_heads=4, d_head=32))
    assert layer is not None

def test_xlstm_forward():
    layer = XLSTMLayer(XLSTMConfig(d_model=64, d_hidden=128, n_heads=4, d_head=32))
    x = torch.randn(1, 8, 64)
    out = layer(x)
    assert out.shape == (1, 8, 64)

def test_xlstm_state_persistence():
    layer = XLSTMLayer(XLSTMConfig(d_model=32, d_hidden=64, n_heads=2, d_head=32))
    x = torch.randn(1, 4, 32)
    _ = layer(x)
    assert layer.cell_state is not None
    layer.reset_state()
    assert layer.cell_state is None


# === Thalamus tests (mirrors thalamus.rs tests) ===

def test_thalamus_creation():
    router = Thalamus(ThalamusConfig(d_model=64, n_experts=4, top_k=2, d_state=8))
    assert router is not None

def test_thalamus_routing():
    router = Thalamus(ThalamusConfig(d_model=64, n_experts=4, top_k=2, d_state=8))
    x = torch.randn(1, 8, 64)
    weights, indices = router(x)
    assert weights.shape == (1, 8, 2)
    assert indices.shape == (1, 8, 2)
    assert (indices < 4).all()

def test_hebbian_learning():
    router = Thalamus(ThalamusConfig(
        d_model=32, n_experts=4, top_k=2, d_state=4,
        hebbian_learning=True, hebbian_lr=0.1,
    ))
    x = torch.randn(1, 4, 32)
    _ = router(x)
    assert len(router.hebbian_strengths) > 0


# === Expert tests (mirrors expert.rs tests) ===

def test_expert_creation():
    expert = Expert(64, 256)
    assert expert is not None

def test_expert_forward():
    expert = Expert(64, 256)
    x = torch.randn(4, 64)
    out = expert(x)
    assert out.shape == (4, 64)

def test_expert_pool():
    pool = ExpertPool(ExpertPoolConfig(d_model=64, d_expert=256, n_experts=4))
    assert len(pool.experts) == 4

def test_expert_pool_forward():
    pool = ExpertPool(ExpertPoolConfig(d_model=32, d_expert=128, n_experts=4))
    x = torch.randn(1, 4, 32)
    weights = torch.tensor([[[0.6, 0.4], [0.5, 0.5], [0.7, 0.3], [0.55, 0.45]]])
    indices = torch.tensor([[[0, 1], [1, 2], [0, 3], [2, 3]]])
    out = pool(x, weights, indices)
    assert out.shape == (1, 4, 32)


# === Fast-weight tests (mirrors fast_weights.rs tests) ===

def test_fast_weight_creation():
    mem = FastWeightMemory(FastWeightConfig(d_key=16, d_value=16, d_model=32, n_heads=2))
    assert mem is not None

def test_fast_weight_forward():
    mem = FastWeightMemory(FastWeightConfig(d_key=16, d_value=16, d_model=32, n_heads=2))
    x = torch.randn(1, 4, 32)
    out = mem(x)
    assert out.shape == (1, 4, 32)

def test_fast_weight_memory_persists():
    mem = FastWeightMemory(FastWeightConfig(
        d_key=16, d_value=16, d_model=32, n_heads=2,
        write_strength=1.0, decay=0.99,
    ))
    x = torch.randn(1, 4, 32)
    _ = mem(x)
    assert mem.fast_weights.norm().item() > 0
    mem.clear_memory()
    assert mem.fast_weights.norm().item() < 1e-10


# === SynapseModel tests (mirrors synapse_model.rs tests) ===

def test_model_creation():
    config = SynapseModelConfig.small_test()
    model = SynapseModel(config)
    assert model is not None

def test_param_counting():
    config = SynapseModelConfig.small_test()
    assert config.total_params() > 0
    assert config.active_params_per_token() > 0
    assert config.active_params_per_token() <= config.total_params()

def test_model_forward():
    config = SynapseModelConfig.small_test()
    model = SynapseModel(config)
    input_ids = torch.tensor([[1, 5, 10, 50]])
    result = model(input_ids)
    assert result["logits"].shape == (1, 4, 100)

def test_model_with_loss():
    config = SynapseModelConfig.small_test()
    model = SynapseModel(config)
    input_ids = torch.tensor([[1, 5, 10, 50]])
    labels = torch.tensor([[5, 10, 50, 20]])
    result = model(input_ids, labels=labels)
    assert "loss" in result
    assert result["loss"].item() > 0

def test_model_backward():
    """Test that gradients flow through the entire model."""
    config = SynapseModelConfig.small_test()
    model = SynapseModel(config)
    input_ids = torch.tensor([[1, 5, 10, 50]])
    labels = torch.tensor([[5, 10, 50, 20]])
    result = model(input_ids, labels=labels, training=True)
    result["loss"].backward()
    # Check that at least some parameters have gradients
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad, "No gradients flowed through the model"

def test_model_reset():
    config = SynapseModelConfig.small_test()
    model = SynapseModel(config)
    input_ids = torch.tensor([[1, 5, 10]])
    _ = model(input_ids)
    model.reset_state()  # Should not panic


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
