import torch
import pytest
from llu.models.model import (
    LiquidLinear,
    Rank1LiquidLN,
    RankRLiquidLN,
    StableLiquidLN,
    SharedMomentumLiquidLN,
    BatchMomentumLiquidLN,
)

# Helper function to get device
def _get_device(model):
    return model.linear_core.weight.device

# ----------------- LiquidLinear Tests -----------------

def test_liquid_linear_behavior():
    torch.manual_seed(42)
    in_features = 8
    out_features = 4
    
    # 1. Init Equivalence
    model = LiquidLinear(in_features, out_features, bias=True, dynamic_bias=True)
    device = _get_device(model)
    x = torch.randn(2, 3, in_features, device=device)
    
    out = model(x)
    core_out = model.linear_core(x)
    assert torch.allclose(out, core_out, atol=1e-6)
    
    # 2. Shapes
    x_1d = torch.randn(in_features, device=device)
    assert model(x_1d).shape == (out_features,)
    x_3d = torch.randn(2, 3, in_features, device=device)
    assert model(x_3d).shape == (2, 3, out_features)
    
    # 3. Freezing
    # Reset grad flags
    for p in model.parameters():
        p.requires_grad = True
    model.freeze_core()
    assert not model.linear_core.weight.requires_grad
    if model.linear_core.bias is not None:
        assert not model.linear_core.bias.requires_grad
    assert model.hypernetwork.weight.requires_grad
    assert model.scale.requires_grad
    
    for p in model.parameters():
        p.requires_grad = True
    model.freeze_hypernetwork()
    assert model.linear_core.weight.requires_grad
    assert not model.hypernetwork.weight.requires_grad
    assert not model.scale.requires_grad

    # 4. Training Step / Grad flow
    model = LiquidLinear(in_features, out_features, bias=True, dynamic_bias=True)
    x = torch.randn(2, in_features, device=device)
    y = torch.randn(2, out_features, device=device)
    loss = (model(x) - y).pow(2).sum()
    loss.backward()
    
    # Check gradients exist
    assert model.linear_core.weight.grad is not None
    assert model.hypernetwork.weight.grad is not None
    assert model.scale.grad is not None
    if model.bias_dynamic is not None:
        assert model.bias_dynamic.weight.grad is not None

# ----------------- Rank1LiquidLN Tests -----------------

def test_rank1_liquid_behavior():
    torch.manual_seed(42)
    in_features = 8
    out_features = 4
    
    model = Rank1LiquidLN(in_features, out_features, bias=True, dynamic_bias=True, normalize_input=True)
    device = _get_device(model)
    x = torch.randn(2, 3, in_features, device=device)
    
    # 1. Init Equivalence
    out = model(x)
    core_out = model.linear_core(x)
    assert torch.allclose(out, core_out, atol=1e-6)
    
    # 2. Shapes
    assert model(x).shape == (2, 3, out_features)
    
    # 3. Freezing
    for p in model.parameters():
        p.requires_grad = True
    model.freeze_core()
    assert not model.linear_core.weight.requires_grad
    assert model.hypernetwork.weight.requires_grad
    
    for p in model.parameters():
        p.requires_grad = True
    model.freeze_hypernetwork()
    assert model.linear_core.weight.requires_grad
    assert not model.hypernetwork.weight.requires_grad

# ----------------- RankRLiquidLN Tests -----------------

def test_rankr_liquid_behavior():
    torch.manual_seed(42)
    in_features = 8
    out_features = 4
    rank = 3
    
    # With non-linear hypernet
    model = RankRLiquidLN(
        in_features, out_features, rank=rank, bias=True,
        dynamic_bias=True, nonlinear_hypernet=True
    )
    device = _get_device(model)
    x = torch.randn(2, 3, in_features, device=device)
    
    # 1. Init Equivalence
    out = model(x)
    core_out = model.linear_core(x)
    assert torch.allclose(out, core_out, atol=1e-6)
    
    # 2. Shapes
    assert model(x).shape == (2, 3, out_features)
    
    # 3. Freezing
    for p in model.parameters():
        p.requires_grad = True
    model.freeze_core()
    assert not model.linear_core.weight.requires_grad
    # The hypernetwork is a Sequential container, check its first layer's weight
    assert next(model.hypernetwork.parameters()).requires_grad
    
    for p in model.parameters():
        p.requires_grad = True
    model.freeze_hypernetwork()
    assert model.linear_core.weight.requires_grad
    assert not next(model.hypernetwork.parameters()).requires_grad

# ----------------- StableLiquidLN Tests -----------------

def test_stable_liquid_behavior():
    torch.manual_seed(42)
    in_features = 8
    out_features = 4
    
    model = StableLiquidLN(in_features, out_features, rank=2, bias=True, dynamic_bias=True)
    device = _get_device(model)
    x = torch.randn(2, 3, in_features, device=device)
    
    # 1. Init Equivalence
    out = model(x)
    core_out = model.linear_core(x)
    assert torch.allclose(out, core_out, atol=1e-6)
    
    # 2. Shapes
    assert model(x).shape == (2, 3, out_features)
    
    # 3. Conditioning
    cond = torch.randn(2, 3, in_features, device=device)
    out_cond = model(x, cond=cond)
    assert out_cond.shape == (2, 3, out_features)

# ----------------- SharedMomentumLiquidLN Tests -----------------

def test_shared_momentum_behavior():
    torch.manual_seed(42)
    in_features = 8
    out_features = 4
    rank = 2
    decay_rate = 0.5
    
    model = SharedMomentumLiquidLN(
        in_features, out_features, decay_rate=decay_rate, rank=rank, bias=True
    )
    device = _get_device(model)
    
    # 1. Ensure buffers start at zero before any forward pass
    assert torch.all(model.a_raw == 0)
    assert torch.all(model.b_raw == 0)
    
    # 2. Init Equivalence at step 1
    x = torch.randn(2, 3, in_features, device=device)
    out = model(x)
    core_out = model.linear_core(x)
    assert torch.allclose(out, core_out, atol=1e-6)
    
    # 3. Momentum buffer shape is (rank, feat) - no batch dimension!
    assert model.a_raw.shape == (rank, out_features)
    assert model.b_raw.shape == (rank, in_features)
    
    # 4. Under no-training, b_raw remains zero, but a_raw is non-zero
    assert not torch.all(model.a_raw == 0)
    assert torch.all(model.b_raw == 0)
    
    a_raw_1 = model.a_raw.clone()
    
    # Run another forward pass to verify a_raw accumulates momentum
    _ = model(x)
    a_raw_2 = model.a_raw.clone()
    assert not torch.allclose(a_raw_1, a_raw_2)
    assert torch.all(model.b_raw == 0)
    
    # 5. Optimize to update hypernetwork and check that b_raw updates too
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    y = torch.randn_like(out)
    loss = (model(x) - y).pow(2).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # Forward pass after optimization
    _ = model(x)
    assert not torch.all(model.b_raw == 0)

# ----------------- BatchMomentumLiquidLN Tests -----------------

def test_batch_momentum_behavior():
    torch.manual_seed(42)
    in_features = 8
    out_features = 4
    rank = 2
    decay_rate = 0.5
    
    model = BatchMomentumLiquidLN(
        in_features, out_features, decay_rate=decay_rate, rank=rank, bias=True
    )
    device = _get_device(model)
    
    # 1. Ensure buffers start at zero (placeholder shape is (1, rank, feat))
    assert torch.all(model.a_raw == 0)
    assert torch.all(model.b_raw == 0)
    
    # 2. Init Equivalence
    x1 = torch.randn(2, 3, in_features, device=device)
    out1 = model(x1)
    core_out1 = model.linear_core(x1)
    assert torch.allclose(out1, core_out1, atol=1e-6)
    
    # 3. Momentum buffer has batch dimensions matching input: (B, T, rank, feat)
    assert model.a_raw.shape == (2, 3, rank, out_features)
    assert model.b_raw.shape == (2, 3, rank, in_features)
    
    # 4. Under no-training, b_raw remains zero, but a_raw is non-zero
    assert not torch.all(model.a_raw == 0)
    assert torch.all(model.b_raw == 0)
    
    a_raw_1 = model.a_raw.clone()
    _ = model(x1)
    a_raw_2 = model.a_raw.clone()
    assert not torch.allclose(a_raw_1, a_raw_2)
    assert torch.all(model.b_raw == 0)
    
    # 5. Optimize to update hypernetwork and check that b_raw updates too
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    y = torch.randn_like(out1)
    loss = (model(x1) - y).pow(2).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # Forward pass after optimization
    _ = model(x1)
    assert not torch.all(model.b_raw == 0)
    
    # 6. Dynamic resizing: change batch dimension and ensure no errors + correct shape adjustment
    x2 = torch.randn(4, 5, in_features, device=device)
    out2 = model(x2)
    assert out2.shape == (4, 5, out_features)
    assert model.a_raw.shape == (4, 5, rank, out_features)
    assert model.b_raw.shape == (4, 5, rank, in_features)

# ----------------- Init Methods Test -----------------

@pytest.mark.parametrize("init_method", ["hyperfan_in", "hyperfan_out", "xavier", "small"])
def test_all_initialization_methods(init_method):
    in_features = 8
    out_features = 4
    # Ensure all initialization schemes run without error
    for model_cls in [
        LiquidLinear,
        Rank1LiquidLN,
        RankRLiquidLN,
        StableLiquidLN,
        SharedMomentumLiquidLN,
        BatchMomentumLiquidLN,
    ]:
        model = model_cls(in_features, out_features, init_method=init_method)
        # Smoke test forward pass
        x = torch.randn(2, in_features, device=_get_device(model))
        out = model(x)
        assert out.shape == (2, out_features)

# ----------------- Activation Tests -----------------

@pytest.mark.parametrize("factor_activation", ["tanh", "norm", "rmsnorm", "none"])
def test_all_activations(factor_activation):
    in_features = 8
    out_features = 4
    for model_cls in [
        LiquidLinear,
        Rank1LiquidLN,
        RankRLiquidLN,
        StableLiquidLN,
        SharedMomentumLiquidLN,
        BatchMomentumLiquidLN,
    ]:
        model = model_cls(in_features, out_features, factor_activation=factor_activation)
        # Smoke test forward pass
        x = torch.randn(2, in_features, device=_get_device(model))
        out = model(x)
        assert out.shape == (2, out_features)

def test_invalid_activation():
    in_features = 8
    out_features = 4
    for model_cls in [
        LiquidLinear,
        Rank1LiquidLN,
        RankRLiquidLN,
        StableLiquidLN,
        SharedMomentumLiquidLN,
        BatchMomentumLiquidLN,
    ]:
        model = model_cls(in_features, out_features, factor_activation="invalid_act")
        x = torch.randn(2, in_features, device=_get_device(model))
        with pytest.raises(ValueError, match="Unknown factor_activation"):
            _ = model(x)

# ----------------- Advanced PyTorch compatibility tests -----------------

def test_gradcheck():
    """Verify that analytical gradients match finite differences via autograd.gradcheck."""
    torch.manual_seed(42)
    in_features = 4
    out_features = 2
    for model_cls in [
        LiquidLinear,
        Rank1LiquidLN,
        RankRLiquidLN,
        StableLiquidLN,
        SharedMomentumLiquidLN,
        BatchMomentumLiquidLN,
    ]:
        # Using double precision (float64) is required for gradcheck stability
        model = model_cls(in_features, out_features, bias=True, dynamic_bias=True).double()
        x = torch.randn(2, in_features, dtype=torch.float64, device=_get_device(model), requires_grad=True)
        # check_undefined_grad=False is needed for momentum layers due to stateful buffer updates
        res = torch.autograd.gradcheck(model, (x,), eps=1e-6, atol=1e-4, check_undefined_grad=False)
        assert res is True

def test_dtype_and_compile_compatibility():
    """Verify half-precision casting (bfloat16) and torch.compile compatibility."""
    torch.manual_seed(42)
    in_features = 8
    out_features = 4
    for model_cls in [
        LiquidLinear,
        Rank1LiquidLN,
        RankRLiquidLN,
        StableLiquidLN,
        SharedMomentumLiquidLN,
        BatchMomentumLiquidLN,
    ]:
        model = model_cls(in_features, out_features)
        device = _get_device(model)
        
        # 1. bfloat16 casting & forward/backward verification
        model_bf16 = model_cls(in_features, out_features).to(dtype=torch.bfloat16)
        x_bf16 = torch.randn(2, in_features, dtype=torch.bfloat16, device=device)
        out_bf16 = model_bf16(x_bf16)
        assert out_bf16.dtype == torch.bfloat16
        assert out_bf16.shape == (2, out_features)
        
        loss_bf16 = out_bf16.pow(2).sum()
        loss_bf16.backward()
        
        # Verify gradient dtype is bfloat16
        assert model_bf16.linear_core.weight.grad is not None
        assert model_bf16.linear_core.weight.grad.dtype == torch.bfloat16
        
        # 2. torch.compile compatibility
        compiled_model = torch.compile(model)
        x_compile = torch.randn(2, in_features, device=device)
        out_compile = compiled_model(x_compile)
        assert out_compile.shape == (2, out_features)


