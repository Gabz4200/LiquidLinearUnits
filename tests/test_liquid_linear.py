import torch
from llu.models import LiquidLinear

def _get_device(model):
    return model.linear_core.weight.device

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
