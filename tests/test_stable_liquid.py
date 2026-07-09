import torch
from llu.models import StableLiquidLN

def _get_device(model):
    return model.linear_core.weight.device

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
