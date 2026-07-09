import torch
from llu.models import RankRLiquidLN

def _get_device(model):
    return model.linear_core.weight.device

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
