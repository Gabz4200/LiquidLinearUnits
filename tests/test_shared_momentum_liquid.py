import torch
from llu.models import SharedMomentumLiquidLN

def _get_device(model):
    return model.linear_core.weight.device

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
