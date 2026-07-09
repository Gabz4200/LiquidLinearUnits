import torch
from llu.models import BatchMomentumLiquidLN

def _get_device(model):
    return model.linear_core.weight.device

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
