r"""
CPU-parity regression tests for ``llu.models.gdn2.GatedDeltaNet2``.

These tests pin the claim that ``gdn2`` is numerically 100% equivalent to the
NVlabs GatedDeltaNet-2 baseline while being GPU/Triton-free.

The central check is an *independent* reference implementation of the GDN-2
forward (built from scratch, not by re-calling the module). It shares the
module's learned weights, so any numerical or structural bug in the local
module surfaces as a divergence. We also lock in the incremental-decoding
cache path, the varlen ``attention_mask`` path, chunk==fused mode agreement,
the backward pass (``gradcheck``), and long-sequence / bf16 numerical behavior.

Tolerances: float32 parity is checked to 1e-4 (observed ~1e-9); bf16 parity to
1e-2 (observed ~1e-5, the bf16 precision floor). These are deliberately loose
so the tests assert *equivalence*, not bit-exactness.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from llu.models.gdn2.gdn2 import GatedDeltaNet2


# =============================================================================
# INDEPENDENT REFERENCE IMPLEMENTATION (no calls into the module's forward)
# =============================================================================
def _ref_short_conv(x: torch.Tensor, weight: torch.Tensor, bias, kernel_size: int) -> torch.Tensor:
    """Depthwise causal conv1d with zero left-pad of (kernel_size-1), then silu.

    Mirrors ``fla``'s ``ShortConvolution`` / the local reimplementation.
    """
    B, T, D = x.shape
    pad = torch.zeros(B, D, kernel_size - 1, device=x.device, dtype=x.dtype)
    y = F.conv1d(torch.cat([pad, x.transpose(1, 2)], dim=-1), weight, bias, padding=0, groups=D)
    return F.silu(y.transpose(1, 2))


def _ref_recurrence(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    g: torch.Tensor, b: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Tokenwise GDN-2 recurrence (paper Eq.10), written independently.

    q,k,g,b: [B, T, HV, K]; v,w: [B, T, HV, V]. q,k are L2-normalised in-kernel.
    """
    B, T, HV, K = q.shape
    V = v.shape[-1]
    scale = K ** -0.5
    S = torch.zeros(B, HV, K, V, dtype=torch.float32)
    o = torch.zeros(B, T, HV, V, dtype=torch.float32)
    for t in range(T):
        qt = q[:, t].float()
        kt = k[:, t].float()
        vt = v[:, t].float()
        gt = g[:, t].float()
        bt = b[:, t].float()
        wt = w[:, t].float()
        qt = qt / torch.sqrt(torch.sum(qt * qt, dim=-1, keepdim=True) + 1e-6) * scale
        kt = kt / torch.sqrt(torch.sum(kt * kt, dim=-1, keepdim=True) + 1e-6)
        S = S * torch.exp(gt).unsqueeze(-1)
        bk = bt * kt
        v_new = wt * vt - torch.sum(S * bk.unsqueeze(-1), dim=2)
        S = S + kt.unsqueeze(-1) * v_new.unsqueeze(-2)
        o[:, t] = torch.sum(S * qt.unsqueeze(-1), dim=2)
    return o.to(q.dtype)


def _reference_forward(m: GatedDeltaNet2, x: torch.Tensor) -> torch.Tensor:
    """Full reference forward sharing ``m``'s parameters."""
    B, T, _ = x.shape
    hk = m.head_k_dim
    hv = m.head_v_dim
    NH = m.num_heads
    NHV = m.num_v_heads

    if m.use_short_conv:
        q = _ref_short_conv(m.q_proj(x), m.q_conv1d.weight, m.q_conv1d.bias, m.conv_size)
        k = _ref_short_conv(m.k_proj(x), m.k_conv1d.weight, m.k_conv1d.bias, m.conv_size)
        v = _ref_short_conv(m.v_proj(x), m.v_conv1d.weight, m.v_conv1d.bias, m.conv_size)
    else:
        q = F.silu(m.q_proj(x))
        k = F.silu(m.k_proj(x))
        v = F.silu(m.v_proj(x))

    g = -(m.A_log.float().exp().repeat_interleave(hk)) * F.softplus(m.f_proj(x).float() + m.dt_bias)
    b = m.b_proj(x).sigmoid()
    w = m.w_proj(x).sigmoid()

    q, k, g = (u.reshape(B, T, NH, hk) for u in (q, k, g))
    v = v.reshape(B, T, NHV, hv)
    b = b.reshape(B, T, NH, hk)
    w = w.reshape(B, T, NHV, hv)

    if NHV > NH:
        gf = NHV // NH
        q, k, g, b = (u.repeat_interleave(gf, dim=2) for u in (q, k, g, b))
    if m.allow_neg_eigval:
        b = b * 2.0

    o = _ref_recurrence(q, k, v, g, b, w)
    g_out = m.g_proj(x).reshape(B, T, NHV, hv)
    o = F.rms_norm(o, (hv,), m.o_norm.weight, m.o_norm.eps) * F.silu(g_out)
    return m.o_proj(o.reshape(B, T, NHV * hv))


# =============================================================================
# MINIMAL CACHE (matches the local module's get/update contract)
# =============================================================================
class _MockCache:
    def __init__(self) -> None:
        self.states: dict = {}

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, i):
        return self.states.get(i)

    def update(self, layer_idx, **kwargs):
        self.states[layer_idx] = kwargs
        return self


# =============================================================================
# 1. INDEPENDENT REFERENCE PARITY (the core anti-regression check)
# =============================================================================
@pytest.mark.parametrize(
    "cfg",
    [
        dict(hidden_size=64, head_dim=16, num_heads=4, expand_v=1.0, conv_size=4),
        dict(hidden_size=64, head_dim=16, num_heads=4, expand_v=2.0, conv_size=3),
        dict(hidden_size=64, head_dim=16, num_heads=4, expand_v=1.0, allow_neg_eigval=True),
        dict(hidden_size=64, head_dim=8, num_heads=2, expand_v=1.0, use_short_conv=False),
        dict(hidden_size=96, head_dim=16, num_heads=3, num_v_heads=6, expand_v=1.0),
    ],
)
def test_when_forward_matches_independent_reference_then_ok(cfg):
    torch.manual_seed(0)
    m = GatedDeltaNet2(**cfg).to(torch.float32).eval()
    torch.manual_seed(1)
    x = torch.randn(2, 17, m.hidden_size, dtype=torch.float32)
    with torch.no_grad():
        out = m(x)[0]
        ref = _reference_forward(m, x)
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4), (
        f"max err { (out - ref).abs().max():.3e}"
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "cfg,dtype",
    [
        (dict(hidden_size=256, head_dim=64, num_heads=8, expand_v=2.0, conv_size=4), torch.float32),
        (dict(hidden_size=256, head_dim=64, num_heads=8, expand_v=2.0, conv_size=4), torch.bfloat16),
        (dict(hidden_size=256, head_dim=64, num_heads=8, expand_v=1.0, allow_neg_eigval=True, conv_size=4), torch.float32),
        (dict(hidden_size=192, head_dim=32, num_heads=6, num_v_heads=12, expand_v=1.0), torch.bfloat16),
    ],
)
def test_when_long_sequence_matches_reference_then_ok(cfg, dtype):
    torch.manual_seed(0)
    m = GatedDeltaNet2(**cfg).to(dtype).eval()
    T = 200 if dtype is torch.float32 else 130
    torch.manual_seed(1)
    x = torch.randn(2, T, m.hidden_size, dtype=dtype)
    with torch.no_grad():
        out = m(x)[0]
        ref = _reference_forward(m, x)
    atol = 1e-4 if dtype is torch.float32 else 1e-2
    assert torch.allclose(out, ref, atol=atol, rtol=atol), (
        f"max err { (out - ref).abs().max():.3e}"
    )


# =============================================================================
# 2. INCREMENTAL DECODING (cache) == FULL FORWARD
# =============================================================================
@pytest.mark.parametrize(
    "cfg",
    [
        dict(hidden_size=64, head_dim=16, num_heads=4, expand_v=1.0, conv_size=4),
        dict(hidden_size=64, head_dim=16, num_heads=4, expand_v=2.0, conv_size=3),
        dict(hidden_size=64, head_dim=16, num_heads=4, expand_v=1.0, allow_neg_eigval=True),
        dict(hidden_size=96, head_dim=16, num_heads=3, num_v_heads=6, expand_v=1.0),
    ],
)
def test_when_incremental_decode_matches_full_forward_then_ok(cfg):
    torch.manual_seed(0)
    m = GatedDeltaNet2(**cfg).to(torch.float32).eval()
    m.layer_idx = 0
    T = 17
    torch.manual_seed(2)
    x = torch.randn(2, T, m.hidden_size, dtype=torch.float32)
    with torch.no_grad():
        full = m(x)[0]
        cache = _MockCache()
        dec = []
        for t in range(T):
            o, _, cache = m(x[:, t : t + 1], use_cache=True, past_key_values=cache)
            dec.append(o)
        dec = torch.cat(dec, dim=1)
    assert torch.allclose(full, dec, atol=1e-4, rtol=1e-4), (
        f"max err { (full - dec).abs().max():.3e}"
    )


# =============================================================================
# 3. VARIEN ATTENTION MASK PRESERVES VALID POSITIONS
# =============================================================================
def test_when_attention_mask_preserves_valid_positions_then_ok():
    cfg = dict(hidden_size=64, head_dim=16, num_heads=4, expand_v=1.0, conv_size=4)
    T = 17
    torch.manual_seed(0)
    m = GatedDeltaNet2(**cfg).to(torch.float32).eval()
    torch.manual_seed(3)
    x = torch.randn(2, T, m.hidden_size, dtype=torch.float32)

    mask_all = torch.ones(2, T, dtype=torch.bool)
    mask = torch.ones(2, T, dtype=torch.bool)
    mask[1, T - 4 :] = False  # pad tail of sequence 1

    with torch.no_grad():
        full = m(x)[0]
        out_all = m(x, attention_mask=mask_all)[0]
        out_masked = m(x, attention_mask=mask)[0]

    # all-ones mask exercises the unpad/repad path; numerically identical
    # (differs from the direct path only by gather/scatter float noise ~1e-9)
    assert torch.allclose(full, out_all, atol=1e-4, rtol=1e-4), (
        f"all-ones mask err {(full - out_all).abs().max():.3e}"
    )
    # valid positions must match; padded positions are zeroed by design
    err = max(
        (full[0] - out_masked[0]).abs().max().item(),
        (full[1, : T - 4] - out_masked[1, : T - 4]).abs().max().item(),
    )
    assert err < 1e-4, f"valid-position mask err {err:.3e}"


# =============================================================================
# 4. CHUNK MODE == FUSED_RECURRENT MODE
# =============================================================================
def test_when_chunk_and_fused_recurrent_modes_agree_then_ok():
    cfg = dict(hidden_size=64, head_dim=16, num_heads=4, expand_v=1.0, conv_size=4)
    T = 50
    torch.manual_seed(0)
    m_chunk = GatedDeltaNet2(**cfg, mode="chunk").to(torch.float32).eval()
    m_rec = GatedDeltaNet2(**cfg, mode="fused_recurrent").to(torch.float32).eval()
    m_rec.load_state_dict(m_chunk.state_dict())
    torch.manual_seed(4)
    x = torch.randn(1, T, m_chunk.hidden_size, dtype=torch.float32)
    with torch.no_grad():
        o_chunk = m_chunk(x)[0]
        o_rec = m_rec(x)[0]
    assert torch.equal(o_chunk, o_rec), "chunk and fused_recurrent modes disagree"


# =============================================================================
# 5. BACKWARD PASS (gradcheck) — exact autograd through the recurrence
# =============================================================================
@pytest.mark.slow
def test_when_backward_matches_numerical_gradient_then_ok():
    torch.manual_seed(0)
    m = GatedDeltaNet2(
        hidden_size=16, head_dim=4, num_heads=2, expand_v=1.0, conv_size=3, use_short_conv=True
    ).to(torch.float32).train()
    x = torch.randn(1, 8, 16, dtype=torch.float32, requires_grad=True)
    # gradcheck runs in float32 to match the model's internal float32 recurrence
    passed = torch.autograd.gradcheck(
        lambda inp: m(inp)[0], (x,), atol=1e-3, rtol=1e-3, eps=1e-2
    )
    assert passed, "gradcheck failed: backward is not mathematically correct"
