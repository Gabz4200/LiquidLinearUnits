r"""LiquidTransformer: a classic Transformer pipeline built entirely from LLU layers.

Every ``nn.Linear`` in a standard Transformer block is replaced by a Liquid
Linear Unit (LLU) from :mod:`llu.models.llns`:

* Token mixing is **sliding-window causal attention** (local, not full
  attention) -- the query/key/value/output projections are LLU layers.
* The feed-forward sublayer is a (SwiGLU) MLP whose linear transforms are LLU
  layers.

Design notes / deliberate choices (see project discussion):

* **GDN-2 is used only in FFN positions.** It was designed as an FFN-style
  sequence mixer and is a poor fit for the query/key/value *projection*
  matrices (which should map a single token deterministically). For the GDN
  families the Q/K/V/O projections use the non-GDN counterpart of the same
  family (``StableLiquidLN`` for ``GDNLiquidLN``, ``SharedMomentumLiquidLN``
  for ``MomentumGDNLiquidLN``); the FFN uses the GDN-2 unit.
* The model is run **token-by-token (autoregressive)** so that the recurrent
  state of every LLU variant evolves correctly over time: GDN-2 threads its
  ``past_key_values`` cache, and the momentum units update their buffers step
  by step. A rolling KV cache supplies the sliding-window attention context.
* The readout head is a per-position ``StableLiquidLN`` (neutral, no cross-step
  recurrence) so it does not bias the architecture comparison.
"""

from __future__ import annotations

import inspect
import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .llns import (
    LiquidLinear,
    Rank1LiquidLN,
    RankRLiquidLN,
    StableLiquidLN,
    SharedMomentumLiquidLN,
    BatchMomentumLiquidLN,
    GDNLiquidLN,
    MomentumGDNLiquidLN,
    CrossAttnLoraLN,
)


# ---------------------------------------------------------------------------
# LLU registry / helpers
# ---------------------------------------------------------------------------

ARCH_FACTORIES = {
    "LiquidLinear": LiquidLinear,
    "Rank1LiquidLN": Rank1LiquidLN,
    "RankRLiquidLN": RankRLiquidLN,
    "StableLiquidLN": StableLiquidLN,
    "SharedMomentumLiquidLN": SharedMomentumLiquidLN,
    "BatchMomentumLiquidLN": BatchMomentumLiquidLN,
    "GDNLiquidLN": GDNLiquidLN,
    "MomentumGDNLiquidLN": MomentumGDNLiquidLN,
    "StableGDNCondLiquidLN": StableLiquidLN,
    "CrossAttnLoraLN": CrossAttnLoraLN,
}

# GDN-2 units: suitable for FFN positions, not for projection matrices.
_GDN_LAYERS = (GDNLiquidLN, MomentumGDNLiquidLN)
# Units whose forward accepts an explicit `cond` argument.
_COND_LAYERS = (
    StableLiquidLN,
    SharedMomentumLiquidLN,
    BatchMomentumLiquidLN,
    GDNLiquidLN,
    MomentumGDNLiquidLN,
    CrossAttnLoraLN,
)

# Architectures that carry cross-step recurrence (delta-rule memory or momentum
# buffers), so they can "see the whole sequence" without attention. These are
# the ones that admit a meaningful attention-free ablation.
RECURRENT_ARCHS = {
    "GDNLiquidLN",
    "MomentumGDNLiquidLN",
    "SharedMomentumLiquidLN",
    "BatchMomentumLiquidLN",
    "StableGDNCondLiquidLN",
}


def is_valid_arch(arch: str) -> bool:
    """True for any registered arch or its ``_noattn`` ablation variant."""
    if arch in ARCH_FACTORIES:
        return True
    if arch.endswith("_noattn"):
        return arch[: -len("_noattn")] in RECURRENT_ARCHS
    return False


def _arch_map(arch: str) -> tuple[str, str]:
    """Return ``(projection_llu, ffn_llu)`` names for an architecture.

    GDN-2 families use a non-GDN counterpart for the Q/K/V/O projections and
    keep GDN-2 for the FFN.
    """
    if arch == "GDNLiquidLN":
        return ("StableLiquidLN", "GDNLiquidLN")
    if arch == "MomentumGDNLiquidLN":
        return ("SharedMomentumLiquidLN", "MomentumGDNLiquidLN")
    if arch == "StableGDNCondLiquidLN":
        return ("StableLiquidLN", "StableLiquidLN")
    return (arch, arch)


def _llu_kwargs_for(
    cls: type,
    rank: int = 4,
    decay_rate: float = 0.4,
    head_dim: int = 16,
    num_heads: int = 4,
    learnable_decay: bool = False,
    layer_idx: int = 0,
    parameterization: str = "lora",
    attn_dim: int = 32,
    attn_heads: int = 2,
) -> dict:
    """Build a kwargs dict containing only the params ``cls.__init__`` accepts.

    Introspecting the constructor signature keeps the factory robust to the
    differing parameter sets of the eight LLU variants. ``attn_dim`` /
    ``attn_heads`` apply only to cross-attention-based units (e.g.
    ``CrossAttnLoraLN``); they are silently skipped for others.
    """
    params = inspect.signature(cls.__init__).parameters
    kw: dict = {}
    if "rank" in params:
        kw["rank"] = rank
    if "decay_rate" in params:
        kw["decay_rate"] = decay_rate
    if "initial_decay_rate" in params:
        kw["initial_decay_rate"] = decay_rate
    if "learnable_decay_rate" in params:
        kw["learnable_decay_rate"] = learnable_decay
    if "head_dim" in params:
        kw["head_dim"] = head_dim
    if "num_heads" in params:
        kw["num_heads"] = num_heads
    if "layer_idx" in params:
        kw["layer_idx"] = layer_idx
    if "normalize_input" in params:
        kw["normalize_input"] = False
    if "dynamic_bias" in params:
        kw["dynamic_bias"] = False
    if "parameterization" in params:
        kw["parameterization"] = parameterization
    if "attn_dim" in params:
        kw["attn_dim"] = attn_dim
    if "attn_heads" in params:
        kw["attn_heads"] = attn_heads
    return kw


class _GDNCache:
    """Minimal cache container compatible with GDN-2's ``get/update_layer_cache``.

    GDN-2 indexes the cache by ``layer_idx``; each GDN sublayer owns its own
    cache, so a size-1 cache indexed at 0 is sufficient.
    """

    def __init__(self, layer_idx: int = 0) -> None:
        self.layer_idx = layer_idx
        self.states: list[Optional[dict]] = [None] * (layer_idx + 1)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, i: int) -> Optional[dict]:
        return self.states[i]

    def update(self, layer_idx: int, **kwargs: Any) -> "_GDNCache":
        self.states[layer_idx] = kwargs
        return self


def _llu_step(llu: nn.Module, x: torch.Tensor, cache: Any, cond: Optional[torch.Tensor] = None):
    """Run one LLU sublayer on a single token ``x`` (``(B, D)``), threading state.

    * GDN-2 units receive the token as a length-1 sequence and thread their
      recurrent cache so the GDN-2 state evolves across tokens.
    * ``cond``-accepting units get ``cond`` if supplied, else ``cond = x``
      (sanctioned: the conditioning tensor may equal ``x`` or be a linear
      projection of it; we use ``x`` by default).
    * Plain units just transform ``x``.
    """
    if isinstance(llu, _GDN_LAYERS):
        out, cache = llu(x.unsqueeze(1), use_cache=True, past_key_values=cache)
        return out.squeeze(1), cache
    if isinstance(llu, _COND_LAYERS):
        return llu(x, cond=cond if cond is not None else x), cache
    return llu(x), cache


def _reset_llu(llu: nn.Module) -> None:
    """Reset momentum buffers between sequences.

    A fresh, grad-free tensor is created (not an in-place zero of the previous
    tensor) so a freed autograd graph from the prior forward is not reattached
    to the next one -- otherwise training loops raise "backward through the
    graph a second time".
    """
    if hasattr(llu, "a_raw") and llu.a_raw is not None:
        llu.a_raw = torch.zeros_like(llu.a_raw)
    if hasattr(llu, "b_raw") and llu.b_raw is not None:
        llu.b_raw = torch.zeros_like(llu.b_raw)
    if hasattr(llu, "g_raw") and llu.g_raw is not None:
        llu.g_raw = torch.zeros_like(llu.g_raw)


class RMSNorm(nn.Module):
    """Learnable root-mean-square normalization (matches the LLU family)."""

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


def _valid_n_heads(d_model: int, requested: int) -> int:
    """Largest ``n_heads <= requested`` that divides ``d_model`` (>= 1)."""
    n = min(max(1, requested), d_model)
    while n > 1 and d_model % n != 0:
        n -= 1
    return n


def _split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    # (B, D) -> (B, H, d_head)
    return x.view(x.shape[0], n_heads, x.shape[1] // n_heads)


def _merge_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    # (B, H, d_head) -> (B, D)
    return x.reshape(x.shape[0], n_heads * x.shape[2])


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class LiquidTransformerBlock(nn.Module):
    """One pre-norm Transformer block: sliding-window attention + LLU FFN.

    All projections (Q, K, V, O) and the FFN transforms are LLU layers. GDN-2
    units appear only in the FFN (``ffn_llu``); the projections use
    ``proj_llu`` (a non-GDN counterpart for the GDN families).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        proj_cls: type,
        ffn_cls: type,
        window: int,
        use_swiglu: bool = True,
        swiglu_mult: int = 4,
        use_attention: bool = True,
        proj_kwargs: Optional[dict] = None,
        ffn_kwargs: Optional[dict] = None,
        cond_llu: Optional[type] = None,
        cond_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window = window
        self.use_swiglu = use_swiglu
        self.use_attention = use_attention

        proj_kwargs = dict(proj_kwargs or {})
        ffn_kwargs = dict(ffn_kwargs or {})

        self._attn_names: list[str] = []
        if use_attention:
            self.q_proj = proj_cls(d_model, d_model, **proj_kwargs)
            self.k_proj = proj_cls(d_model, d_model, **proj_kwargs)
            self.v_proj = proj_cls(d_model, d_model, **proj_kwargs)
            self.o_proj = proj_cls(d_model, d_model, **proj_kwargs)
            self.norm1 = RMSNorm(d_model)
            self._attn_names = ["q_proj", "k_proj", "v_proj", "o_proj"]

        if use_swiglu:
            self.ffn_gate = ffn_cls(d_model, d_model * swiglu_mult, **ffn_kwargs)
            self.ffn_up = ffn_cls(d_model, d_model * swiglu_mult, **ffn_kwargs)
            self.ffn_down = ffn_cls(d_model * swiglu_mult, d_model, **ffn_kwargs)
            self.ffn_names = ("ffn_gate", "ffn_up", "ffn_down")
        else:
            self.ffn = ffn_cls(d_model, d_model, **ffn_kwargs)
            self.ffn_names = ("ffn",)
        self.norm2 = RMSNorm(d_model)

        # Optional GDN-2 cond provider: its output conditions the FFN
        # (StableLiquidLN). Provides the `cond` half of (SWA, GDN-2) -> (x, cond).
        self.cond_provider = None
        if cond_llu is not None:
            self.cond_provider = cond_llu(d_model, d_model, **(cond_kwargs or {}))

        # Per-sublayer recurrent caches (GDN-2 only); rolling KV cache for SWA.
        self.caches: dict[str, Any] = {}
        self._sublayer_names = [*self._attn_names, *self.ffn_names]
        if self.cond_provider is not None:
            self._sublayer_names = [*self._sublayer_names, "cond_provider"]
        self._gdn_sublayers = [
            n for n in self._sublayer_names
            if isinstance(getattr(self, n), _GDN_LAYERS)
        ]
        self.kv_cache: list[tuple[torch.Tensor, torch.Tensor]] = []

    def reset(self) -> None:
        self.kv_cache = []
        for name in self._sublayer_names:
            sub = getattr(self, name)
            _reset_llu(sub)
            self.caches[name] = _GDNCache(0) if isinstance(sub, _GDN_LAYERS) else None

    def _attention(self, h: torch.Tensor) -> torch.Tensor:
        h_norm = self.norm1(h)
        q = _split_heads(self.q_proj(h_norm), self.n_heads)        # (B, H, dh)
        k = _split_heads(self.k_proj(h_norm), self.n_heads)
        v = _split_heads(self.v_proj(h_norm), self.n_heads)

        self.kv_cache.append((k, v))
        if len(self.kv_cache) > self.window:
            self.kv_cache.pop(0)

        K = torch.stack([c[0] for c in self.kv_cache], dim=2)      # (B, H, L, dh)
        V = torch.stack([c[1] for c in self.kv_cache], dim=2)
        q_ = q.unsqueeze(2)                                        # (B, H, 1, dh)
        scores = (q_ @ K.transpose(-1, -2)) / math.sqrt(self.d_head)
        attn = scores.softmax(dim=-1)
        ctx = (attn @ V).squeeze(2)                                # (B, H, dh)
        ctx = _merge_heads(ctx, self.n_heads)                     # (B, D)
        out, self.caches["o_proj"] = _llu_step(self.o_proj, ctx, self.caches["o_proj"])
        return out

    def _ffn(self, h: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        h_norm = self.norm2(h)
        if self.use_swiglu:
            g, self.caches["ffn_gate"] = _llu_step(self.ffn_gate, h_norm, self.caches["ffn_gate"], cond=cond)
            u, self.caches["ffn_up"] = _llu_step(self.ffn_up, h_norm, self.caches["ffn_up"], cond=cond)
            fused = F.silu(g) * u
            out, self.caches["ffn_down"] = _llu_step(self.ffn_down, fused, self.caches["ffn_down"], cond=cond)
            return out
        out, self.caches["ffn"] = _llu_step(self.ffn, h_norm, self.caches["ffn"], cond=cond)
        return out

    def step(self, h: torch.Tensor) -> torch.Tensor:
        if self.cond_provider is not None:
            # (SWA, GDN-2) -> (x, cond) -> FFN(StableLiquidLN) -> +residual
            x = self._attention(h) if self.use_attention else h
            cond, self.caches["cond_provider"] = _llu_step(
                self.cond_provider, h, self.caches["cond_provider"]
            )
            h = h + self._ffn(x, cond=cond)
        else:
            if self.use_attention:
                h = h + self._attention(h)
            h = h + self._ffn(h)
        return h


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LiquidTransformer(nn.Module):
    """Stacked :class:`LiquidTransformerBlock` with an LLU readout head.

    Processes a sequence token-by-token; the readout is applied to the full
    hidden stream so per-position predictions (e.g. at a query position) can be
    selected by the downstream task mask.
    """

    def __init__(
        self,
        d_model: int,
        out_dim: int,
        arch: str = "StableLiquidLN",
        num_layers: int = 2,
        window: int = 16,
        n_heads: int = 4,
        use_swiglu: bool = True,
        swiglu_mult: int = 4,
        rank: int = 4,
        decay_rate: float = 0.4,
        head_dim: int = 16,
        num_heads_gdn: int = 4,
        learnable_decay: bool = False,
        use_attention: bool = True,
        parameterization: str = "lora",
        lln_attn_dim: int = 32,
        lln_attn_heads: int = 2,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            n_heads = max(1, d_model // 16)
        n_heads = _valid_n_heads(d_model, n_heads)
        self.d_model = d_model
        self.out_dim = out_dim
        self.arch = arch

        proj_name, ffn_name = _arch_map(arch)
        proj_cls = ARCH_FACTORIES[proj_name]
        ffn_cls = ARCH_FACTORIES[ffn_name]
        proj_kwargs = _llu_kwargs_for(proj_cls, rank, decay_rate, head_dim, num_heads_gdn, learnable_decay, parameterization=parameterization, attn_dim=lln_attn_dim, attn_heads=lln_attn_heads)
        ffn_kwargs = _llu_kwargs_for(ffn_cls, rank, decay_rate, head_dim, num_heads_gdn, learnable_decay, parameterization=parameterization, attn_dim=lln_attn_dim, attn_heads=lln_attn_heads)

        # Optional GDN-2 cond provider (feeds `cond` to the StableLiquidLN FFN).
        # Its output is d_model-sized, so the FFN sublayers condition on a
        # `cond_dim=d_model` vector independent of their own in_features.
        cond_llu = GDNLiquidLN if arch == "StableGDNCondLiquidLN" else None
        if cond_llu is not None:
            ffn_kwargs = dict(ffn_kwargs)
            ffn_kwargs["cond_dim"] = d_model
            cond_kwargs = _llu_kwargs_for(
                GDNLiquidLN, rank=1, decay_rate=decay_rate,
                head_dim=8, num_heads=2, learnable_decay=learnable_decay,
                parameterization=parameterization,
            )
        else:
            cond_kwargs = None

        self.blocks = nn.ModuleList(
            [
                LiquidTransformerBlock(
                    d_model, n_heads, proj_cls, ffn_cls, window,
                    use_swiglu=use_swiglu, swiglu_mult=swiglu_mult,
                    proj_kwargs=proj_kwargs, ffn_kwargs=ffn_kwargs,
                    use_attention=use_attention,
                    cond_llu=cond_llu, cond_kwargs=cond_kwargs,
                )
                for _ in range(num_layers)
            ]
        )
        # Neutral per-position readout (no cross-step recurrence) for fair comparison.
        self.readout = StableLiquidLN(d_model, out_dim, **_llu_kwargs_for(StableLiquidLN, rank, parameterization=parameterization))

    def reset_state(self) -> None:
        for block in self.blocks:
            block.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        self.reset_state()
        B, T, D = x.shape
        stream = []
        for t in range(T):
            h = x[:, t]
            for block in self.blocks:
                h = block.step(h)
            stream.append(h)
        stream = torch.stack(stream, dim=1)  # (B, T, D)
        return self.readout(stream)          # (B, T, out_dim)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(arch: str, d_model: int, out_dim: int, **overrides: Any) -> LiquidTransformer:
    """Construct a :class:`LiquidTransformer` for ``arch`` with sensible defaults.

    An arch name suffixed with ``_noattn`` builds the same architecture with the
    sliding-window attention sublayer removed entirely (the LLU recurrence alone
    must carry sequence state). This is the attention-free ablation variant.
    """
    use_attention = overrides.pop("use_attention", True)
    base_arch = arch
    if arch.endswith("_noattn"):
        base_arch = arch[: -len("_noattn")]
        use_attention = False
    cfg = dict(
        num_layers=2,
        window=16,
        n_heads=4,
        use_swiglu=True,
        swiglu_mult=4,
        rank=4,
        decay_rate=0.4,
        head_dim=16,
        num_heads_gdn=4,
        learnable_decay=False,
        use_attention=use_attention,
        parameterization="lora",
        lln_attn_dim=32,
        lln_attn_heads=2,
    )
    cfg.update(overrides)
    cfg["n_heads"] = _valid_n_heads(d_model, cfg.get("n_heads", 4))
    return LiquidTransformer(d_model, out_dim, arch=base_arch, **cfg)


__all__ = [
    "ARCH_FACTORIES",
    "LiquidTransformerBlock",
    "LiquidTransformer",
    "build_model",
    "RMSNorm",
    "RECURRENT_ARCHS",
    "is_valid_arch",
]
