r"""LLM-scale models built from the LLU family.

Two architectures share one config so they can be compared at an identical
parameter budget:

* ``LiquidGDNCondLLM`` -- the novel design. Each block runs **sliding-window
  causal attention (SWA)** to produce the token-mixed ``X`` and a **Gated
  DeltaNet-2 (GDN-2)** recurrence to produce a ``cond`` vector. ``X`` and
  ``cond`` feed an *intermediary* MLP made of two :class:`StableLiquidLN`
  layers (each conditioned on a projected view of ``cond``); the result is
  added to the residual stream, and a standard SwiGLU FFN follows. This is the
  ``(SWA, GDN-2) -> (x, cond) -> StableLiquidLN FFN`` idea scaled to an LLM.
* ``GDN2BaselineLLM`` -- the lit_gpt-style reference: GDN-2 is the *mixer*
  (replacing attention entirely) with a SwiGLU FFN, built on the same
  CPU-compatible :mod:`llu.models.gdn2`. This is the comparison baseline.

Both use RoPE, pre-LN blocks with an optional parallel residual (GPT-NeoX
style), and a token embedding + final RMSNorm + lm_head.

Hardware note: these are written for CPU. GDN-2 falls back to its chunk
kernel in training and fused-recurrent kernel when decoding short sequences;
SWA uses ``F.scaled_dot_product_attention`` with a sliding-window causal mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gdn2 import GatedDeltaNet2
from .llns import (
    StableLiquidLN,
    CrossAttnLoraLN,
    SharedMomentumLiquidLN,
    BatchMomentumLiquidLN,
)

# Intermediary LLN classes that can sit between the SWA/X mixer and the
# main residual stream in ``LiquidGDNCondBlock``. Every entry here accepts a
# separate ``cond`` tensor (the GDN-2 conditioning stream) and is parameterised
# by ``rank`` / ``parameterization`` so it can run as a learned low-rank update:
#
# * ``StableLiquidLN``      -- input-adaptive hypernetwork factors (the original).
# * ``CrossAttnLoraLN``     -- LoRA-style factors refined by cross-attention over
#                              the ``cond`` sequence (the novel option).
# * ``SharedMomentumLiquidLN``   -- factors with an EMA momentum over the batch.
# * ``BatchMomentumLiquidLN``    -- factors with a per-batch-element momentum.
#
# ``RankRLiquidLN`` is deliberately excluded: it has no ``cond`` port, so it
# cannot act as the intermediary. The two GDN-2 LLUs (``GDNLiquidLN`` /
# ``MomentumGDNLiquidLN``) are also excluded because the ``ours`` block already
# produces ``cond`` via a GDN-2 recurrence -- stacking a second GDN-2 inside the
# intermediary would be redundant and ~5x slower/step.
LLN_REGISTRY = {
    "StableLiquidLN": StableLiquidLN,
    "CrossAttnLoraLN": CrossAttnLoraLN,
    "SharedMomentumLiquidLN": SharedMomentumLiquidLN,
    "BatchMomentumLiquidLN": BatchMomentumLiquidLN,
}

def _lln_kwargs_for(
    cls: type,
    *,
    rank: int,
    cond_dim: int,
    parameterization: str,
    normalize_input: bool,
    factor_activation: str,
    attn_dim: int,
    attn_heads: int,
    decay_rate: float = 0.4,
    learnable_decay_rate: bool = False,
) -> dict:
    """Forward only the kwargs ``cls.__init__`` accepts.

    Keeps ``IntermediaryMLP`` robust across LLN variants: ``StableLiquidLN``
    ignores ``attn_dim``/``attn_heads``/``decay_rate``; ``CrossAttnLoraLN``
    consumes the attention kwargs; the momentum variants consume ``decay_rate``.
    Every candidate kwarg is dropped unless the class declares it.
    """
    params = set(inspect.signature(cls.__init__).parameters)
    cand = dict(
        rank=rank,
        cond_dim=cond_dim,
        parameterization=parameterization,
        normalize_input=normalize_input,
        factor_activation=factor_activation,
        attn_dim=attn_dim,
        attn_heads=attn_heads,
        decay_rate=decay_rate,
        learnable_decay_rate=learnable_decay_rate,
    )
    return {k: v for k, v in cand.items() if k in params}


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """Standard rotary position embedding (cached inv_freq)."""

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)            # (T, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)          # (T, dim)
        return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, T, H, dh); cos/sin: (T, dh)
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    return x * cos + _rotate_half(x) * sin


def _causal_window_mask(seq_len: int, window: int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(seq_len, device=device)
    dist = idx[:, None] - idx[None, :]
    ok = (dist >= 0) & (dist < window)
    return torch.where(ok, torch.zeros_like(dist, dtype=torch.float32), torch.full_like(dist, float("-inf"), dtype=torch.float32))


# ---------------------------------------------------------------------------
# Sublayers
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    """Main feed-forward MLP (gate/up/down with SiLU) — plain nn.Linear."""

    def __init__(self, d_model: int, mult: int = 4) -> None:
        super().__init__()
        hidden = int(d_model * mult)
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class SlidingWindowAttention(nn.Module):
    """Sliding-window causal attention with RoPE, via scaled_dot_product_attention."""

    def __init__(self, d_model: int, n_head: int, window: int) -> None:
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.window = window
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        mask = _causal_window_mask(T, self.window, x.device)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask
        )
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class IntermediaryMLP(nn.Module):
    """Intermediary liquid MLP sitting between the SWA/X mixer and the residual.

    ``X`` (from SWA) flows through two LLN layers (``lln_cls``); each is
    conditioned on a *separate* linear projection of the GDN-2 ``cond``. The
    LLN does the input-adaptive heavy lifting. ``lln_cls`` is configurable
    (``StableLiquidLN`` by default, ``CrossAttnLoraLN`` to refine factors via
    cross-attention over the ``cond`` sequence). Only the kwargs each LLN
    accepts are forwarded.
    """

    def __init__(
        self,
        d_model: int,
        inter_dim: int,
        cond_dim: int,
        rank: int,
        parameterization: str = "svd",
        lln_cls: type = StableLiquidLN,
        attn_dim: int = 32,
        attn_heads: int = 2,
    ) -> None:
        super().__init__()
        self.proj1 = nn.Linear(cond_dim, cond_dim, bias=False)
        self.proj2 = nn.Linear(cond_dim, cond_dim, bias=False)
        # X and Cond may have different dims: in_features varies per layer,
        # cond_dim is fixed (the GDN-2 output size). StableLiquidLN and
        # CrossAttnLoraLN both decouple the two via their `cond_dim` argument.
        lln_kwargs = _lln_kwargs_for(
            lln_cls, rank=rank, cond_dim=cond_dim,
            parameterization=parameterization, normalize_input=False,
            factor_activation="norm", attn_dim=attn_dim, attn_heads=attn_heads,
        )
        self.sl1 = lln_cls(d_model, inter_dim, **lln_kwargs)
        self.act = nn.SiLU()
        self.sl2 = lln_cls(inter_dim, d_model, **lln_kwargs)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        c1 = self.proj1(cond)
        c2 = self.proj2(cond)
        h = self.sl1(x, cond=c1)
        h = self.act(h)
        h = self.sl2(h, cond=c2)
        return h


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

class LiquidGDNCondBlock(nn.Module):
    """SWA (X) + GDN-2 (cond) -> Intermediary liquid MLP -> residual.

    The intermediary MLP class is configurable (``cfg.lln``): ``StableLiquidLN``
    (input-adaptive factors) or ``CrossAttnLoraLN`` (factors refined by
    cross-attention over the GDN-2 ``cond`` sequence).
    """

    def __init__(self, cfg: "LLMConfig") -> None:
        super().__init__()
        d, H, w = cfg.n_embd, cfg.n_head, cfg.window
        inter = int(d * cfg.inter_mult)
        self.parallel = cfg.parallel_residual

        self.norm1 = nn.RMSNorm(d)
        self.swa = SlidingWindowAttention(d, H, w)
        self.gdn2 = GatedDeltaNet2(
            hidden_size=d, expand_v=cfg.gdn_expand_v,
            head_dim=d // H, num_heads=H, mode="chunk",
        )
        lln_cls = LLN_REGISTRY[cfg.lln]
        self.inter = IntermediaryMLP(
            d, inter, d, cfg.rank, parameterization=cfg.parameterization,
            lln_cls=lln_cls, attn_dim=cfg.lln_attn_dim, attn_heads=cfg.lln_attn_heads,
        )
        self.norm2 = None if self.parallel else nn.RMSNorm(d)
        self.mlp = SwiGLU(d, cfg.swiglu_mult)

        # Initialize linear layers with Xavier uniform
        for m in [self.swa.qkv, self.swa.proj, self.inter.proj1, self.inter.proj2,
                  self.mlp.gate, self.mlp.up, self.mlp.down]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        n1 = self.norm1(x)
        X = self.swa(n1, cos, sin)
        Cond = self.gdn2(n1)[0]
        h = self.inter(X, Cond)
        if self.parallel:
            x = x + h + self.mlp(n1)
        else:
            x = x + h
            x = x + self.mlp(self.norm2(x))  # type: ignore[arg-type]
        return x


class GDN2BaselineBlock(nn.Module):
    """lit_gpt-style block: GDN-2 mixer (no attention) + SwiGLU FFN."""

    def __init__(self, cfg: "LLMConfig") -> None:
        super().__init__()
        d, H = cfg.n_embd, cfg.n_head
        self.parallel = cfg.parallel_residual
        self.norm1 = nn.RMSNorm(d)
        self.gdn2 = GatedDeltaNet2(
            hidden_size=d, expand_v=cfg.gdn_expand_v,
            head_dim=d // H, num_heads=H, mode="chunk",
        )
        self.norm2 = None if self.parallel else nn.RMSNorm(d)
        self.mlp = SwiGLU(d, cfg.swiglu_mult)

        # Initialize linear layers with Xavier uniform
        for m in [self.mlp.gate, self.mlp.up, self.mlp.down]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        n1 = self.norm1(x)
        h = self.gdn2(n1)[0]
        if self.parallel:
            x = x + h + self.mlp(n1)
        else:
            x = x + h
            x = x + self.mlp(self.norm2(x))  # type: ignore[arg-type]
        return x


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class LiquidGDNCondLLM(nn.Module):
    """Our architecture: SWA + GDN-2-conditioned liquid intermediary MLP.

    The intermediary LLN is configurable via ``cfg.lln`` (``StableLiquidLN``
    by default; pass ``CrossAttnLoraLN`` to refine the factors with
    cross-attention over the GDN-2 conditioning stream).
    """

    def __init__(self, cfg: "LLMConfig") -> None:
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        # nn.Embedding defaults to N(0, 1); at vocab=50257 that makes logits
        # explode (init CE ~300, training never recovers). Scale to
        # 1/sqrt(n_embd) so logits start at unit scale and init CE ~= ln(vocab).
        nn.init.normal_(self.wte.weight, mean=0.0, std=1.0 / (cfg.n_embd ** 0.5))
        self.blocks = nn.ModuleList([LiquidGDNCondBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_embed:
            self.lm_head.weight = self.wte.weight
        else:
            nn.init.xavier_uniform_(self.lm_head.weight)
        self.rope = RotaryEmbedding(cfg.n_embd // cfg.n_head)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        x = self.wte(idx)
        cos, sin = self.rope(T, x.device)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.ln_f(x)
        return self.lm_head(x)


class GDN2BaselineLLM(nn.Module):
    """Reference GDN-2 LLM (GDN-2 as the mixer, no attention)."""

    def __init__(self, cfg: "LLMConfig") -> None:
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        # nn.Embedding defaults to N(0, 1); at vocab=50257 that makes logits
        # explode (init CE ~300, training never recovers). Scale to
        # 1/sqrt(n_embd) so logits start at unit scale and init CE ~= ln(vocab).
        nn.init.normal_(self.wte.weight, mean=0.0, std=1.0 / (cfg.n_embd ** 0.5))
        self.blocks = nn.ModuleList([GDN2BaselineBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_embed:
            self.lm_head.weight = self.wte.weight
        else:
            nn.init.xavier_uniform_(self.lm_head.weight)
        self.rope = RotaryEmbedding(cfg.n_embd // cfg.n_head)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        x = self.wte(idx)
        cos, sin = self.rope(T, x.device)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.ln_f(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# Config / factory
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    """Shared config for both LLM variants (so they compare at one budget)."""

    variant: str = "ours"            # "ours" | "baseline"
    vocab_size: int = 50257          # GPT-2 vocab
    n_layer: int = 8
    n_embd: int = 320
    n_head: int = 8
    window: int = 256                # SWA context; >= block_size => global
    block_size: int = 1024
    swiglu_mult: int = 4
    inter_mult: int = 4              # intermediary StableLiquidLN hidden mult
    rank: int = 4                    # StableLiquidLN factor rank
    gdn_expand_v: float = 1.0        # GDN-2 value expansion ("2x dimension")
    parallel_residual: bool = True   # GPT-NeoX parallel residual
    tie_embed: bool = True           # tie wte/lm_head (saves params)
    parameterization: str = "svd"    # "lora" | "svd"
    lln: str = "StableLiquidLN"       # intermediary LLN class name
    lln_attn_dim: int = 32           # CrossAttnLoraLN cross-attn dim
    lln_attn_heads: int = 2          # CrossAttnLoraLN cross-attn heads

    # Preset budgets, per architecture. The baseline (GDN-2 mixer, no
    # attention) is cheaper per layer than ``ours`` (which also pays for SWA +
    # the two StableLiquidLN intermediary layers), so it gets more layers to
    # land at the same total parameter budget.
    #
    # ``tiny`` is sized for a weak laptop CPU (i5-8250U, ~7.6 GB RAM, no CUDA):
    # a full LLN comparison across 4 variants fits in a few minutes and a few
    # hundred MB. ``small``/``medium``/``0.5B`` are for GPU-class hardware.
    PRESETS = {
        "tiny":    {"ours": dict(n_layer=2,  n_embd=128),
                    "baseline": dict(n_layer=4, n_embd=128)},
        "small":  {"ours": dict(n_layer=8,  n_embd=320),
                  "baseline": dict(n_layer=24, n_embd=320)},
        "medium": {"ours": dict(n_layer=12, n_embd=512),
                  "baseline": dict(n_layer=36, n_embd=512)},
        "0.5B":   {"ours": dict(n_layer=24, n_embd=1024),
                  "baseline": dict(n_layer=72, n_embd=1024)},
    }

    @classmethod
    def from_preset(cls, preset: str = "small", **overrides: Any) -> "LLMConfig":
        variant = overrides.get("variant", "ours")
        cfg = cls(**overrides)
        if preset in cls.PRESETS:
            for k, v in cls.PRESETS[preset].get(variant, {}).items():
                setattr(cfg, k, v)
        return cfg


def build_llm(variant: str = "ours", preset: str = "small", lln: str = "StableLiquidLN", **overrides: Any) -> nn.Module:
    """Build an LLM of the requested variant at a shared parameter budget."""
    overrides.setdefault("variant", variant)
    overrides["lln"] = lln
    cfg = LLMConfig.from_preset(preset, **overrides)
    model = LiquidGDNCondLLM(cfg) if variant == "ours" else GDN2BaselineLLM(cfg)
    return model


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


__all__ = [
    "LLMConfig",
    "LLN_REGISTRY",
    "LiquidGDNCondLLM",
    "GDN2BaselineLLM",
    "build_llm",
    "num_params",
]
