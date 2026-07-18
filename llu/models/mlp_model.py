r"""Static (single-input / single-output) MLP built entirely from LLU layers.

``LiquidMLP`` replaces every ``nn.Linear`` in a plain feed-forward net with an
LLU from :mod:`llu.models.llns`, so the same liquid-linear mechanism studied in
the sequence and graph benchmarks can be screened on static function-
approximation tasks -- modular arithmetic / grokking, spectral bias, sparse
parity, teacher-student recovery, and so on.

All LLU variants satisfy the init-zero invariant (``model(x) == linear_core(x)``
right after construction), so the network starts as an ordinary MLP and the
input-conditioned adaptive path is what differentiates the architectures. Each
layer is called as ``lln(x)``; LLNs that accept a ``cond`` tensor fall back to
``cond = x`` (self-conditioning), which is exactly the regime these tasks probe.
"""

from __future__ import annotations

import inspect

import torch
import torch.nn as nn

from .llns import (
    StableLiquidLN,
    RankRLiquidLN,
    SharedMomentumLiquidLN,
    BatchMomentumLiquidLN,
)

# Plain LLU variants that make sense as static linear maps. ``CrossAttnLoraLN``
# is excluded because its cross-attention source degenerates to ``cond = x``
# here, and the GDN variants carry sequence/attention machinery that does not
# apply to a stateless vector map.
IO_LLN_REGISTRY = {
    "StableLiquidLN": StableLiquidLN,
    "RankRLiquidLN": RankRLiquidLN,
    "SharedMomentumLiquidLN": SharedMomentumLiquidLN,
    "BatchMomentumLiquidLN": BatchMomentumLiquidLN,
}


def _lln_kwargs(cls: type, *, in_f: int, out_f: int, rank: int,
                parameterization: str, normalize_input: bool = True) -> dict:
    """Build kwargs for an LLU layer, forwarding only params the class accepts."""
    params = set(inspect.signature(cls.__init__).parameters)
    kw: dict = {"in_features": in_f, "out_features": out_f}
    if "rank" in params:
        kw["rank"] = rank
    if "parameterization" in params:
        kw["parameterization"] = parameterization
    if "normalize_input" in params:
        kw["normalize_input"] = normalize_input
    if "cond_dim" in params:
        # cond defaults to x; keep the conditioning dimension explicit.
        kw["cond_dim"] = in_f
    return kw


_ACTS = {
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
    "silu": nn.SiLU(),
    "tanh": nn.Tanh(),
    "none": nn.Identity(),
}


class LiquidMLP(nn.Module):
    """Feed-forward net whose every linear map is an LLU from ``llu.models.llns``.

    Args:
        in_dim: input vector width.
        hidden: width of every hidden layer.
        n_layers: number of hidden layers (>=1).
        out_dim: output vector width.
        lln_cls: plain LLU class used for every linear map.
        parameterization: ``"lora"`` or ``"svd"`` (forwarded if accepted).
        rank: LoRA/SVD rank (forwarded if accepted).
        act: nonlinearity between hidden layers (none after the final layer).
    """

    def __init__(self, in_dim: int, hidden: int = 128, n_layers: int = 2,
                 out_dim: int = 1, lln_cls: type = StableLiquidLN,
                 parameterization: str = "lora", rank: int = 4,
                 act: str = "relu") -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if act not in _ACTS:
            raise ValueError(f"unknown act {act!r}; choose from {sorted(_ACTS)}")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden = hidden
        self.n_layers = n_layers
        self.act = _ACTS[act]
        dims = [in_dim] + [hidden] * n_layers + [out_dim]
        self.layers = nn.ModuleList([
            lln_cls(**_lln_kwargs(
                lln_cls, in_f=dims[i], out_f=dims[i + 1],
                rank=rank, parameterization=parameterization,
            ))
            for i in range(len(dims) - 1)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.act(x)
        return x

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


__all__ = ["LiquidMLP", "IO_LLN_REGISTRY"]
