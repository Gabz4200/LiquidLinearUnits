r"""EngramRetrievedLoraLN: Engram-retrieved LoRA expert bank modulated by diagonal sequence gate.

This Liquid Linear Unit combines pattern-indexed retrieval with continuous sequence-conditioned
gating over low-rank adapters:

1. **Retrieval**: The sequence conditioning tensor ``cond`` (e.g. from an Engram memory lookup
   or local hidden state) indexes a candidate bank of :math:`K` LoRA expert matrices (or singular
   bases).
2. **Sequence-Conditioned Diagonal Gate**: The tensor ``cond`` computes a per-rank diagonal
   modulation scale :math:`g(\text{cond})` that rescales each low-rank subspace direction.
3. **Adaptive Update**: The final adapter output is applied as:
   :math:`y_{\text{adaptive}} = \sum_k w_k B_k \operatorname{diag}(g(\text{cond})) A_k x`.

Supports both ``"lora"`` (expert bank of :math:`(A_k, B_k)` pairs) and ``"svd"`` (expert bank of
singular factors :math:`(V_k, U_k)`). Zero-initialised so the adaptive path contributes zero at
step 1.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .base import BaseLLU
from .utils import (
    _activate,
    _small_init,
    _validate_parameterization,
    _zero_out_last,
)


class EngramRetrievedLoraLN(BaseLLU):
    """Engram-retrieved LoRA expert bank modulated by a sequence-conditioned diagonal gate."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        num_experts: int = 8,
        bias: bool = True,
        dynamic_bias: bool = False,
        factor_activation: str = "norm",
        scale_init: float = 0.01,
        normalize_input: bool = True,
        cond_dim: int | None = None,
        init_method: str = "hyperfan_in",
        parameterization: str = "lora",
        lora_alpha: float = 1.0,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if rank < 1:
            raise ValueError("rank must be >= 1")
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        _validate_parameterization(parameterization)

        super().__init__(
            in_features,
            out_features,
            bias=bias,
            scale_init=scale_init,
            factor_activation=factor_activation,
            init_method=init_method,
            device=device,
            dtype=dtype,
        )

        self.rank = rank
        self.num_experts = num_experts
        self.parameterization = parameterization
        self.lora_alpha = lora_alpha
        self.normalize_input = normalize_input
        self.cond_dim = cond_dim if cond_dim is not None else in_features

        self.retrieval_router = nn.Linear(
            self.cond_dim, num_experts, bias=False, device=device, dtype=dtype
        )
        self.gate_net = nn.Linear(
            self.cond_dim, rank, bias=False, device=device, dtype=dtype
        )

        if parameterization == "lora":
            self.A_bank = nn.Parameter(
                torch.empty(num_experts, rank, in_features, device=device, dtype=dtype)
            )
            self.B_bank = nn.Parameter(
                torch.empty(num_experts, out_features, rank, device=device, dtype=dtype)
            )
        else:
            self.V_bank = nn.Parameter(
                torch.empty(num_experts, in_features, rank, device=device, dtype=dtype)
            )
            self.U_bank = nn.Parameter(
                torch.empty(num_experts, rank, out_features, device=device, dtype=dtype)
            )

        if dynamic_bias:
            self.bias_dynamic = nn.Linear(
                self.cond_dim, out_features, device=device, dtype=dtype
            )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.retrieval_router.weight)
        _small_init(self.gate_net)
        _zero_out_last(self.gate_net)

        if self.parameterization == "lora":
            nn.init.xavier_uniform_(self.A_bank)
            nn.init.zeros_(self.B_bank)
        else:
            nn.init.xavier_uniform_(self.V_bank)
            nn.init.xavier_uniform_(self.U_bank)

        self._init_bias_dynamic()

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise RuntimeError(
                f"x last dimension ({x.shape[-1]}) does not match in_features ({self.in_features})"
            )

        cond = cond if cond is not None else x
        if cond.shape[-1] != self.cond_dim:
            raise RuntimeError(
                f"cond last dimension ({cond.shape[-1]}) does not match cond_dim ({self.cond_dim})"
            )

        norm_x = F.layer_norm(x, (self.in_features,)) if self.normalize_input else x
        norm_cond = F.layer_norm(cond, (self.cond_dim,)) if self.normalize_input else cond

        logits = self.retrieval_router(norm_cond)
        weights = F.softmax(logits, dim=-1)

        raw_g = self.gate_net(norm_cond)
        g = _activate(raw_g, self.factor_activation)

        if self.parameterization == "lora":
            v = torch.einsum("...i, ...k, kri -> ...r", norm_x, weights, self.A_bank)
            v_gated = v * g
            y_adaptive = torch.einsum("...r, ...k, kor -> ...o", v_gated, weights, self.B_bank)
        else:
            v = torch.einsum("...i, ...k, kir -> ...r", norm_x, weights, self.V_bank)
            v_gated = v * g
            y_adaptive = torch.einsum("...r, ...k, kro -> ...o", v_gated, weights, self.U_bank)

        scale = self.lora_alpha / self.rank
        out = self.linear_core(x) + scale * y_adaptive
        return self._apply_dynamic_bias(out, norm_cond)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, num_experts={self.num_experts}, "
            f"parameterization='{self.parameterization}'"
        )


__all__ = ["EngramRetrievedLoraLN"]
