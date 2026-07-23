r"""StableLiquidLN: production-oriented variant with nonlinear hypernetwork."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .base import BaseLLU
from .utils import (
    DEVICE,
    _activate,
    _factorized_hyperfan_init,
    _validate_parameterization,
)


class StableLiquidLN(BaseLLU):
    """Production‑oriented variant with nonlinear hypernetwork and support for both LoRA and SVD parameterizations.

    Supports two modes of parameterization via the `parameterization` argument:

    1. "lora" (default): Generates both factor matrices dynamically:
        W(x) = W_0 + A(cond) B(cond)^T
        This provides high expressive capacity suitable for synthetic tasks.

    2. "svd": Utilizes learned static factor matrices U and V, generating only a diagonal scale g:
        W(x) = W_0 + U diag(g(cond)) V^T
        This is extremely parameter-efficient and fast, suitable for large LLMs.

    Unlike simpler variants, this module accepts an optional separate *cond* tensor for conditioning.
    Zero-initialised so the adaptive path contributes nothing at step 1.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        hyper_hidden_dim: Optional[int] = None,
        bias: bool = True,
        dynamic_bias: bool = False,
        factor_activation: str = "norm",
        scale_init: float = 0.01,
        normalize_input: bool = True,
        cond_dim: Optional[int] = None,
        init_method: str = "hyperfan_in",
        parameterization: str = "lora",
        lora_alpha: float = 1.0,
        factorized: bool = False,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        r"""__init__(in_features, out_features, rank=4, ...) -> None

        Args:
            in_features: size of each input sample.
            out_features: size of each output sample.
            rank: number of factor pairs :math:`(a_r, b_r)`.  Default: ``4``.
            hyper_hidden_dim: hidden dimension of the MLP hypernetwork.
                Default: ``None`` (``max(in_features // 4, rank * 16)``).
            bias: whether the core :class:`~nn.Linear` has a learnable bias.
                Default: ``True``.
            dynamic_bias: if ``True``, an input-dependent bias from an
                MLP is added to the output.  Default: ``False``.
            factor_activation: activation for the factor vectors.
                One of ``"tanh"``, ``"norm"``, ``"rmsnorm"``, or ``"none"``.
                Default: ``"norm"``.
            scale_init: initial value of the per-channel scalar multiplier
                on the adaptive path.  Default: ``0.01``.
            normalize_input: if ``True``, apply RMSNorm to the conditioning
                input before the hypernetwork.  Default: ``True``.
            cond_dim: dimension of the conditioning tensor fed to
                the hypernetwork.  Defaults to ``in_features``.
            init_method: weight initialisation method for the hypernetwork.
                One of ``"hyperfan_in"``, ``"hyperfan_out"``, ``"xavier"``,
                or ``"small"``.  Default: ``"hyperfan_in"``.
            parameterization: parameterization mode for the update, either
                ``"lora"`` or ``"svd"``. Default: ``"lora"``.
            lora_alpha: LoRA alpha for scaling (``alpha / rank``).
                Default: ``1.0``.
            factorized: if ``True``, use separate A/B projections
                (Zhyper-style) instead of a single monolithic output.
                Reduces hypernetwork parameters by ~2x.  Default: ``False``.
            device: device of parameters.  Default: ``None``.
            dtype: data type of parameters.  Default: ``torch.float32``.
        """
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        _validate_parameterization(parameterization)

        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            scale_init=scale_init,
            factor_activation=factor_activation,
            init_method=init_method,
            device=device,
            dtype=dtype,
        )
        dev = device if device is not None else DEVICE

        self.rank = rank
        self.normalize_input = normalize_input
        self.cond_dim = cond_dim if cond_dim is not None else in_features
        self.parameterization = parameterization
        self.lora_alpha = lora_alpha
        self.factorized = factorized

        hidden_dim = hyper_hidden_dim or max(in_features // 4, rank * 16)

        if self.parameterization == "lora":
            if factorized:
                self.proj_a = nn.Sequential(
                    nn.Linear(self.cond_dim, hidden_dim, device=dev, dtype=dtype),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, rank * out_features, device=dev, dtype=dtype),
                )
                self.proj_b = nn.Sequential(
                    nn.Linear(self.cond_dim, hidden_dim, device=dev, dtype=dtype),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, rank * in_features, device=dev, dtype=dtype),
                )
                self.hypernetwork = None
            else:
                self.hypernetwork = nn.Sequential(
                    nn.Linear(self.cond_dim, hidden_dim, device=dev, dtype=dtype),
                    nn.SiLU(),
                    nn.Linear(
                        hidden_dim, rank * (out_features + in_features), device=dev, dtype=dtype
                    ),
                )
                self.proj_a = None
                self.proj_b = None
        else:
            self.hypernetwork = nn.Sequential(
                nn.Linear(self.cond_dim, hidden_dim, device=dev, dtype=dtype),
                nn.SiLU(),
                nn.Linear(hidden_dim, rank, device=dev, dtype=dtype),
            )
            self.proj_a = None
            self.proj_b = None
            self._create_svd_factors(dev, dtype)

        self.bias_dynamic: Optional[nn.Module] = (
            nn.Sequential(
                nn.Linear(self.cond_dim, hidden_dim, device=dev, dtype=dtype),
                nn.SiLU(),
                nn.Linear(hidden_dim, out_features, device=dev, dtype=dtype),
            )
            if dynamic_bias
            else None
        )

        self._init_weights()

    def _init_weights(self) -> None:
        if self.parameterization == "lora":
            if self.factorized:
                _factorized_hyperfan_init(
                    self.proj_a,
                    self.proj_b,
                    self.in_features,
                    self.out_features,
                    self.rank,
                )
                last_b = self.proj_b[-1] if isinstance(self.proj_b, nn.Sequential) else self.proj_b
                with torch.no_grad():
                    last_b.weight.data.zero_()
                    if last_b.bias is not None:
                        last_b.bias.data.zero_()
            else:
                self._init_low_rank_adaptive(
                    self.hypernetwork, self.rank * self.out_features, rank=self.rank
                )
        else:
            self._init_svd_projection(self.hypernetwork)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        cond = cond if cond is not None else x
        h_in = F.rms_norm(cond, (self.cond_dim,)) if self.normalize_input else cond

        core_out = self.linear_core(x)

        if self.parameterization == "lora":
            if self.factorized:
                a_raw = self.proj_a(h_in).reshape(*h_in.shape[:-1], self.rank, self.out_features)
                b_raw = self.proj_b(h_in).reshape(*h_in.shape[:-1], self.rank, self.in_features)
            else:
                raw = self.hypernetwork(h_in)
                split = self.rank * self.out_features
                a_raw = raw[..., :split].reshape(*h_in.shape[:-1], self.rank, self.out_features)
                b_raw = raw[..., split:].reshape(*h_in.shape[:-1], self.rank, self.in_features)

            a = _activate(a_raw, self.factor_activation)
            b = _activate(b_raw, self.factor_activation)
            adaptive = self._compute_low_rank_adaptive(a, b, x)
        else:
            g_raw = self.hypernetwork(h_in)
            g = _activate(g_raw, self.factor_activation)
            adaptive = self._compute_svd_adaptive(x, g)

        lora_scale = self.lora_alpha / self.rank
        out = core_out + lora_scale * adaptive
        out = self._apply_dynamic_bias(out, cond)
        return out

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, rank={self.rank}, "
            f"act={self.factor_activation}, norm_input={self.normalize_input}, "
            f"mode={self.parameterization}, factorized={self.factorized}, "
            f"lora_alpha={self.lora_alpha}"
        )
