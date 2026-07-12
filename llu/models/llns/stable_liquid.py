r"""StableLiquidLN: production-oriented variant with nonlinear hypernetwork."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .base import BaseLLU
from .utils import (
    DEVICE,
    _activate,
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
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        r"""__init__(in_features, out_features, rank=4, hyper_hidden_dim=None, bias=True, dynamic_bias=False, factor_activation="norm", scale_init=0.01, normalize_input=True, init_method="hyperfan_in", parameterization="lora", device=None, dtype=torch.float32) -> None

        Args:
            in_features (int): size of each input sample.
            out_features (int): size of each output sample.
            rank (int): number of factor pairs :math:`(a_r, b_r)`.  Default: ``4``.
            hyper_hidden_dim (int, optional): hidden dimension of the MLP
                hypernetwork.  Default: ``None`` (``max(in_features // 4, rank * 16)``).
            bias (bool): whether the core :class:`~nn.Linear` has a learnable bias.
                Default: ``True``.
            dynamic_bias (bool): if ``True``, an input-dependent bias from an
                MLP is added to the output.  Default: ``False``.
            factor_activation (str): activation for the factor vectors.
                One of ``"tanh"``, ``"norm"``, ``"rmsnorm"``, or ``"none"``.
                Default: ``"norm"``.
            scale_init (float): initial value of the per-channel scalar multiplier
                on the adaptive path.  Default: ``0.01``.
            normalize_input (bool): if ``True``, apply RMSNorm to the conditioning
                input before the hypernetwork.  Default: ``True``.
            cond_dim (int, optional): dimension of the conditioning tensor fed to
                the hypernetwork.  Defaults to ``in_features`` so that, when no
                separate condition is supplied, the input itself conditions the
                factors.  Set it to a different size to drive the factors from an
                external ``cond`` whose last dimension differs from ``in_features``
                (e.g. a d_model-sized dynamic conditioner feeding layers whose
                ``in_features`` vary).
            init_method (str): weight initialisation method for the hypernetwork.
                One of ``"hyperfan_in"``, ``"hyperfan_out"``, ``"xavier"``,
                or ``"small"``.  Default: ``"hyperfan_in"``.
            parameterization (str): parameterization mode for the update, either
                ``"lora"`` or ``"svd"``. Default: ``"lora"``.
            device (torch.device, optional): the desired device of the parameters.
                Default: ``None``.
            dtype (torch.dtype): the desired data type of the parameters.
                Default: ``torch.float32``.
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

        # MLP hypernetwork (conditions on the `cond_dim`-sized cond vector)
        hidden_dim = hyper_hidden_dim or max(in_features // 4, rank * 16)

        if self.parameterization == "lora":
            # Generates both factor matrices dynamically
            self.hypernetwork = nn.Sequential(
                nn.Linear(self.cond_dim, hidden_dim, device=dev, dtype=dtype),
                nn.SiLU(),
                nn.Linear(hidden_dim, rank * (out_features + in_features), device=dev, dtype=dtype),
            )
        else:
            # SVD mode: hypernetwork outputs dynamic diagonal scaling vector
            self.hypernetwork = nn.Sequential(
                nn.Linear(self.cond_dim, hidden_dim, device=dev, dtype=dtype),
                nn.SiLU(),
                nn.Linear(hidden_dim, rank, device=dev, dtype=dtype),
            )
            self._create_svd_factors(dev, dtype)

        # Dynamic bias with MLP
        self.bias_dynamic: Optional[nn.Sequential] = (
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
        r"""_init_weights() -> None

        Initialise hypernetwork layers with the chosen init method.
        Under "lora", zeroes the b-section.
        Under "svd", zeroes the final layer of the hypernetwork and initialises U and V.
        """
        if self.parameterization == "lora":
            self._init_low_rank_adaptive(self.hypernetwork, self.rank * self.out_features, rank=self.rank)
        else:
            self._init_svd_projection(self.hypernetwork)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""forward(x, cond=None) -> Tensor

        Args:
            x (Tensor): input tensor of shape ``(..., in_features)``.
            cond (Tensor, optional): optional conditioning tensor.  When
            ``None``, *x* is used.  The conditioning drives the hypernetwork
            while *x* always goes through the core linear path.

        Returns:
            Tensor: output tensor of shape ``(..., out_features)``.
        """
        cond = cond if cond is not None else x

        # RMSNorm for magnitude invariance (over the conditioning dim)
        h_in = F.rms_norm(cond, (self.cond_dim,)) if self.normalize_input else cond

        core_out = self.linear_core(x)

        if self.parameterization == "lora":
            raw = self.hypernetwork(h_in)

            split = self.rank * self.out_features
            a_raw = raw[..., :split].reshape(*h_in.shape[:-1], self.rank, self.out_features)
            b_raw = raw[..., split:].reshape(*h_in.shape[:-1], self.rank, self.in_features)

            a = _activate(a_raw, self.factor_activation)
            b = _activate(b_raw, self.factor_activation)

            adaptive = self._compute_low_rank_adaptive(a, b, x)
        else:
            # Output g_raw has shape (..., rank)
            g_raw = self.hypernetwork(h_in)

            # Activate the scaling coefficients using factor_activation
            g = _activate(g_raw, self.factor_activation)

            adaptive = self._compute_svd_adaptive(x, g)

        out = core_out + adaptive
        out = self._apply_dynamic_bias(out, cond)

        return out

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, rank={self.rank}, "
            f"act={self.factor_activation}, norm_input={self.normalize_input}, "
            f"mode={self.parameterization}"
        )
