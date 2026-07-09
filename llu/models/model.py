"""Liquid Linear Units variants: input-adaptive linear transformations.

Each variant replaces a standard ``nn.Linear`` with a core path plus an
input-conditioned low-rank (or full) update via hypernetwork.

The adaptive path is zero‑initialised so the layer behaves as a
normal linear layer at step 1 (residual identity).
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Helpers

_VALID_ACTIVATIONS = frozenset({"tanh", "norm", "rmsnorm", "none"})


def _activate(t: torch.Tensor, mode: str, eps: float = 1e-6) -> torch.Tensor:
    """Apply one of the supported factor activations / normalisations."""
    if mode == "none":
        return t
    if mode == "tanh":
        return torch.tanh(t)
    if mode == "norm":
        return F.normalize(t, dim=-1, p=2, eps=eps)
    if mode == "rmsnorm":
        return F.rms_norm(t, (t.shape[-1],), eps=eps)
    raise ValueError(
        f"Unknown factor_activation '{mode}'; expected one of {sorted(_VALID_ACTIVATIONS)}"
    )


def _small_init(module: nn.Module, gain: float = 0.02) -> None:
    """Apply xavier-uniform with small gain."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _last_linear(module: nn.Module) -> Optional[nn.Linear]:
    """Return the last ``nn.Linear`` in a module or ``nn.Sequential``, or *None*."""
    if isinstance(module, nn.Sequential):
        if len(module) == 0:
            return None
        module = module[-1]
    return module if isinstance(module, nn.Linear) else None


def _zero_out_last(module: nn.Module) -> None:
    """Zero the weights (and bias) of the last ``Linear`` in a chain.

    Respects ``nn.Sequential`` — only the final layer is zeroed so that earlier
    layers can keep a standard init and receive gradient from step 1.

    .. note::
       This function zeros the *entire* output, which is correct for
       non‑factored outputs (e.g. full weight matrix in ``LiquidLinear``).
       For factored rank outputs use ``_zero_b_section`` instead so that
       the a‑factors remain active for gradient flow.
    """
    last = _last_linear(module)
    if last is not None:
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            nn.init.zeros_(last.bias)


def _zero_b_section(module: nn.Module, b_start: int) -> None:
    """Zero the **b‑section** rows (``[b_start:]``) of the last linear layer.

    The a‑section (``[:b_start]``) keeps its previous init.  This partial
    zero‑init ensures that at step 1:

    * the adaptive output is zero (because b‑factors are zero), *yet*
    * gradients flow into the hypernetwork (because a‑factors are non‑zero).

    The a‑section (``[:b_start]``) should have non‑zero weights (from a prior
    init call such as ``_small_init``) so gradient flows through the
    hypernetwork.  The bias is zeroed independently here.
    """
    last = _last_linear(module)
    if last is not None:
        with torch.no_grad():
            last.weight.data[b_start:].zero_()
            # Zero entire bias (shared across a/b, a-section still fires via W_a@h).
            if last.bias is not None:
                last.bias.data.zero_()


# _FreezeMixin: shared freeze_core / freeze_hypernetwork


class _FreezeMixin:
    """Mixin providing ``freeze_core`` and ``freeze_hypernetwork``.

    Subclasses MUST have a ``linear_core`` attribute (a module whose
    parameters delimit the "core" from the "adaptive path").
    """

    def freeze_core(self) -> None:
        """Freeze core weights, train only adaptive path."""
        for p in self.linear_core.parameters():
            p.requires_grad = False

    def freeze_hypernetwork(self) -> None:
        """Freeze the entire adaptive path, training only the core."""
        core_ids = {id(p) for p in self.linear_core.parameters()}
        for p in self.parameters():
            if id(p) not in core_ids:
                p.requires_grad = False


# LiquidLinear: full per‑example weight matrix


class LiquidLinear(_FreezeMixin, nn.Module):
    """Input‑conditioned full weight matrix.

    Memory **O(B · O · I)**: use only for small dimensions.  The hypernetwork
    is a single linear layer; the generated full matrix is squashed with
    *factor_activation* (only ``"tanh"`` or ``"none"`` are well‑motivated here).

    Zero-initialised so the layer behaves as a plain ``nn.Linear`` at step 1.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dynamic_bias: bool = True,
        factor_activation: str = "tanh",
        scale_init: float = 0.9,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        dev = device if device is not None else DEVICE

        self.in_features = in_features
        self.out_features = out_features
        self.factor_activation = factor_activation

        # Core weight (always active)
        self.linear_core = nn.Linear(in_features, out_features, bias=bias, device=dev, dtype=dtype)

        # Hypernetwork -> full out×in matrix
        self.hypernetwork = nn.Linear(
            in_features,
            out_features * in_features,
            bias=True,
            device=dev,
            dtype=dtype,
        )

        # Per-channel dial for adaptive path
        self.scale = nn.Parameter(torch.full((out_features,), scale_init, device=dev, dtype=dtype))

        # Optional input‑dependent bias
        self.bias_dynamic: Optional[nn.Linear] = (
            nn.Linear(in_features, out_features, bias=True, device=dev, dtype=dtype)
            if dynamic_bias
            else None
        )

        self._init_weights()

    # Init

    def _init_weights(self) -> None:
        # Zero hypernetwork -> adaptive path = 0
        _small_init(self.hypernetwork)
        _zero_out_last(self.hypernetwork)

        if self.bias_dynamic is not None:
            _small_init(self.bias_dynamic)
            _zero_out_last(self.bias_dynamic)

    # Forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        core_out = self.linear_core(x)  # (..., O)

        # Dynamic matrix, activated
        W_raw = self.hypernetwork(x)  # (..., O*I)
        W = _activate(W_raw, self.factor_activation)
        W = W.reshape(*x.shape[:-1], self.out_features, self.in_features)  # (..., O, I)

        # delta_W = W @ x
        adaptive = torch.matmul(W, x.unsqueeze(-1)).squeeze(-1)  # (..., O)

        # Scale adaptive path, add bias
        out = core_out + adaptive * self.scale

        if self.bias_dynamic is not None:
            out = out + self.bias_dynamic(x)

        return out

    # Utilities

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, act={self.factor_activation}"


# Rank1LiquidLN: rank‑1 adaptive factors


class Rank1LiquidLN(_FreezeMixin, nn.Module):
    """Input‑conditioned rank‑1 update (lightweight).

    The hypernetwork outputs a single pair of vectors :math:`a \\in \\mathbb{R}^O`,
    :math:`b \\in \\mathbb{R}^I` which are then activated (default ``"norm"``).
    The effective dynamic update is :math:`\\Delta W = a \\otimes b`.

    Zero-initialised so :math:`\\Delta W = 0` at step 1.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dynamic_bias: bool = False,
        factor_activation: str = "norm",
        scale_init: float = 0.5,
        normalize_input: bool = False,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        dev = device if device is not None else DEVICE

        self.in_features = in_features
        self.out_features = out_features
        self.factor_activation = factor_activation
        self.normalize_input = normalize_input

        self.linear_core = nn.Linear(in_features, out_features, bias=bias, device=dev, dtype=dtype)

        # Single hypernetwork -> a (O) + b (I)
        self.hypernetwork = nn.Linear(
            in_features,
            out_features + in_features,
            bias=True,
            device=dev,
            dtype=dtype,
        )

        self.scale = nn.Parameter(torch.full((out_features,), scale_init, device=dev, dtype=dtype))

        self.bias_dynamic: Optional[nn.Linear] = (
            nn.Linear(in_features, out_features, bias=True, device=dev, dtype=dtype)
            if dynamic_bias
            else None
        )

        self._init_weights()

    def _init_weights(self) -> None:
        _small_init(self.hypernetwork)
        # Zero b-section only; a keeps gradient flowing
        _zero_b_section(self.hypernetwork, self.out_features)

        if self.bias_dynamic is not None:
            _small_init(self.bias_dynamic)
            _zero_out_last(self.bias_dynamic)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm for magnitude invariance
        h_in = F.rms_norm(x, (self.in_features,)) if self.normalize_input else x

        raw = self.hypernetwork(h_in)  # (..., O + I)

        a = _activate(raw[..., : self.out_features], self.factor_activation)
        b = _activate(raw[..., self.out_features :], self.factor_activation)

        # delta_W x = a(b·x) so no full matrix needed (avoid unnecessary memory allocation)
        dot = torch.sum(x * b, dim=-1, keepdim=True)  # (..., 1)
        adaptive = a * dot  # (..., O)

        out = self.linear_core(x) + adaptive * self.scale

        if self.bias_dynamic is not None:
            out = out + self.bias_dynamic(x)

        return out

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"act={self.factor_activation}, norm_input={self.normalize_input}"
        )


# RankRLiquidLN — rank‑R adaptive factors (best efficiency / expressivity)


class RankRLiquidLN(_FreezeMixin, nn.Module):
    """Input‑conditioned rank‑R update.

    Generates :math:`R` pairs of factors :math:`\\{a_r, b_r\\}`.  The dynamic
    update is :math:`\\Delta W = \\sum_{r=1}^R \\alpha_r \\, a_r \\otimes b_r`
    where :math:`\\alpha_r` = ``rank_scale[r]``.

    The hypernetwork can be a plain Linear (faster) or a 2‑layer MLP with
    SiLU (more expressive) by setting *nonlinear_hypernet*.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 1,
        bias: bool = True,
        dynamic_bias: bool = False,
        factor_activation: str = "norm",
        scale_init: float = 0.5,
        normalize_input: bool = False,
        nonlinear_hypernet: bool = False,
        hyper_hidden_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        dev = device if device is not None else DEVICE

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.factor_activation = factor_activation
        self.normalize_input = normalize_input

        self.linear_core = nn.Linear(in_features, out_features, bias=bias, device=dev, dtype=dtype)

        # Hypernetwork: linear or 2‑layer MLP
        hyper_out_dim = rank * (out_features + in_features)
        if nonlinear_hypernet:
            hidden = hyper_hidden_dim or max(in_features // 4, rank * 16)
            self.hypernetwork: nn.Module = nn.Sequential(
                nn.Linear(in_features, hidden, device=dev, dtype=dtype),
                nn.SiLU(),
                nn.Linear(hidden, hyper_out_dim, device=dev, dtype=dtype),
            )
        else:
            self.hypernetwork = nn.Linear(
                in_features, hyper_out_dim, bias=True, device=dev, dtype=dtype
            )

        self.scale = nn.Parameter(torch.full((out_features,), scale_init, device=dev, dtype=dtype))
        self.rank_scale = nn.Parameter(torch.full((rank,), 1.0, device=dev, dtype=dtype))

        self.bias_dynamic: Optional[nn.Linear] = (
            nn.Linear(in_features, out_features, bias=True, device=dev, dtype=dtype)
            if dynamic_bias
            else None
        )

        self._init_weights()

    def _init_weights(self) -> None:
        if isinstance(self.hypernetwork, nn.Sequential):
            for m in self.hypernetwork:
                _small_init(m)
        else:
            _small_init(self.hypernetwork)

        # Zero b-section; a-weights keep small init (avoid tanh saturation).
        _zero_b_section(self.hypernetwork, self.rank * self.out_features)

        if self.bias_dynamic is not None:
            _small_init(self.bias_dynamic)
            _zero_out_last(self.bias_dynamic)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_in = F.rms_norm(x, (self.in_features,)) if self.normalize_input else x
        raw = self.hypernetwork(h_in)

        split = self.rank * self.out_features
        a_raw = raw[..., :split].reshape(*x.shape[:-1], self.rank, self.out_features)
        b_raw = raw[..., split:].reshape(*x.shape[:-1], self.rank, self.in_features)

        a = _activate(a_raw, self.factor_activation)
        b = _activate(b_raw, self.factor_activation)

        # Dot products: b_r @ x -> (..., rank)
        dot = torch.matmul(b, x.unsqueeze(-1)).squeeze(-1)  # (..., rank)
        dot = dot * self.rank_scale

        # Weighted sum over rank
        adaptive = torch.matmul(dot.unsqueeze(-2), a).squeeze(-2)  # (..., O)

        out = self.linear_core(x) + adaptive * self.scale

        if self.bias_dynamic is not None:
            out = out + self.bias_dynamic(x)

        return out

    def extra_repr(self) -> str:
        nonlinear = isinstance(self.hypernetwork, nn.Sequential)
        return (
            f"in={self.in_features}, out={self.out_features}, rank={self.rank}, "
            f"act={self.factor_activation}, norm_input={self.normalize_input}, "
            f"nonlinear_hypernet={nonlinear}"
        )


# StableLiquidLN: nonlinear hypernet + factor normalisation + cond support


class StableLiquidLN(_FreezeMixin, nn.Module):
    """Production‑oriented variant with nonlinear hypernetwork and normalised factors.

    The hypernetwork is a 2‑layer MLP (SiLU).  Generated factors are
    activated (default L2‑norm) and summed over rank.  Unlike simpler variants,
    this module accepts an optional separate *cond* tensor for conditioning.

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
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        dev = device if device is not None else DEVICE

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.factor_activation = factor_activation
        self.normalize_input = normalize_input

        self.linear_core = nn.Linear(in_features, out_features, bias=bias, device=dev, dtype=dtype)

        # MLP hypernetwork
        hidden_dim = hyper_hidden_dim or max(in_features // 4, rank * 16)
        self.hypernetwork = nn.Sequential(
            nn.Linear(in_features, hidden_dim, device=dev, dtype=dtype),
            nn.SiLU(),
            nn.Linear(hidden_dim, rank * (out_features + in_features), device=dev, dtype=dtype),
        )

        # Scaling dial
        self.scale = nn.Parameter(torch.full((out_features,), scale_init, device=dev, dtype=dtype))
        self.rank_scale = nn.Parameter(torch.full((rank,), 1.0, device=dev, dtype=dtype))

        # Dynamic bias with MLP
        self.bias_dynamic: Optional[nn.Sequential] = (
            nn.Sequential(
                nn.Linear(in_features, hidden_dim, device=dev, dtype=dtype),
                nn.SiLU(),
                nn.Linear(hidden_dim, out_features, device=dev, dtype=dtype),
            )
            if dynamic_bias
            else None
        )

        self._init_weights()

    def _init_weights(self) -> None:
        # Small init avoids tanh saturation when ``factor_activation="tanh"``.
        for m in self.hypernetwork:
            _small_init(m)

        # Zero b-section; a-factors keep gradient flowing
        _zero_b_section(self.hypernetwork, self.rank * self.out_features)

        if self.bias_dynamic is not None:
            for m in self.bias_dynamic[:-1]:
                _small_init(m)
            _zero_out_last(self.bias_dynamic)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor ``(..., in_features)``.
            cond: Optional conditioning tensor.  When *None*, uses *x* itself.
                  The conditioning is fed into the hypernetwork (after optional
                  RMSNorm) while *x* always goes through the core linear path.

        Returns:
            Tensor ``(..., out_features)``.
        """
        cond = cond if cond is not None else x

        # RMSNorm for magnitude invariance
        h_in = F.rms_norm(cond, (self.in_features,)) if self.normalize_input else cond

        core_out = self.linear_core(x)

        raw = self.hypernetwork(h_in)

        split = self.rank * self.out_features
        a_raw = raw[..., :split].reshape(*h_in.shape[:-1], self.rank, self.out_features)
        b_raw = raw[..., split:].reshape(*h_in.shape[:-1], self.rank, self.in_features)

        a = _activate(a_raw, self.factor_activation)
        b = _activate(b_raw, self.factor_activation)

        dot = torch.matmul(b, x.unsqueeze(-1)).squeeze(-1)  # (..., rank)
        dot = dot * self.rank_scale

        adaptive = torch.matmul(dot.unsqueeze(-2), a).squeeze(-2)  # (..., O)

        out = core_out + adaptive * self.scale

        if self.bias_dynamic is not None:
            out = out + self.bias_dynamic(cond)

        return out

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, rank={self.rank}, "
            f"act={self.factor_activation}, norm_input={self.normalize_input}"
        )
