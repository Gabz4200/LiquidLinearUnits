r"""Base classes for Liquid Linear Units."""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from .utils import (
    DEVICE,
    _activate,
    _small_init,
    _zero_out_last,
    _zero_b_section,
    _init_hypernetwork,
    _ensure_buffer_shape,
    _FreezeMixin,
)


class BaseLLU(_FreezeMixin, nn.Module):
    r"""Abstract base class for all Liquid Linear Units.

    Handles initialization of features, device/dtype routing, the core linear
    layer, and the per-channel adaptive scale. Also provides helpers for dynamic
    bias initialization/application and low-rank factor operations.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        scale_init: float = 0.9,
        factor_activation: str = "norm",
        init_method: str = "hyperfan_in",
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.factor_activation = factor_activation
        self.init_method = init_method

        dev = device if device is not None else DEVICE
        self.linear_core = nn.Linear(
            in_features, out_features, bias=bias, device=dev, dtype=dtype
        )
        nn.init.xavier_uniform_(self.linear_core.weight)
        if self.linear_core.bias is not None:
            nn.init.zeros_(self.linear_core.bias)
        self.bias_dynamic: Optional[nn.Module] = None
        self.U: Optional[nn.Parameter] = None
        self.V: Optional[nn.Parameter] = None
        # Declared here (not just on subclasses) so the SVD- and
        # parameterization-aware helpers can reference them without risking an
        # AttributeError before a subclass assigns the concrete value.
        self.rank: Optional[int] = None
        self.parameterization: Optional[str] = None

    def _init_bias_dynamic(self) -> None:
        if self.bias_dynamic is not None:
            _small_init(self.bias_dynamic)
            _zero_out_last(self.bias_dynamic)

    def _init_low_rank_adaptive(
        self,
        target: nn.Module,
        b_start: int,
        rank: Optional[int] = None,
    ) -> None:
        r"""_init_low_rank_adaptive(target, b_start, rank=None) -> None

        Initialise a low-rank hypernetwork / projection module so the adaptive
        path contributes zero at step 1 while gradients still flow through the
        a-factors.  Shared by every rank-based LLU variant; the GDN variants
        prepend the GDN-2 internal init before calling this.

        Args:
            target (nn.Module): the hypernetwork or projection module whose
                last linear layer is initialised.
            b_start (int): row index where the b-section begins
                (``rank * out_features``).
            rank (int, optional): rank passed to the hyperfan variance scaling.
        """
        _init_hypernetwork(
            target, self.init_method, self.in_features, self.out_features, rank=rank
        )
        _zero_b_section(target, b_start)
        self._init_bias_dynamic()

    def _apply_dynamic_bias(self, out: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if self.bias_dynamic is not None:
            out = out + self.bias_dynamic(cond)
        return out

    def _compute_svd_adaptive(
        self,
        x: torch.Tensor,
        g: torch.Tensor,
    ) -> torch.Tensor:
        r"""_compute_svd_adaptive(x, g) -> Tensor

        SVD-parameterized adaptive path: ``((x @ V) * g) @ U``.
        """
        v = torch.matmul(x, self.V)
        modulated = v * g
        return torch.matmul(modulated, self.U)

    def _create_svd_factors(
        self,
        device: Optional[torch.device],
        dtype: torch.dtype,
    ) -> None:
        r"""_create_svd_factors(device, dtype) -> None

        Create the static U/V factors for SVD parameterization (uninitialised;
        :meth:`_init_svd_projection` fills them with xavier)."""
        self.U = nn.Parameter(
            torch.empty(self.rank, self.out_features, device=device, dtype=dtype)
        )
        self.V = nn.Parameter(
            torch.empty(self.in_features, self.rank, device=device, dtype=dtype)
        )

    def _init_svd_projection(self, proj: nn.Module) -> None:
        r"""_init_svd_projection(proj) -> None

        Initialise the SVD projection module and U/V: hyperfan-init *proj*, zero
        its last layer (zero adaptive path at step 1), then xavier-init U/V and
        the optional dynamic bias."""
        _init_hypernetwork(
            proj, self.init_method, self.in_features, self.out_features, rank=self.rank
        )
        _zero_out_last(proj)
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)
        self._init_bias_dynamic()

    def _compute_low_rank_adaptive(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        dot = torch.matmul(b, x.unsqueeze(-1)).squeeze(-1)  # (..., rank)
        adaptive = torch.matmul(dot.unsqueeze(-2), a).squeeze(-2)  # (..., O)
        return adaptive


class BaseMomentumLLU(BaseLLU):
    r"""Base class for Liquid Linear Units with momentum buffers."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        decay_rate: float = 0.4,
        rank: int = 4,
        bias: bool = True,
        scale_init: float = 0.01,
        factor_activation: str = "norm",
        init_method: str = "hyperfan_in",
        learnable_decay_rate: bool = False,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
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
        self.rank = rank
        self.decay_rate = nn.Parameter(
            torch.full((), decay_rate, device=self.linear_core.weight.device)
        )
        self.decay_rate.requires_grad = learnable_decay_rate

    def set_decay_rate_learnable(self, learnable: bool = True) -> None:
        self.decay_rate.requires_grad = learnable

    @property
    def local_decay_rate(self) -> torch.Tensor:
        return torch.sigmoid(self.decay_rate)

    def _update_shared_momentum(
        self,
        a_new: torch.Tensor,
        b_new: torch.Tensor,
        detach: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if detach:
            self.a_raw = self.a_raw.detach()
            self.b_raw = self.b_raw.detach()

        dims = tuple(range(a_new.ndim - 2))
        if dims:
            a_mean = a_new.mean(dim=dims)
            b_mean = b_new.mean(dim=dims)
        else:
            a_mean = a_new
            b_mean = b_new

        self.a_raw = self.a_raw * self.local_decay_rate + a_mean
        self.b_raw = self.b_raw * self.local_decay_rate + b_mean

        a = _activate(self.a_raw.expand_as(a_new), self.factor_activation)
        b = _activate(self.b_raw.expand_as(b_new), self.factor_activation)
        return a, b

    def _update_g_shared_momentum(
        self,
        g_new: torch.Tensor,
        detach: bool = False,
    ) -> torch.Tensor:
        if detach:
            self.g_raw = self.g_raw.detach()

        dims = tuple(range(g_new.ndim - 1))
        if dims:
            g_mean = g_new.mean(dim=dims)
        else:
            g_mean = g_new

        self.g_raw = self.g_raw * self.local_decay_rate + g_mean
        g = _activate(self.g_raw.expand_as(g_new), self.factor_activation)
        return g

    def _update_batch_momentum(
        self,
        a_new: torch.Tensor,
        b_new: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.a_raw = _ensure_buffer_shape(self.a_raw, a_new)
        self.b_raw = _ensure_buffer_shape(self.b_raw, b_new)

        self.a_raw = self.a_raw * self.local_decay_rate + a_new
        self.b_raw = self.b_raw * self.local_decay_rate + b_new

        a = _activate(self.a_raw, self.factor_activation)
        b = _activate(self.b_raw, self.factor_activation)
        return a, b

    def _update_g_batch_momentum(
        self,
        g_new: torch.Tensor,
    ) -> torch.Tensor:
        self.g_raw = _ensure_buffer_shape(self.g_raw, g_new)
        self.g_raw = self.g_raw * self.local_decay_rate + g_new
        g = _activate(self.g_raw, self.factor_activation)
        return g

    def _register_momentum_buffers(
        self,
        device: Optional[torch.device],
        dtype: torch.dtype,
        batch: bool = False,
    ) -> None:
        r"""_register_momentum_buffers(device, dtype, batch=False) -> None

        Register the momentum factor buffers and SVD factors for the active
        parameterization.  In ``"lora"`` mode the shared/batch ``a_raw``/``b_raw``
        buffers are created (U/V stay ``None``); in ``"svd"`` mode ``g_raw`` is
        created and the static U/V factors are built.
        """
        prefix: tuple[int, ...] = (1,) if batch else ()
        if self.parameterization == "lora":
            self.register_buffer(
                "a_raw",
                torch.zeros(*prefix, self.rank, self.out_features, device=device, dtype=dtype),
                persistent=True,
            )
            self.register_buffer(
                "b_raw",
                torch.zeros(*prefix, self.rank, self.in_features, device=device, dtype=dtype),
                persistent=True,
            )
            self.g_raw = None
        else:
            self.a_raw = None
            self.b_raw = None
            self.register_buffer(
                "g_raw",
                torch.zeros(*prefix, self.rank, device=device, dtype=dtype),
                persistent=True,
            )
            self._create_svd_factors(device, dtype)
