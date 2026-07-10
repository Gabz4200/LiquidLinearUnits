r"""Utility functions and mixins for Liquid Linear Units."""

import math
from typing import Optional, Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

_VALID_ACTIVATIONS = frozenset({"tanh", "norm", "rmsnorm", "none"})


def _activate(t: torch.Tensor, mode: str, eps: float = 1e-6) -> torch.Tensor:
    r"""_activate(t, mode, eps=1e-6) -> Tensor

    Apply one of the supported factor activations or normalisations.

    Args:
        t (Tensor): input tensor.
        mode (str): activation mode.  One of ``"tanh"``, ``"norm"``,
            ``"rmsnorm"``, or ``"none"``.
        eps (float): epsilon for numerical stability in ``"norm"`` and
            ``"rmsnorm"``.  Default: ``1e-6``.

    Returns:
        Tensor: activated tensor of the same shape as *t*.
    """
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
    r"""_small_init(module, gain=0.02) -> None

    Apply Xavier uniform initialisation with a small gain recursively to all
    linear layers in the module. Biases (if present) are zeroed.

    Args:
        module (nn.Module): a module.
        gain (float): scaling gain.  Default: ``0.02``.
    """
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def _last_linear(module: nn.Module) -> Optional[nn.Linear]:
    r"""_last_linear(module) -> nn.Linear or None

    Return the last :class:`~nn.Linear` in a module or :class:`~nn.Sequential`.

    Args:
        module (nn.Module): a module, possibly :class:`~nn.Sequential`.

    Returns:
        nn.Linear or None: the last linear layer, or ``None`` if the
        module is an empty :class:`~nn.Sequential` or does not end with
        a linear layer.
    """
    if isinstance(module, nn.Sequential):
        if len(module) == 0:
            return None
        module = module[-1]
    return module if isinstance(module, nn.Linear) else None


def _zero_out_last(module: nn.Module) -> None:
    r"""_zero_out_last(module) -> None

    Zero the weights and bias of the last :class:`~nn.Linear` in a chain.

    Respects :class:`~nn.Sequential` -- only the final layer is zeroed so that
    earlier layers can keep a standard init and receive gradient from step 1.

    Args:
        module (nn.Module): a module or :class:`~nn.Sequential` whose last
            linear layer is zeroed.
    """
    _zero_b_section(module, 0)


def _zero_b_section(module: nn.Module, b_start: int) -> None:
    r"""_zero_b_section(module, b_start) -> None

    Zero the **b-section** rows (``[b_start:]``) of the last linear layer.

    The a-section (``[:b_start]``) keeps its previous init.  This partial
    zero-init ensures that at step 1:

    * the adaptive output is zero (because b-factors are zero), *yet*
    * gradients flow into the hypernetwork (because a-factors are non-zero).

    The a-section (``[:b_start]``) should have non-zero weights (from a prior
    init call such as :func:`_small_init`) so gradient flows through the
    hypernetwork.  The bias is zeroed independently here.

    Args:
        module (nn.Module): a module or :class:`~nn.Sequential` whose last
            linear layer is modified.
        b_start (int): row index where the b-section begins.
    """
    last = _last_linear(module)
    if last is not None:
        with torch.no_grad():
            last.weight.data[b_start:].zero_()
            # Zero entire bias (shared across a/b, a-section still fires via W_a@h).
            if last.bias is not None:
                last.bias.data.zero_()


def _hyperfan_init(
    module: nn.Module,
    in_features: int,
    out_features: int,
    mode: str = "fan_in",
    nonlinearity: str = "linear",
    var_e: float = 1.0,
    uniform: bool = True,
    rank: Optional[int] = None,
) -> None:
    r"""_hyperfan_init(module, in_features, out_features, mode="fan_in", nonlinearity="linear", var_e=1.0, uniform=True, rank=None) -> None

    Apply Hyperfan initialisation to the last linear layer of a module.

    Assumes Case 1 of the Hyperfan paper: the hypernetwork generates the weights
    but not the biases of the mainnet.  Biases of the hypernetwork output
    layer are zeroed.

    For fullweight hypernetworks (rank=None), initializes weights with variance:
    - fan_in: gain^2 / (in_features * d_hyper * var_e)
    - fan_out: gain^2 / (out_features * d_hyper * var_e)

    For factorized low-rank hypernetworks (rank is not None), initializes
    weights generating factors with variance:
    - fan_in: gain / (d_hyper * var_e * sqrt(rank * in_features))
    - fan_out: gain / (d_hyper * var_e * sqrt(rank * out_features))

    Args:
        module (nn.Module): the hypernetwork module or container.
        in_features (int): mainnet input dimension.
        out_features (int): mainnet output dimension.
        mode (str): initialization mode, either ``"fan_in"`` or ``"fan_out"``.
            Default: ``"fan_in"``.
        nonlinearity (str): nonlinearity of the mainnet, used to calculate gain.
            Default: ``"linear"``.
        var_e (float): variance of the conditioning vector e.  Default: ``1.0``.
        uniform (bool): if ``True``, use uniform distribution; otherwise normal.
            Default: ``True``.
        rank (int, optional): rank of the factorized update.  If ``None``,
            assumes fullweight generator.
    """
    last = _last_linear(module)
    if isinstance(module, nn.Sequential):
        for m in module[:-1]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    if last is None:
        return

    if mode not in {"fan_in", "fan_out"}:
        raise ValueError(f"Unknown mode '{mode}'; expected 'fan_in' or 'fan_out'")

    d_hyper = last.in_features
    gain = nn.init.calculate_gain(cast(Any, nonlinearity))

    base = in_features if mode == "fan_in" else out_features
    if rank is None:
        den = base * d_hyper * var_e
        var = (gain**2) / den
    else:
        den = d_hyper * var_e * math.sqrt(rank * base)
        var = gain / den

    std = math.sqrt(var)

    with torch.no_grad():
        if uniform:
            bound = math.sqrt(3.0) * std
            nn.init.uniform_(last.weight, a=-bound, b=bound)
        else:
            nn.init.normal_(last.weight, mean=0.0, std=std)

        if last.bias is not None:
            nn.init.zeros_(last.bias)


def _init_hypernetwork(
    hypernetwork: nn.Module,
    init_method: str,
    in_features: int,
    out_features: int,
    rank: Optional[int] = None,
) -> None:
    r"""_init_hypernetwork(hypernetwork, init_method, in_features, out_features, rank=None) -> None

    Initialise the hypernetwork using the specified method.
    """
    if init_method == "hyperfan_in":
        _hyperfan_init(hypernetwork, in_features, out_features, mode="fan_in", rank=rank)
    elif init_method == "hyperfan_out":
        _hyperfan_init(hypernetwork, in_features, out_features, mode="fan_out", rank=rank)
    elif init_method == "xavier":
        _small_init(hypernetwork, gain=1.0)
    else:  # "small"
        _small_init(hypernetwork)


def _ensure_buffer_shape(buffer: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    r"""_ensure_buffer_shape(buffer, target) -> Tensor

    Return a zeroed tensor with the same shape, device, and dtype as target
    if buffer does not match, otherwise return buffer.
    """
    if (
        buffer.shape != target.shape
        or buffer.device != target.device
        or buffer.dtype != target.dtype
    ):
        return torch.zeros_like(target)
    return buffer


class _FreezeMixin:
    r"""Mixin providing ``freeze_core`` and ``freeze_hypernetwork``.

    Subclasses MUST have a :attr:`linear_core` attribute (a module whose
    parameters delimit the "core" from the "adaptive path").
    """

    linear_core: nn.Linear

    def freeze_core(self) -> None:
        r"""freeze_core() -> None

        Freeze all parameters of the core linear layer.
        Only the adaptive path (hypernetwork, scale, and dynamic bias)
        remains trainable.
        """
        for p in self.linear_core.parameters():
            p.requires_grad = False

    def freeze_hypernetwork(self) -> None:
        r"""freeze_hypernetwork() -> None

        Freeze all parameters *except* those of the core linear layer.
        Only :attr:`linear_core` remains trainable.
        """
        core_ids = {id(p) for p in self.linear_core.parameters()}
        assert isinstance(self, nn.Module)
        for p in self.parameters():
            if id(p) not in core_ids:
                p.requires_grad = False
