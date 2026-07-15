r"""
GPU-optimized GDN-2 backend.

This module wraps ``fla``'s Triton-kernel ``GatedDeltaNet2`` -- the canonical
NVlabs GatedDeltaNet-2 implementation. It exposes the *same* constructor and
``forward`` contract as the pure-PyTorch CPU implementation in ``gdn2.py``:

    GatedDeltaNet2(hidden_size, expand_v, head_dim, num_heads, num_v_heads,
                   mode, use_short_conv, allow_neg_eigval, conv_size,
                   conv_bias, layer_idx, norm_eps)
    forward(hidden_states, attention_mask, past_key_values, use_cache)
        -> (output, None, past_key_values)

Crucially, ``fla``/``triton`` are imported *lazily* inside the callable, so
importing this module never requires a GPU or those packages to be installed.
The package factory (``__init__.py``) only invokes this on a machine where
CUDA and ``fla`` are actually available; otherwise it falls back to the CPU
implementation, leaving that path completely untouched.
"""

from __future__ import annotations

from typing import Any, Literal


def GatedDeltaNet2GPU(
    hidden_size: int = 2048,
    expand_v: float = 1,
    head_dim: int = 128,
    num_heads: int = 16,
    num_v_heads: int | None = None,
    mode: Literal["chunk", "fused_recurrent"] = "chunk",
    use_short_conv: bool = True,
    allow_neg_eigval: bool = False,
    conv_size: int = 4,
    conv_bias: bool = False,
    layer_idx: int | None = None,
    norm_eps: float = 1e-5,
    **kwargs: Any,
) -> "object":
    """Construct a GPU-optimized GDN-2 layer using ``fla``'s Triton kernels.

    The ``fla`` dependency is imported lazily, so this callable is safe to
    reference on CPU-only machines. Raises ``ImportError`` if ``fla`` (and its
    Triton kernels) are not installed.

    Args match the CPU implementation in ``gdn2.py``; any extra keyword
    arguments are forwarded to ``fla.layers.GatedDeltaNet2``.
    """
    from fla.layers import GatedDeltaNet2 as _FlaGatedDeltaNet2

    return _FlaGatedDeltaNet2(
        hidden_size=hidden_size,
        expand_v=expand_v,
        head_dim=head_dim,
        num_heads=num_heads,
        num_v_heads=num_v_heads,
        mode=mode,
        use_short_conv=use_short_conv,
        allow_neg_eigval=allow_neg_eigval,
        conv_size=conv_size,
        conv_bias=conv_bias,
        layer_idx=layer_idx,
        norm_eps=norm_eps,
        **kwargs,
    )
