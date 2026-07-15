# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Copyright Lightning AI. Licensed under the Apache License 2.0,
# see LICENSE file at https://github.com/Lightning-AI/litgpt/blob/main/LICENSE

import os
import sys
import warnings

import torch

__all__ = ["GatedDeltaNet2"]


def _warn_gpu_fallback(exc: Exception) -> None:
    """Emit an impossible-to-miss warning when the GPU backend fails to load.

    Only called when CUDA is available (so the GPU path was actually intended)
    but ``fla``/Triton could not be imported or constructed. The banner is
    printed straight to stderr so it survives ``warnings`` filters.
    """
    reason = f"{type(exc).__name__}: {exc}"
    banner = "\n".join(
        [
            "=" * 88,
            "!!! GDN-2 GPU BACKEND UNAVAILABLE -- FALLING BACK TO CPU IMPLEMENTATION !!!",
            "=" * 88,
            "CUDA is available, but the GPU-optimized GDN-2 backend (fla / Triton kernels)",
            f"could NOT be loaded. Reason: {reason}",
            "GatedDeltaNet2(...) is now using the pure-PyTorch CPU implementation instead.",
            "Set LLU_FORCE_CPU=1 to silence this fallback, or fix the fla/Triton install",
            "so GDN-2 runs on the Triton-optimized kernels as intended.",
            "=" * 88,
        ]
    )
    print(banner, file=sys.stderr, flush=True)
    warnings.warn(
        f"GDN-2 GPU backend unavailable ({reason}); fell back to CPU implementation.",
        stacklevel=2,
    )


def _make_gated_delta_net2(*args: object, **kwargs: object) -> object:
    """Construct a GDN-2 layer, selecting the GPU backend when available.

    The GPU-optimized ``fla``/Triton implementation is used only when all of:
      * ``LLU_FORCE_CPU`` is not set to ``1``, and
      * CUDA is available, and
      * the ``fla`` package imports and constructs cleanly.

    If CUDA is present but the GPU backend fails to load, a loud warning is
    emitted and the pure-PyTorch CPU implementation is used. Any other path
    (no CUDA, or ``LLU_FORCE_CPU=1``) is silent and uses the CPU version.
    """
    if os.environ.get("LLU_FORCE_CPU") != "1" and torch.cuda.is_available():
        try:
            from .gdn2_gpu import GatedDeltaNet2GPU

            return GatedDeltaNet2GPU(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 -- fall back to CPU on any GPU failure
            _warn_gpu_fallback(exc)
    from .gdn2 import GatedDeltaNet2 as _CpuGatedDeltaNet2

    return _CpuGatedDeltaNet2(*args, **kwargs)


# Public constructor: GPU backend when possible, CPU otherwise.
GatedDeltaNet2 = _make_gated_delta_net2
