# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Copyright Lightning AI. Licensed under the Apache License 2.0,
# see LICENSE file at https://github.com/Lightning-AI/litgpt/blob/main/LICENSE

import os

import torch

__all__ = ["GatedDeltaNet2"]


def _make_gated_delta_net2(*args: object, **kwargs: object) -> object:
    """Construct a GDN-2 layer, selecting the GPU backend when available.

    The GPU-optimized ``fla``/Triton implementation is used only when all of:
      * ``LLU_FORCE_CPU`` is not set to ``1``, and
      * CUDA is available, and
      * the ``fla`` package imports cleanly.

    Any failure (missing CUDA, missing ``fla``/Triton, or a construction
    error) falls back to the pure-PyTorch CPU implementation. This guarantees
    the CPU version is never compromised and the package is always importable
    on CPU-only machines.
    """
    if os.environ.get("LLU_FORCE_CPU") != "1" and torch.cuda.is_available():
        try:
            from .gdn2_gpu import GatedDeltaNet2GPU

            return GatedDeltaNet2GPU(*args, **kwargs)
        except Exception:
            pass
    from .gdn2 import GatedDeltaNet2 as _CpuGatedDeltaNet2

    return _CpuGatedDeltaNet2(*args, **kwargs)


# Public constructor: GPU backend when possible, CPU otherwise.
GatedDeltaNet2 = _make_gated_delta_net2
