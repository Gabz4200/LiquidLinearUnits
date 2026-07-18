r"""Static (single-input / single-output) benchmark tasks.

Each task exposes a small, uniform contract so :mod:`train_io` can drive any of
them the same way::

    class IOTask:
        input_dim, out_dim : int
        loss_type          : "ce" | "mse"
        name               : str
        full_data(rng) -> (x, y)        # fixed dataset, (N, input_dim) / (N,) or (N, out_dim)
        metric(pred, y) -> float        # higher is better (accuracy, or -RMSE)
        sweep() -> list[dict]           # difficulty variations (for reports)

Tasks included:

* ``ModularArithmetic`` -- ``(a, b) -> (a+b)%p`` or ``(a*b)%p`` over one-hot
  inputs. The canonical grokking benchmark: train on a fraction of all pairs
  and watch memorisation -> generalisation as a phase transition.
* ``FourierTarget`` -- fit a sum of sinusoids at a controlled frequency. A
  direct probe of spectral bias (does the architecture learn high frequencies?).
* ``SparseParity`` -- target is the XOR of a ``k``-sparse subset of input bits.
  Plain MLPs trained with SGD need exponential samples to find the right
  feature combination, so this isolates compositional inductive bias.

All datasets are generated deterministically from a seed; ground-truth targets
mean we get exact error, not a proxy metric.
"""

from __future__ import annotations

import math

import numpy as np
import torch


class IOTask:
    input_dim: int = 0
    out_dim: int = 0
    loss_type: str = "ce"
    name: str = "task"

    def full_data(self, rng) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def metric(self, pred: torch.Tensor, y: torch.Tensor) -> float:
        if self.loss_type == "ce":
            return float((pred.argmax(dim=-1) == y).float().mean().item())
        # regression: report negative RMSE so "higher is better" holds uniformly.
        return float(-torch.sqrt(((pred - y) ** 2).mean()).item())

    def sweep(self) -> list[dict]:
        return [{}]


class ModularArithmetic(IOTask):
    """``(a, b) -> (a op b) mod p`` over concatenated one-hot inputs.

    ``full_data`` enumerates the complete ``p * p`` grid so the harness can hold
    out a fraction for the grokking test.
    """

    def __init__(self, p: int = 97, op: str = "add") -> None:
        if op not in ("add", "mul"):
            raise ValueError(f"op must be 'add' or 'mul', got {op!r}")
        self.p = p
        self.op = op
        self.input_dim = 2 * p
        self.out_dim = p
        self.loss_type = "ce"
        self.name = f"mod{p}_{op}"

    def full_data(self, rng) -> tuple[torch.Tensor, torch.Tensor]:
        N = self.p * self.p
        a = np.arange(self.p).repeat(self.p)
        b = np.tile(np.arange(self.p), self.p)
        y = (a + b) % self.p if self.op == "add" else (a * b) % self.p
        x = np.zeros((N, self.input_dim), dtype=np.float32)
        x[np.arange(N), a] = 1.0
        x[np.arange(N), self.p + b] = 1.0
        return torch.from_numpy(x), torch.from_numpy(y.astype(np.int64))

    def sweep(self) -> list[dict]:
        return [{"p": self.p, "op": self.op}]


class FourierTarget(IOTask):
    """Fit ``y = sum_k a_k * sin(w_k . x + phi_k)`` with controlled frequencies.

    ``w_k`` are drawn in ``[0.5, max_w]``, so raising ``max_w`` / ``n_freqs``
    makes the target progressively higher-frequency (harder for spectral bias).
    """

    def __init__(self, dim: int = 1, n_freqs: int = 3, max_w: float = 6.0,
                 n_samples: int = 4096, seed: int = 0) -> None:
        self.dim = dim
        self.n_freqs = n_freqs
        self.max_w = max_w
        self.n_samples = n_samples
        self.seed = seed
        self.input_dim = dim
        self.out_dim = 1
        self.loss_type = "mse"
        self.name = f"fourier_d{dim}_f{n_freqs}"

    def _coeffs(self):
        rng = np.random.default_rng(self.seed)
        ws = rng.uniform(0.5, self.max_w, size=(self.n_freqs, self.dim))
        amps = rng.uniform(0.5, 1.5, size=(self.n_freqs,))
        phases = rng.uniform(0.0, 2.0 * math.pi, size=(self.n_freqs,))
        return ws, amps, phases

    def full_data(self, rng) -> tuple[torch.Tensor, torch.Tensor]:
        ws, amps, phases = self._coeffs()
        gen = np.random.default_rng(self.seed + 1)
        x = gen.uniform(-1.0, 1.0, size=(self.n_samples, self.dim)).astype(np.float32)
        y = np.zeros((self.n_samples, 1), dtype=np.float32)
        for k in range(self.n_freqs):
            y[:, 0] += amps[k] * np.sin(ws[k] @ x.T + phases[k])
        return torch.from_numpy(x), torch.from_numpy(y)

    def sweep(self) -> list[dict]:
        return [{"dim": self.dim, "n_freqs": self.n_freqs, "max_w": self.max_w}]


class SparseParity(IOTask):
    """Target = parity (XOR) of a fixed ``k``-sparse subset of ``dim`` input bits."""

    def __init__(self, dim: int = 20, k: int = 4, n_samples: int = 4096,
                 seed: int = 0) -> None:
        if k > dim:
            raise ValueError("k must be <= dim")
        self.dim = dim
        self.k = k
        self.n_samples = n_samples
        self.seed = seed
        self.input_dim = dim
        self.out_dim = 2
        self.loss_type = "ce"
        self.name = f"parity_d{dim}_k{k}"
        rng = np.random.default_rng(seed)
        self.subset = np.sort(rng.choice(dim, size=k, replace=False))

    def full_data(self, rng) -> tuple[torch.Tensor, torch.Tensor]:
        gen = np.random.default_rng(self.seed + 1)
        x = gen.integers(0, 2, size=(self.n_samples, self.dim)).astype(np.float32)
        y = (x[:, self.subset].sum(axis=1) % 2).astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)

    def sweep(self) -> list[dict]:
        return [{"dim": self.dim, "k": self.k}]


# Factories take the resolved config so CLI flags (--p, --op, ...) flow through.
TASK_FACTORIES = {
    "mod_add": lambda c: ModularArithmetic(p=c["p"], op="add"),
    "mod_mul": lambda c: ModularArithmetic(p=c["p"], op="mul"),
    "fourier": lambda c: FourierTarget(dim=c["dim"], n_freqs=c["n_freqs"], max_w=c["max_w"]),
    "parity": lambda c: SparseParity(dim=c.get("parity_dim", 20), k=c["k"]),
}


def make_task(name: str, cfg: dict) -> IOTask:
    if name not in TASK_FACTORIES:
        raise KeyError(f"unknown IO task {name!r}; choose from {sorted(TASK_FACTORIES)}")
    return TASK_FACTORIES[name](cfg)


__all__ = [
    "IOTask",
    "ModularArithmetic",
    "FourierTarget",
    "SparseParity",
    "TASK_FACTORIES",
    "make_task",
]
