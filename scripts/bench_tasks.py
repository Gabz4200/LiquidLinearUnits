r"""Synthetic benchmark tasks for comparing Liquid Linear Unit architectures.

Every task is a generator that yields one batch as ``(x, y, mask)`` torch
tensors:

* ``x``     : ``(B, T, token_dim)`` float32 input stream (one token per step).
* ``y``     : target. For ``"mse"`` tasks ``(B, T, out_dim)`` float32; for
              ``"ce"`` tasks ``(B, T)`` long class indices; for ``"bce"`` tasks
              ``(B, T, 1)`` float32 in ``{0, 1}``.
* ``mask``  : ``(B, T)`` bool -- which positions contribute to the loss
              (usually only the query / final position).

The downstream model emits ``(B, T, out_dim)`` logits; :func:`loss_fn` and
:func:`compute_metric` reduce over the masked positions. ``token_dim`` /
``out_dim`` tell the training script how to size the model.

Tasks are tunable in sequence length / key dimension / correlation so the CPU
runtime can be controlled directly. Sweepable tasks expose :meth:`sweep`.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Shared loss / metric helpers
# ---------------------------------------------------------------------------

def loss_fn(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, loss_type: str) -> torch.Tensor:
    pm = pred[mask]
    ym = y[mask]
    if loss_type == "mse":
        return torch.nn.functional.mse_loss(pm, ym)
    if loss_type == "ce":
        return torch.nn.functional.cross_entropy(pm, ym.long())
    if loss_type == "bce":
        return torch.nn.functional.binary_cross_entropy_with_logits(pm, ym)
    raise ValueError(f"unknown loss_type {loss_type}")


def compute_metric(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, loss_type: str) -> dict:
    pm = pred[mask].detach()
    ym = y[mask].detach()
    if loss_type == "mse":
        mse = ((pm - ym) ** 2).mean().item()
        per = (pm - ym).pow(2).sum(-1).sqrt()
        scale = ym.pow(2).sum(-1).sqrt() + 1e-8
        rel = per / scale
        succ = (rel < 0.5).float().mean().item()
        return {"mse": mse, "success_rate": succ}
    if loss_type == "ce":
        pred_cls = pm.argmax(-1)
        acc = (pred_cls == ym.long()).float().mean().item()
        return {"acc": acc}
    if loss_type == "bce":
        acc = ((pm.sigmoid() > 0.5).float() == ym.float()).float().mean().item()
        return {"acc": acc}
    return {}


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class SyntheticTask:
    loss_type = "mse"
    token_dim = 8
    out_dim = 8

    def generate(self, B: int, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def sweep(self) -> list[dict]:
        """List of param-override dicts to run as separate sub-runs."""
        return [{}]

    def ceiling(self, x, y, mask) -> dict:
        return {}


def _sample_keys(rng: np.random.Generator, n: int, kd: int) -> np.ndarray:
    """Orthogonal-ish unit keys."""
    k = rng.standard_normal((n, kd))
    k /= (np.linalg.norm(k, axis=1, keepdims=True) + 1e-8)
    return k


# ---------------------------------------------------------------------------
# 1. Overwrite associative recall
# ---------------------------------------------------------------------------

class OverwriteRecall(SyntheticTask):
    """Present ``n_pairs`` (key, value) writes then a query for one key.

    If a key repeats, the *latest* write must win (overwrite). The query is the
    last token; only it is masked for the loss.
    """

    def __init__(self, n_pairs: int = 8, kd: int = 8, vd: int = 8,
                 value_scale: float = 1.0, overwrite_prob: float = 0.5):
        self.n_pairs = n_pairs
        self.kd = kd
        self.vd = vd
        self.value_scale = value_scale
        self.overwrite_prob = overwrite_prob
        self.token_dim = kd + vd + 1
        self.out_dim = vd
        self.loss_type = "mse"

    def generate(self, B, rng):
        n, kd, vd = self.n_pairs, self.kd, self.vd
        T = n + 1
        x = np.zeros((B, T, self.token_dim), dtype=np.float32)
        y = np.zeros((B, T, vd), dtype=np.float32)
        mask = np.zeros((B, T), dtype=bool)
        for b in range(B):
            keys = _sample_keys(rng, n, kd)
            if rng.random() < self.overwrite_prob and n >= 2:
                dup = rng.integers(n)
                src = rng.integers(n)
                keys[dup] = keys[src]
            vals = rng.standard_normal((n, vd)).astype(np.float32) * self.value_scale
            for i in range(n):
                x[b, i, :kd] = keys[i]
                x[b, i, kd:kd + vd] = vals[i]
                x[b, i, kd + vd] = 1.0  # write marker
                y[b, i] = vals[i]
            qi = rng.integers(n)
            qkey = keys[qi]
            last_val = vals[qi].copy()
            for i in range(n):
                if np.allclose(keys[i], qkey):
                    last_val = vals[i]
            x[b, n, :kd] = qkey
            x[b, n, kd + vd] = 0.0  # query marker
            y[b, n] = last_val
            mask[b, n] = True
        return (torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask))


class CorrelatedKeyRecall(OverwriteRecall):
    """Overwrite recall where every key is correlated with a shared base key.

    Correlation ``rho`` controls interference: ``rho=0`` => orthogonal keys,
    ``rho->1`` => near-collinear keys. Measures interference-under-write, the
    core delta-rule vs momentum distinction.
    """

    def __init__(self, rho: float = 0.0, **kw):
        super().__init__(**kw)
        self.rho = rho

    def generate(self, B, rng):
        n, kd, vd = self.n_pairs, self.kd, self.vd
        T = n + 1
        x = np.zeros((B, T, self.token_dim), dtype=np.float32)
        y = np.zeros((B, T, vd), dtype=np.float32)
        mask = np.zeros((B, T), dtype=bool)
        r = math.sqrt(max(1e-6, 1.0 - self.rho ** 2))
        for b in range(B):
            base = rng.standard_normal(kd)
            base /= (np.linalg.norm(base) + 1e-8)
            keys = np.array([base * self.rho + r * rng.standard_normal(kd) for _ in range(n)])
            keys /= (np.linalg.norm(keys, axis=1, keepdims=True) + 1e-8)
            if rng.random() < self.overwrite_prob and n >= 2:
                dup = rng.integers(n)
                src = rng.integers(n)
                keys[dup] = keys[src]
            vals = rng.standard_normal((n, vd)).astype(np.float32) * self.value_scale
            for i in range(n):
                x[b, i, :kd] = keys[i]
                x[b, i, kd:kd + vd] = vals[i]
                x[b, i, kd + vd] = 1.0
                y[b, i] = vals[i]
            qi = rng.integers(n)
            qkey = keys[qi]
            last_val = vals[qi].copy()
            for i in range(n):
                if np.allclose(keys[i], qkey):
                    last_val = vals[i]
            x[b, n, :kd] = qkey
            x[b, n, kd + vd] = 0.0
            y[b, n] = last_val
            mask[b, n] = True
        return (torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask))

    def sweep(self):
        return [{"rho": v} for v in (0.0, 0.5, 0.9, 0.99)]


class CapacitySweep(OverwriteRecall):
    """Capacity sweep: vary number of stored KV pairs (8 -> 256)."""

    def sweep(self):
        return [{"n_pairs": v} for v in (8, 16, 32, 64, 128)]


# ---------------------------------------------------------------------------
# 4. Needle-in-a-haystack (distance sweep)
# ---------------------------------------------------------------------------

class NeedleInHaystack(SyntheticTask):
    """One relevant (key, value) write at position 0, then ``distractors``
    distractor writes, then a query for the first key.

    Distant relevant write isolates decay-by-time (momentum) vs decay-by-
    relevance (GDN-2): GDN-2 should recall the relevant key regardless of how
    many distractors sit between the write and the query.
    """

    def __init__(self, distractors: int = 8, kd: int = 8, vd: int = 8, value_scale: float = 1.0):
        self.distractors = distractors
        self.kd = kd
        self.vd = vd
        self.value_scale = value_scale
        self.token_dim = kd + vd + 1
        self.out_dim = vd
        self.loss_type = "mse"

    def generate(self, B, rng):
        d, kd, vd = self.distractors, self.kd, self.vd
        T = d + 2  # relevant write + distractors + query
        x = np.zeros((B, T, self.token_dim), dtype=np.float32)
        y = np.zeros((B, T, vd), dtype=np.float32)
        mask = np.zeros((B, T), dtype=bool)
        for b in range(B):
            rkey = _sample_keys(rng, 1, kd)[0]
            rval = rng.standard_normal(vd).astype(np.float32) * self.value_scale
            dkeys = _sample_keys(rng, d, kd)
            dvals = rng.standard_normal((d, vd)).astype(np.float32) * self.value_scale
            # position 0: relevant write
            x[b, 0, :kd] = rkey
            x[b, 0, kd:kd + vd] = rval
            x[b, 0, kd + vd] = 1.0
            y[b, 0] = rval
            # positions 1..d: distractors
            for i in range(d):
                x[b, 1 + i, :kd] = dkeys[i]
                x[b, 1 + i, kd:kd + vd] = dvals[i]
                x[b, 1 + i, kd + vd] = 1.0
                y[b, 1 + i] = dvals[i]
            # position T-1: query for relevant key
            x[b, T - 1, :kd] = rkey
            x[b, T - 1, kd + vd] = 0.0
            y[b, T - 1] = rval
            mask[b, T - 1] = True
        return (torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask))

    def sweep(self):
        return [{"distractors": v} for v in (1, 4, 8, 16, 32, 64)]


# ---------------------------------------------------------------------------
# 5. XOR-style non-linear key -> value mapping
# ---------------------------------------------------------------------------

class XORNonlinear(SyntheticTask):
    """Each token is an ``n_bits`` binary key; the value is the XOR of two
    designated bits (a non-linearly-separable function). Confirms MLP/GDN-2
    beat a plain linear map on non-linear structure.
    """

    def __init__(self, n_bits: int = 8, bit_a: int = 0, bit_b: int = 1):
        self.n_bits = n_bits
        self.bit_a = bit_a
        self.bit_b = bit_b
        self.token_dim = n_bits
        self.out_dim = 1
        self.loss_type = "bce"

    def generate(self, B, rng):
        nb = self.n_bits
        T = 1
        keys = (rng.random((B, T, nb)) > 0.5).astype(np.float32)
        val = (keys[..., self.bit_a].astype(np.int64) ^ keys[..., self.bit_b].astype(np.int64)).astype(np.float32)
        x = torch.from_numpy(keys)
        y = torch.from_numpy(val[..., None])  # (B, 1, 1)
        mask = torch.ones(B, T, dtype=torch.bool)
        return x, y, mask


# ---------------------------------------------------------------------------
# 6. In-context linear regression
# ---------------------------------------------------------------------------

class InContextLinearRegression(SyntheticTask):
    """Per sequence, sample a fresh linear map ``w``; present ``n_pairs``
    (x_i, y_i=w.x_i+noise) writes, then query x_q for y_q=w.x_q. The input at a
    write carries x_i but NOT y_i, so the model must infer w in-context.

    Compared against a closed-form ridge-regression ceiling.
    """

    def __init__(self, n_pairs: int = 8, x_dim: int = 4, noise: float = 0.05,
                 w_scale: float = 1.0):
        self.n_pairs = n_pairs
        self.x_dim = x_dim
        self.noise = noise
        self.w_scale = w_scale
        self.token_dim = x_dim + 1
        self.out_dim = 1
        self.loss_type = "mse"

    def generate(self, B, rng):
        p, xd = self.n_pairs, self.x_dim
        T = p + 1
        x = np.zeros((B, T, self.token_dim), dtype=np.float32)
        y = np.zeros((B, T, 1), dtype=np.float32)
        mask = np.zeros((B, T), dtype=bool)
        for b in range(B):
            w = rng.standard_normal(xd).astype(np.float32) * self.w_scale
            X = rng.standard_normal((p, xd)).astype(np.float32)
            Y = X @ w + self.noise * rng.standard_normal(p).astype(np.float32)
            for i in range(p):
                x[b, i, :xd] = X[i]
                x[b, i, xd] = 1.0  # write marker
                y[b, i, 0] = Y[i]
                mask[b, i] = True
            xq = rng.standard_normal(xd).astype(np.float32)
            x[b, p, :xd] = xq
            x[b, p, xd] = 0.0  # query marker
            y[b, p, 0] = float(xq @ w)
            mask[b, p] = True
        return (torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask))

    def ceiling(self, x, y, mask) -> dict:
        """Closed-form ridge regression on the write pairs, evaluated at query."""
        B, T, _ = x.shape
        xd = self.x_dim
        errs = []
        for b in range(B):
            idx = mask[b].nonzero(as_tuple=False).flatten().tolist()
            if len(idx) < 2:
                continue
            qpos = idx[-1]
            pair_idx = idx[:-1]
            Xp = x[b, pair_idx, :xd]
            Yp = y[b, pair_idx, 0:1]
            Xq = x[b, qpos, :xd].unsqueeze(0)
            eye = torch.eye(xd, dtype=Xp.dtype)
            w_hat = torch.linalg.solve(Xp.T @ Xp + 0.01 * eye, Xp.T @ Yp)
            pred = float(Xq @ w_hat)
            errs.append((pred - float(y[b, qpos, 0])) ** 2)
        return {"ceiling_mse": float(np.mean(errs))} if errs else {"ceiling_mse": float("nan")}


# ---------------------------------------------------------------------------
# 7. Permutation composition (S3, then S5)
# ---------------------------------------------------------------------------

class PermutationComposition(SyntheticTask):
    """Compose a sequence of random permutations of an n-element set; output
    the running composition at each step (one-hot over the n! group elements).

    State tracking beyond flat recall; GDN-2 is expected to separate here.
    """

    def __init__(self, n: int = 3, n_perms: int = 4, seed_base: int = 0):
        self.n = n
        self.n_perms = n_perms
        self.group = self._build_group(n)
        self.ng = len(self.group["perms"])
        self.token_dim = self.ng
        self.out_dim = self.ng
        self.loss_type = "ce"

    @staticmethod
    def _build_group(n):
        from itertools import permutations
        perms = [tuple(p) for p in permutations(range(n))]
        idx = {p: i for i, p in enumerate(perms)}

        def compose(a, b):  # a after b
            return tuple(a[b[i]] for i in range(n))

        mul = {}
        for i, a in enumerate(perms):
            for j, b in enumerate(perms):
                mul[(i, j)] = idx[compose(a, b)]
        return {"perms": perms, "idx": idx, "mul": mul}

    def generate(self, B, rng):
        ng = self.ng
        n = self.n
        T = self.n_perms + 1
        x = np.zeros((B, T, ng), dtype=np.float32)
        y = np.zeros((B, T), dtype=np.int64)
        mask = np.zeros((B, T), dtype=bool)
        mul = self.group["mul"]
        for b in range(B):
            cur = self.group["idx"][tuple(range(n))]  # identity
            x[b, 0, cur] = 1.0
            y[b, 0] = cur
            mask[b, 0] = True
            for i in range(1, T):
                p = rng.integers(ng)
                cur = mul[(p, cur)]
                x[b, i, p] = 1.0
                y[b, i] = cur
                mask[b, i] = True
        return (torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASKS: dict[str, type] = {
    "overwrite_recall": OverwriteRecall,
    "correlated_key": CorrelatedKeyRecall,
    "capacity": CapacitySweep,
    "needle": NeedleInHaystack,
    "xor": XORNonlinear,
    "in_context_regression": InContextLinearRegression,
    "permutation_S3": lambda **kw: PermutationComposition(n=3, **kw),
    "permutation_S5": lambda **kw: PermutationComposition(n=5, **kw),
}

# Default constructor params per task (overridable by CLI / sweeps).
TASK_DEFAULTS: dict[str, dict] = {
    "overwrite_recall": dict(n_pairs=8, kd=8, vd=8),
    "correlated_key": dict(n_pairs=8, kd=8, vd=8),
    "capacity": dict(n_pairs=8, kd=8, vd=8),
    "needle": dict(distractors=8, kd=8, vd=8),
    "xor": dict(n_bits=8),
    "in_context_regression": dict(n_pairs=8, x_dim=4),
    "permutation_S3": dict(n_perms=4),
    "permutation_S5": dict(n_perms=4),
}


def make_task(name: str, overrides: Optional[dict] = None) -> SyntheticTask:
    base = dict(TASK_DEFAULTS.get(name, {}))
    base.update(overrides or {})
    cls = TASKS[name]
    if cls is PermutationComposition:
        return cls(**base)
    if cls.__name__ in ("OverwriteRecall", "CorrelatedKeyRecall", "CapacitySweep",
                        "NeedleInHaystack", "XORNonlinear", "InContextLinearRegression"):
        return cls(**base)
    return cls(**base)


__all__ = ["TASKS", "TASK_DEFAULTS", "make_task", "loss_fn", "compute_metric", "SyntheticTask"]
