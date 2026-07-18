#!/usr/bin/env python
r"""Static single-input / single-output benchmark for the Liquid LLU models.

Drives :class:`llu.models.mlp_model.LiquidMLP` (every ``nn.Linear`` replaced by
an LLU from :mod:`llu.models.llns`) across the mechanism-probing static tasks in
:mod:`io_tasks`, and writes a readable report. This is the cheapest screen for
the *static* regime -- fixed-size vector in, fixed-size vector out -- where the
question is raw expressivity, optimisation behaviour, and inductive bias rather
than sequence / recurrence:

* ``mod_add`` / ``mod_mul`` -- ``(a, b) -> (a op b) mod p``. The canonical
  grokking benchmark: train on a fraction of all pairs and watch the
  memorisation -> generalisation phase transition.
* ``fourier`` -- fit a sum of sinusoids at a controlled frequency. A direct
  probe of spectral bias (does the architecture learn high frequencies?).
* ``parity`` -- target is the XOR of a ``k``-sparse subset of input bits. Plain
  MLPs trained with SGD need exponential samples to find the right feature
  combination, so this isolates compositional inductive bias.

All datasets are generated on the fly (no downloads). Deterministic given
``--seed``.

Usage
-----
    python scripts/train_io.py                       # full matrix (slow)
    python scripts/train_io.py --quick               # tiny smoke
    python scripts/train_io.py --tasks mod_add mod_mul
    python scripts/train_io.py --llns StableLiquidLN RankRLiquidLN
    python scripts/train_io.py --p 23 --op add --train_frac 0.5 --steps 4000
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llu.models.mlp_model import LiquidMLP, IO_LLN_REGISTRY
from io_tasks import make_task, TASK_FACTORIES

ALL_LLNS = list(IO_LLN_REGISTRY.keys())
ALL_TASKS = list(TASK_FACTORIES.keys())


# ---------------------------------------------------------------------------
# Loss / metric
# ---------------------------------------------------------------------------

def loss_fn(pred: torch.Tensor, y: torch.Tensor, loss_type: str) -> torch.Tensor:
    if loss_type == "ce":
        return F.cross_entropy(pred, y.long())
    return F.mse_loss(pred, y)


# ---------------------------------------------------------------------------
# One (task, LLN) run
# ---------------------------------------------------------------------------

def run_one(task, lln_name: str, lln_cls: type, cfg: dict) -> dict:
    device = cfg["device"]
    rng = np.random.default_rng(cfg["seed"])
    x_all, y_all = task.full_data(rng)
    N = x_all.shape[0]
    n_train = max(1, int(round(cfg["train_frac"] * N)))
    perm = rng.permutation(N)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    model = LiquidMLP(
        in_dim=task.input_dim, hidden=cfg["hidden"], n_layers=cfg["n_layers"],
        out_dim=task.out_dim, lln_cls=lln_cls, parameterization=cfg["parameterization"],
        rank=cfg["rank"], act=cfg["act"],
    ).to(device)
    n_params = model.num_params()
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    xt_all = x_all.to(device)
    yt_all = y_all.to(device)
    train_idx_t = torch.as_tensor(train_idx, device=device)

    step_times: list[float] = []
    last_train_loss = float("nan")
    last_train_metric = float("nan")
    last_test_loss = float("nan")
    last_test_metric = float("nan")
    best_test_metric = -math.inf
    notes = ""
    t0 = time.perf_counter()
    for step in range(cfg["steps"]):
        ts = time.perf_counter()
        sel = train_idx[rng.integers(0, len(train_idx), size=cfg["batch"])]
        x = xt_all[train_idx_t.new_tensor(sel)]
        y = yt_all[train_idx_t.new_tensor(sel)]
        pred = model(x)
        loss = loss_fn(pred, y, task.loss_type)
        opt.zero_grad()
        loss.backward()
        opt.step()
        step_times.append(time.perf_counter() - ts)
        last_train_loss = loss.item()
        last_train_metric = task.metric(pred.detach(), y)
        if not math.isfinite(last_train_loss):
            notes = f"non-finite train loss at step {step}"
            break
        if (step + 1) % cfg["eval_every"] == 0 or step == cfg["steps"] - 1:
            if len(test_idx) > 0:
                with torch.no_grad():
                    pt = model(xt_all[test_idx])
                    last_test_loss = loss_fn(pt, yt_all[test_idx], task.loss_type).item()
                    last_test_metric = task.metric(pt, yt_all[test_idx])
                    best_test_metric = max(best_test_metric, last_test_metric)
                kind = "acc" if task.loss_type == "ce" else "rmse"
                val = last_test_metric if task.loss_type == "ce" else -last_test_metric
                print(f"    [{task.name} {lln_name}] step {step + 1:>6} "
                      f"tr_loss={loss.item():.4f} te_{kind}={val:.4f}")
    total = time.perf_counter() - t0
    ms_per_step = (sum(step_times) / len(step_times) * 1000.0) if step_times else float("nan")

    return {
        "task": task.name,
        "lln": lln_name,
        "params": n_params,
        "steps": cfg["steps"],
        "batch": cfg["batch"],
        "train_frac": cfg["train_frac"],
        "loss_type": task.loss_type,
        "final_train_loss": last_train_loss,
        "final_train_metric": last_train_metric,
        "final_test_loss": last_test_loss,
        "final_test_metric": last_test_metric,
        "best_test_metric": best_test_metric,
        "ms_per_step": ms_per_step,
        "total_train_s": total,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _fmt(v, w: int) -> str:
    if isinstance(v, float):
        if not math.isfinite(v):
            return f"{'nan':>{w}}"
        return f"{v:>{w}.4f}"
    return f"{str(v):>{w}}"


def write_report(results: list[dict], cfg: dict, path: str) -> None:
    lines: list[str] = []
    lines.append("# Static (single-input / single-output) LLU benchmark")
    lines.append(f"generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(
        f"config: hidden={cfg['hidden']} n_layers={cfg['n_layers']} rank={cfg['rank']} "
        f"param={cfg['parameterization']} act={cfg['act']} train_frac={cfg['train_frac']} "
        f"steps={cfg['steps']} lr={cfg['lr']} batch={cfg['batch']}"
    )
    lines.append("")
    # header
    cols = ["task", "lln", "params", "tr_loss", "tr_metric", "te_loss", "te_metric",
            "best_te", "ms/step", "tot_s", "notes"]
    widths = [14, 22, 9, 9, 9, 9, 9, 9, 8, 7, 20]
    lines.append("  ".join(_fmt(c, w) for c, w in zip(cols, widths)))
    lines.append("  ".join("-" * w for w in widths))
    for r in results:
        tr_m = _fmt(r["final_train_metric"], widths[4])
        te_m = _fmt(r["final_test_metric"], widths[6])
        best_m = _fmt(r["best_test_metric"], widths[7])
        row = [
            r["task"], r["lln"], r["params"],
            _fmt(r["final_train_loss"], widths[3]), tr_m,
            _fmt(r["final_test_loss"], widths[5]), te_m, best_m,
            _fmt(r["ms_per_step"], widths[8]), _fmt(round(r["total_train_s"], 1), widths[9]),
            r["notes"],
        ]
        lines.append("  ".join(str(c) for c in row))
    lines.append("")
    lines.append("tr_metric / te_metric: accuracy for ce tasks, -RMSE for mse tasks "
                 "(higher is better for both).")
    text = "\n".join(lines) + "\n"
    with open(path, "w") as f:
        f.write(text)
    print(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> dict:
    p = argparse.ArgumentParser(description="LLU static (IO) benchmark")
    p.add_argument("--tasks", nargs="+", default=ALL_TASKS, help="IO tasks to run")
    p.add_argument("--llns", nargs="+", default=ALL_LLNS,
                   help="LLU variants for the MLP linear maps")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--parameterization", choices=["lora", "svd"], default="lora")
    p.add_argument("--act", choices=["relu", "gelu", "silu", "tanh", "none"], default="relu")
    p.add_argument("--train_frac", type=float, default=0.5,
                   help="fraction of pairs held for training (rest = grokking test)")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0,
                   help="L2 on params; weight decay is what drives the grokking "
                        "generalisation phase transition (try 1.0)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="io_bench_report.txt")
    # task-specific overrides
    p.add_argument("--p", type=int, default=97, help="modulus for mod_* tasks")
    p.add_argument("--op", choices=["add", "mul"], default="add")
    p.add_argument("--dim", type=int, default=1, help="input dim for fourier task")
    p.add_argument("--parity_dim", type=int, default=20, help="bit-vector length for parity task")
    p.add_argument("--n_freqs", type=int, default=3, help="fourier task")
    p.add_argument("--max_w", type=float, default=6.0, help="fourier task max frequency")
    p.add_argument("--k", type=int, default=4, help="sparse-parity active bits")
    p.add_argument("--quick", action="store_true",
                   help="tiny smoke: small p/hidden/layers/steps")
    a = p.parse_args()

    if a.quick:
        a.p = 11
        a.hidden = 32
        a.n_layers = 1
        a.steps = 15
        a.eval_every = 5
        a.batch = 64

    cfg = {
        "tasks": a.tasks,
        "llns": a.llns,
        "hidden": a.hidden,
        "n_layers": a.n_layers,
        "rank": a.rank,
        "parameterization": a.parameterization,
        "act": a.act,
        "train_frac": a.train_frac,
        "steps": a.steps,
        "eval_every": a.eval_every,
        "batch": a.batch,
        "lr": a.lr,
        "weight_decay": a.weight_decay,
        "seed": a.seed,
        "device": a.device,
        "out": a.out,
        # task params
        "p": a.p,
        "op": a.op,
        "dim": a.dim,
        "parity_dim": a.parity_dim,
        "n_freqs": a.n_freqs,
        "max_w": a.max_w,
        "k": a.k,
    }
    return cfg


def main() -> None:
    cfg = parse_args()
    unknown = [t for t in cfg["tasks"] if t not in TASK_FACTORIES]
    if unknown:
        raise SystemExit(f"unknown task(s): {unknown}; choose from {ALL_TASKS}")
    unknown_l = [name for name in cfg["llns"] if name not in IO_LLN_REGISTRY]
    if unknown_l:
        raise SystemExit(f"unknown LLN(s): {unknown_l}; choose from {ALL_LLNS}")

    results: list[dict] = []
    for tname in cfg["tasks"]:
        task = make_task(tname, cfg)
        for lname in cfg["llns"]:
            lcls = IO_LLN_REGISTRY[lname]
            res = run_one(task, lname, lcls, cfg)
            results.append(res)
            is_ce = res["loss_type"] == "ce"
            metric_kind = "acc" if is_ce else "rmse"
            metric_val = res["final_test_metric"] if is_ce else -res["final_test_metric"]
            print(f"[{tname:12s} {lname:22s}] params={res['params']:>8} "
                  f"tr_loss={res['final_train_loss']:.4f} "
                  f"te_{metric_kind}={metric_val:.4f} {res['notes']}")

    write_report(results, cfg, cfg["out"])
    print(f"\nReport written to {cfg['out']}")


if __name__ == "__main__":
    main()
