#!/usr/bin/env python
r"""Training + benchmarking pipeline for the Liquid Linear Unit architectures.

Runs every architecture from :mod:`llu.models.llns` (wrapped in
:class:`llu.models.liquid_model.LiquidTransformer`) across the synthetic
benchmark tasks, collects quality / loss / speed / parameter metrics, and
writes a readable text report.

Usage
-----
    python scripts/train.py                      # full matrix (slow on CPU)
    python scripts/train.py --quick              # tiny smoke configuration
    python scripts/train.py --tasks overwrite_recall --archs SharedMomentumLiquidLN
    python scripts/train.py --no-sweeps          # one config per task, no sweeps

All sequences are generated on the fly (no downloads). Every run is fully
deterministic given ``--seed`` and the per-task/sweep data seed, and the data
is identical across architectures for a fair comparison.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from llu.models.liquid_model import build_model, ARCH_FACTORIES, is_valid_arch, RECURRENT_ARCHS
from bench_tasks import TASKS, make_task, loss_fn, compute_metric

ALL_ARCHS = list(ARCH_FACTORIES.keys())


# ---------------------------------------------------------------------------
# One (architecture, task, sweep) run
# ---------------------------------------------------------------------------

def run_one(arch: str, task, d_model: int, out_dim: int, cfg: dict, data_seed: int) -> dict:
    device = cfg["device"]
    torch.manual_seed(cfg["seed"])

    model = build_model(
        arch, d_model, out_dim,
        num_layers=cfg["num_layers"], window=cfg["window"], n_heads=cfg["n_heads"],
        use_swiglu=cfg["use_swiglu"], swiglu_mult=cfg["swiglu_mult"],
        rank=cfg["rank"], decay_rate=cfg["decay_rate"],
        use_attention=cfg["use_attention"],
        parameterization=cfg["parameterization"],
    )
    model.to(device)
    n_params = model.num_params()
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    train_rng = np.random.default_rng(data_seed)
    eval_rng = np.random.default_rng(data_seed + 999)

    model.train()
    step_times: list[float] = []
    last_train_loss = float("nan")
    notes = ""
    t0 = time.perf_counter()
    for step in range(cfg["steps"]):
        ts = time.perf_counter()
        x, y, mask = task.generate(cfg["batch"], train_rng)
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        pred = model(x)
        loss = loss_fn(pred, y, mask, task.loss_type)
        opt.zero_grad()
        loss.backward()
        opt.step()
        step_times.append(time.perf_counter() - ts)
        last_train_loss = loss.item()
        if not math.isfinite(last_train_loss):
            notes = f"non-finite train loss at step {step}"
            break
    total_train = time.perf_counter() - t0

    # Evaluation on a fresh, larger batch.
    model.eval()
    with torch.no_grad():
        xv, yv, maskv = task.generate(cfg["eval_batch"], eval_rng)
        xv, yv, maskv = xv.to(device), yv.to(device), maskv.to(device)
        pv = model(xv)
        eval_loss = loss_fn(pv, yv, maskv, task.loss_type).item()
        metric = compute_metric(pv, yv, maskv, task.loss_type)
        ceiling = task.ceiling(xv, yv, maskv) if hasattr(task, "ceiling") else {}

    ms_per_step = (sum(step_times) / len(step_times) * 1000.0) if step_times else float("nan")
    steps_per_sec = (len(step_times) / sum(step_times)) if step_times else float("nan")

    return {
        "arch": arch,
        "task": task.__class__.__name__,
        "d_model": d_model,
        "out_dim": out_dim,
        "params": n_params,
        "steps": cfg["steps"],
        "batch": cfg["batch"],
        "final_train_loss": last_train_loss,
        "final_eval_loss": eval_loss,
        "metric": metric,
        "ceiling": ceiling,
        "ms_per_step": ms_per_step,
        "steps_per_sec": steps_per_sec,
        "total_train_s": total_train,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _fmt(v, w, kind="f"):
    if isinstance(v, float):
        if not math.isfinite(v):
            return f"{'nan':>{w}}"
        return f"{v:{w}.4g}"
    return f"{str(v):>{w}}"


def write_report(results: list[dict], cfg: dict, path: str) -> None:
    lines: list[str] = []
    A = lines.append
    A("=" * 72)
    A("LIQUID LINEAR UNITS -- BENCHMARK REPORT")
    A(f"generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    A(f"device    : {cfg['device']}")
    A("-" * 72)
    A("GLOBAL CONFIG")
    A(f"  archs        : {', '.join(cfg['archs'])}")
    A(f"  tasks        : {', '.join(cfg['tasks'])}")
    A(f"  sweeps       : {'on' if cfg['sweeps'] else 'off'}")
    A(f"  use_attention : {cfg.get('use_attention', True)}   ablate_attention : {cfg.get('ablate_attention', False)}")
    A(f"  steps        : {cfg['steps']}   batch : {cfg['batch']}   eval_batch : {cfg['eval_batch']}")
    A(f"  num_layers   : {cfg['num_layers']}   window : {cfg['window']}   n_heads : {cfg['n_heads']}")
    A(f"  use_swiglu   : {cfg['use_swiglu']}   swiglu_mult : {cfg['swiglu_mult']}")
    A(f"  rank         : {cfg['rank']}   decay_rate : {cfg['decay_rate']}   lr : {cfg['lr']}")
    A(f"  seed         : {cfg['seed']}   quick : {cfg['quick']}")
    A("=" * 72)
    A("")

    # Group results by task (and sweep label).
    by_task: dict[str, list[dict]] = {}
    for r in results:
        by_task.setdefault(r["task"], []).append(r)

    for task_name, rows in by_task.items():
        A("-" * 72)
        A(f"TASK {task_name}")
        A("-" * 72)
        hdr = (f"{'arch':30s} {'params':>10s} {'tr_loss':>9s} {'ev_loss':>9s} "
               f"{'metric':>40s} {'ms/step':>8s} {'st/s':>7s}")
        A(hdr)
        A("-" * 72)
        for r in rows:
            # Full metric string: key=value (never clipped).
            m = r["metric"]
            mstr = " ".join(f"{k}={v:.3g}" for k, v in m.items()) if m else "-"
            if r["ceiling"]:
                mstr += " " + " ".join(f"{k}={v:.3g}" for k, v in r["ceiling"].items())
            A(f"{r['arch']:30s} {r['params']:>10d} "
              f"{_fmt(r['final_train_loss'], 9)} {_fmt(r['final_eval_loss'], 9)} "
              f"{mstr:>22s} {_fmt(r['ms_per_step'], 8)} {_fmt(r['steps_per_sec'], 7)}")
            if r["notes"]:
                A(f"    ! {r['notes']}")
        A("")

    A("=" * 72)
    A(f"END OF REPORT -- {len(results)} runs")
    A("=" * 72)

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> dict:
    p = argparse.ArgumentParser(description="LLU architecture benchmark")
    p.add_argument("--archs", default=",".join(ALL_ARCHS),
                   help="comma-separated architectures")
    p.add_argument("--tasks", default=",".join(TASKS.keys()),
                   help="comma-separated task names")
    p.add_argument("--d_model", type=int, default=None,
                   help="override model dim (default: task.token_dim)")
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--use_swiglu", action="store_true", default=True)
    p.add_argument("--no_swiglu", dest="use_swiglu", action="store_false")
    p.add_argument("--swiglu_mult", type=int, default=4)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--decay_rate", type=float, default=0.4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--eval_batch", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="benchmark_report.txt")
    p.add_argument("--sweeps", dest="sweeps", action="store_true", default=True)
    p.add_argument("--no_sweeps", dest="sweeps", action="store_false")
    p.add_argument("--parameterization", default="lora", choices=["lora", "svd"],
                   help="parameterization mode for low-rank updates")
    p.add_argument("--no_attention", action="store_true",
                   help="build every selected architecture with attention removed")
    p.add_argument("--ablate_attention", action="store_true",
                   help="compare recurrent archs with and without attention, plus the others")
    p.add_argument("--quick", action="store_true",
                   help="tiny config for a fast smoke test")
    a = p.parse_args()

    if a.quick:
        a.steps = 30
        a.batch = 8
        a.eval_batch = 64
        a.num_layers = 1
        a.window = 8

    cfg = dict(
        archs=[s for s in a.archs.split(",") if s],
        tasks=[s for s in a.tasks.split(",") if s],
        d_model_override=a.d_model,
        num_layers=a.num_layers, window=a.window, n_heads=a.n_heads,
        use_swiglu=a.use_swiglu, swiglu_mult=a.swiglu_mult,
        rank=a.rank, decay_rate=a.decay_rate, lr=a.lr,
        steps=a.steps, batch=a.batch, eval_batch=a.eval_batch,
        seed=a.seed, device=a.device, out=a.out,
        sweeps=a.sweeps, quick=a.quick,
        use_attention=not a.no_attention, ablate_attention=a.ablate_attention,
        parameterization=a.parameterization,
    )
    return cfg


def main() -> None:
    cfg = parse_args()
    if cfg["ablate_attention"]:
        # Recurrent archs with and without attention, plus the others (with attention).
        rec = [a for a in ALL_ARCHS if a in RECURRENT_ARCHS]
        nonrec = [a for a in ALL_ARCHS if a not in RECURRENT_ARCHS]
        cfg["archs"] = [*rec, *[a + "_noattn" for a in rec], *nonrec]
    results: list[dict] = []

    for task_name in cfg["tasks"]:
        if task_name not in TASKS:
            print(f"[skip] unknown task {task_name}")
            continue
        # Base task instance just to read token_dim / out_dim / sweep.
        base_task = make_task(task_name)
        sweeps = base_task.sweep() if cfg["sweeps"] else [{}]
        for sweep in sweeps:
            task = make_task(task_name, sweep)
            sweep_label = ",".join(f"{k}={v}" for k, v in sweep.items())
            d_model = cfg["d_model_override"] or task.token_dim
            out_dim = task.out_dim
            # Deterministic, architecture-independent data seed for this task+sweep.
            data_seed = (abs(hash((task_name, tuple(sorted(sweep.items()))))) % (2 ** 31))
            print(f"\n=== TASK {task_name}  sweep=[{sweep_label}]  d_model={d_model} out_dim={out_dim} ===")
            for arch in cfg["archs"]:
                if not is_valid_arch(arch):
                    print(f"[skip] unknown arch {arch}")
                    continue
                r = run_one(arch, task, d_model, out_dim, cfg, data_seed)
                m = r["metric"]
                mstr = " ".join(f"{k}={v:.3g}" for k, v in m.items())
                print(f"  {arch:26s} params={r['params']:>9d} "
                      f"tr={r['final_train_loss']:.4g} ev={r['final_eval_loss']:.4g} "
                      f"{mstr} {r['ms_per_step']:.1f} ms/step")
                r["task"] = task_name
                r["sweep_label"] = sweep_label
                r["sweep"] = sweep
                results.append(r)

    write_report(results, cfg, cfg["out"])
    print(f"\nReport written to {cfg['out']}")


if __name__ == "__main__":
    main()
