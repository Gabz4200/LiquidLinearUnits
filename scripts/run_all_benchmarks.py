#!/usr/bin/env python
r"""Unified benchmark runner for all LLU architectures.

Runs train_synth.py (sequence tasks) and train_io.py (static MLP tasks)
with tiny configs, then generates a unified comparison table.

Usage:
    python scripts/run_all_benchmarks.py
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPTS_DIR)


def run_cmd(cmd: list[str], label: str) -> str:
    print(f"\n{'=' * 60}")
    print(f"  RUNNING: {label}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=ROOT_DIR,
        timeout=600,
    )
    if result.returncode != 0:
        print(f"STDERR:\n{result.stderr[-2000:]}")
    return result.stdout + result.stderr


def parse_synth_report(path: str) -> list[dict]:
    """Parse synth_bench_report.txt into structured results."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        text = f.read()

    results = []
    current_task = None
    for line in text.split("\n"):
        m = re.match(r"^TASK (\w+)", line)
        if m:
            current_task = m.group(1)
            continue
        if (
            not current_task
            or line.startswith("-")
            or line.startswith("=")
            or line.startswith("  arch")
        ):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        arch = parts[0]
        if arch in ("arch", "---"):
            continue
        try:
            params = int(parts[1])
            tr_loss = float(parts[2])
            ev_loss = float(parts[3])
        except (ValueError, IndexError):
            continue

        metric = {}
        ms_step = float("nan")
        for p in parts[4:]:
            if "=" in p:
                k, v = p.split("=", 1)
                try:
                    metric[k] = float(v)
                except ValueError:
                    pass
            elif re.match(r"^\d+\.?\d*$", p):
                ms_step = float(p)

        results.append(
            {
                "task": current_task,
                "arch": arch,
                "params": params,
                "tr_loss": tr_loss,
                "ev_loss": ev_loss,
                "metric": metric,
                "ms_per_step": ms_step,
            }
        )
    return results


def parse_io_report(path: str) -> list[dict]:
    """Parse io_bench_report.txt into structured results."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        text = f.read()

    results = []
    lines = text.strip().split("\n")
    for line in lines:
        parts = line.split()
        if len(parts) < 8:
            continue
        if parts[0] in ("task", "---", "#", "tr_metric"):
            continue
        try:
            task = parts[0]
            lln = parts[1]
            params = int(parts[2])
            tr_loss = float(parts[3])
            tr_metric = float(parts[4])
            te_loss = float(parts[5])
            te_metric = float(parts[6])
            best_te = float(parts[7])
        except (ValueError, IndexError):
            continue
        # Infer loss_type: positive metric = accuracy (CE), negative = -RMSE (MSE)
        loss_type = "ce" if te_metric >= 0 else "mse"
        results.append(
            {
                "task": task,
                "arch": lln,
                "params": params,
                "tr_loss": tr_loss,
                "tr_metric": tr_metric,
                "te_loss": te_loss,
                "te_metric": te_metric,
                "best_metric": best_te,
                "loss_type": loss_type,
            }
        )
    return results


def fmt(v: float, prec: int = 3) -> str:
    if not math.isfinite(v):
        return "-"
    return f"{v:.{prec}f}"


def fmt_pct(v: float) -> str:
    if not math.isfinite(v):
        return "-"
    return f"{v * 100:.1f}%"


def generate_table(synth_results: list[dict], io_results: list[dict], out_path: str):
    """Generate a unified Markdown comparison table."""
    lines = []
    lines.append("# LLU Architecture Benchmark Comparison")
    lines.append(f"")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Device: CPU | Config: quick (tiny model, 30 steps)")
    lines.append("")
    lines.append("Legend: **bold** = best in column, tr = train loss, ev = eval loss")
    lines.append("")

    # ---- SYNTHETIC SEQUENCE TASKS ----
    if synth_results:
        lines.append("## Synthetic Sequence Tasks (LiquidTransformer)")
        lines.append("")

        # Collect unique tasks and archs
        tasks = sorted(set(r["task"] for r in synth_results))
        archs = sorted(set(r["arch"] for r in synth_results))

        # Index results
        idx = {}
        for r in synth_results:
            idx[(r["task"], r["arch"])] = r

        # Summary table: one row per arch, columns = tasks
        lines.append("### Eval Loss by Task (lower is better)")
        lines.append("")
        header = "| Architecture |"
        sep = "|---|"
        for t in tasks:
            short = (
                t.replace("OverwriteRecall", "OWRecall")
                .replace("InContextLinearRegression", "ICReg")
                .replace("PermutationComposition", "PermComp")
                .replace("CorrelatedKeyRecall", "CorrKey")
                .replace("NeedleInHaystack", "Needle")
                .replace("SelectiveCopy", "SelCopy")
                .replace("InductionHeads", "IndHead")
                .replace("CapacitySweep", "Capacity")
                .replace("XORNonlinear", "XOR")
            )
            header += f" {short} |"
            sep += "---:|"
        lines.append(header)
        lines.append(sep)

        for arch in archs:
            row = f"| {arch} |"
            vals = []
            for t in tasks:
                r = idx.get((t, arch))
                if r:
                    vals.append((r["ev_loss"], fmt(r["ev_loss"])))
                else:
                    vals.append((float("nan"), "-"))
            # Find best (min ev_loss) per task
            best_per_task = {}
            for t in tasks:
                best_val = float("inf")
                for a in archs:
                    r = idx.get((t, a))
                    if r and math.isfinite(r["ev_loss"]) and r["ev_loss"] < best_val:
                        best_val = r["ev_loss"]
                best_per_task[t] = best_val

            for i, t in enumerate(tasks):
                val, s = vals[i]
                if math.isfinite(val) and val <= best_per_task[t] * 1.001:
                    s = f"**{s}**"
                row += f" {s} |"
            lines.append(row)

        # Speed table
        lines.append("")
        lines.append("### Speed (ms/step, lower is better)")
        lines.append("")
        header = "| Architecture |"
        sep = "|---|"
        for t in tasks:
            short = (
                t.replace("OverwriteRecall", "OWRecall")
                .replace("InContextLinearRegression", "ICReg")
                .replace("PermutationComposition", "PermComp")
                .replace("CorrelatedKeyRecall", "CorrKey")
                .replace("NeedleInHaystack", "Needle")
                .replace("SelectiveCopy", "SelCopy")
                .replace("InductionHeads", "IndHead")
                .replace("CapacitySweep", "Capacity")
                .replace("XORNonlinear", "XOR")
            )
            header += f" {short} |"
            sep += "---:|"
        lines.append(header)
        lines.append(sep)

        for arch in archs:
            row = f"| {arch} |"
            for t in tasks:
                r = idx.get((t, arch))
                ms = r["ms_per_step"] if r else float("nan")
                row += f" {fmt(ms, 1)} |"
            lines.append(row)

        # Param counts
        lines.append("")
        lines.append("### Parameter Counts")
        lines.append("")
        lines.append("| Architecture | Params |")
        lines.append("|---|---:|")
        for arch in archs:
            r = idx.get((tasks[0], arch))
            if r:
                lines.append(f"| {arch} | {r['params']:,} |")

    # ---- STATIC IO TASKS ----
    if io_results:
        lines.append("")
        lines.append("## Static IO Tasks (LiquidMLP)")
        lines.append("")

        tasks = sorted(set(r["task"] for r in io_results))
        archs = sorted(set(r["arch"] for r in io_results))

        idx = {}
        for r in io_results:
            idx[(r["task"], r["arch"])] = r

        # Test metric table (higher is better)
        lines.append("### Test Metric (higher is better)")
        lines.append("")
        header = "| Architecture |"
        sep = "|---|"
        for t in tasks:
            header += f" {t} |"
            sep += "---:|"
        lines.append(header)
        lines.append(sep)

        for arch in archs:
            row = f"| {arch} |"
            vals = []
            for t in tasks:
                r = idx.get((t, arch))
                vals.append((r["best_metric"] if r else float("nan"), r))
            best_per_task = {}
            for t in tasks:
                best_val = float("-inf")
                for a in archs:
                    r = idx.get((t, a))
                    if r and math.isfinite(r["best_metric"]) and r["best_metric"] > best_val:
                        best_val = r["best_metric"]
                best_per_task[t] = best_val

            for i, t in enumerate(tasks):
                val, r = vals[i]
                if r and r["loss_type"] == "ce":
                    s = fmt_pct(val)
                    if math.isfinite(val) and val >= best_per_task[t] * 0.999:
                        s = f"**{s}**"
                else:
                    s = fmt(-val, 4) if math.isfinite(val) else "-"
                row += f" {s} |"
            lines.append(row)

        # Train loss table
        lines.append("")
        lines.append("### Train Loss (lower is better)")
        lines.append("")
        header = "| Architecture |"
        sep = "|---|"
        for t in tasks:
            header += f" {t} |"
            sep += "---:|"
        lines.append(header)
        lines.append(sep)

        for arch in archs:
            row = f"| {arch} |"
            for t in tasks:
                r = idx.get((t, arch))
                row += f" {fmt(r['tr_loss']) if r else '-'} |"
            lines.append(row)

        # Param counts
        lines.append("")
        lines.append("### Parameter Counts")
        lines.append("")
        lines.append("| Architecture | Params |")
        lines.append("|---|---:|")
        for arch in archs:
            r = idx.get((tasks[0], arch))
            if r:
                lines.append(f"| {arch} | {r['params']:,} |")

    text = "\n".join(lines) + "\n"
    with open(out_path, "w") as f:
        f.write(text)
    print(f"\nUnified report written to {out_path}")
    print(text)


def main():
    t0 = time.perf_counter()

    synth_report = os.path.join(ROOT_DIR, "synth_bench_report.txt")
    io_report = os.path.join(ROOT_DIR, "io_bench_report.txt")

    # 1. Run synthetic sequence benchmark
    run_cmd(
        [
            sys.executable,
            os.path.join(SCRIPTS_DIR, "train_synth.py"),
            "--quick",
            "--no_sweeps",
            "--steps",
            "30",
            "--batch",
            "8",
            "--eval_batch",
            "64",
            "--num_layers",
            "1",
            "--window",
            "8",
            "--out",
            synth_report,
        ],
        "Synthetic Sequence Benchmark (all archs, quick)",
    )

    # 2. Run static IO benchmark
    run_cmd(
        [
            sys.executable,
            os.path.join(SCRIPTS_DIR, "train_io.py"),
            "--quick",
            "--hidden",
            "32",
            "--n_layers",
            "1",
            "--steps",
            "100",
            "--batch",
            "64",
            "--eval_every",
            "20",
            "--out",
            io_report,
        ],
        "Static IO Benchmark (all LLNs, quick)",
    )

    # 3. Parse and generate unified table
    synth_results = parse_synth_report(synth_report)
    io_results = parse_io_report(io_report)

    table_path = os.path.join(ROOT_DIR, "benchmarks", "UNIFIED_RESULTS.md")
    os.makedirs(os.path.dirname(table_path), exist_ok=True)
    generate_table(synth_results, io_results, table_path)

    elapsed = time.perf_counter() - t0
    print(f"\nTotal benchmark time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
