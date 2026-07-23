#!/usr/bin/env python
r"""LoRA rank sweep benchmark: compare LoRA (ranks 1-16) vs SVD on the LLM.

Tests StableLiquidLN with:
  - LoRA rank 1, 2, 4, 8, 16 (alpha=rank so scale=1.0)
  - LoRA rank 4 with alpha=1 (default, scale=0.25)
  - SVD rank 4 (default)
  - Baseline (GDN-2 mixer, no LLU)

All configs share the same tiny preset. Reports timing, params, train loss,
Wiki ppl, LAMBADA ppl, and LAMBADA accuracy.

Hardware: sized for i5-8250U (7.6 GB RAM, no CUDA).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from llu.models.liquid_llm import build_llm, num_params


def _require_internet() -> None:
    import socket

    try:
        socket.create_connection(("huggingface.co", 443), timeout=8)
    except OSError as e:
        sys.exit(f"[error] Cannot reach huggingface.co: {e}")


class PackedTokens(Dataset):
    def __init__(self, ids, seq_len):
        self.data = torch.tensor(ids[: (len(ids) // seq_len) * seq_len], dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) // self.seq_len

    def __getitem__(self, i):
        s = i * self.seq_len
        return self.data[s : s + self.seq_len]


def build_token_buffer(tok, tokens, dataset, seq_len):
    from datasets import load_dataset

    ids = []
    ds = load_dataset(dataset, split="train", streaming=True)
    for ex in ds:
        text = ex.get("text") or ""
        if not text:
            continue
        ids.extend(tok(text, add_special_tokens=False).input_ids)
        if len(ids) >= tokens:
            break
    return ids[:tokens]


def reset_momentum_buffers(model):
    for m in model.modules():
        for name in ("a_raw", "b_raw", "g_raw"):
            buf = getattr(m, name, None)
            if isinstance(buf, torch.Tensor):
                buf.zero_()


def wiki_ppl(model, tok, device, seq_len, max_tokens=10000):
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = []
    for ex in ds:
        t = (ex.get("text") or "").strip()
        if not t:
            continue
        ids.extend(tok(t, add_special_tokens=False).input_ids)
        if len(ids) >= max_tokens:
            break
    ids = ids[:max_tokens]
    if len(ids) < 2:
        return float("nan")
    model.eval()
    total_loss = 0.0
    total_tok = 0
    with torch.no_grad():
        for i in range(0, len(ids) - seq_len, seq_len):
            chunk = torch.tensor(ids[i : i + seq_len], device=device).unsqueeze(0)
            logits = model(chunk)[0, :-1]
            tgt = torch.tensor(ids[i + 1 : i + seq_len], device=device)
            n = logits.shape[0]
            total_loss += F.cross_entropy(logits, tgt).item() * n
            total_tok += n
    return math.exp(total_loss / total_tok)


def lambada_eval(model, tok, device, seq_len, max_ex=100):
    from datasets import load_dataset

    ds = load_dataset("EleutherAI/lambada_openai", split="test")
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    n_tok = 0
    with torch.no_grad():
        for ex in ds:
            ids = tok(ex["text"], add_special_tokens=False).input_ids
            if len(ids) < 2:
                continue
            ctx = ids[:-1][-(seq_len - 1) :]
            last = ids[-1]
            x = torch.tensor(ctx, device=device).unsqueeze(0)
            logits = model(x)[0, -1]
            pred = int(logits.argmax().item())
            correct += int(pred == last)
            total += 1
            loss_sum += F.cross_entropy(
                logits.unsqueeze(0), torch.tensor([last], device=device)
            ).item()
            n_tok += 1
            if total >= max_ex:
                break
    acc = correct / total if total else float("nan")
    ppl = math.exp(loss_sum / n_tok) if n_tok else float("nan")
    return ppl, acc


def fmt(v):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "-"
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def train_one(args, variant, lln_name, parameterization, device, tok, lora_alpha=1.0, rank=4):
    """Train one configuration and return results dict."""
    lln_label = lln_name or "baseline"
    if parameterization == "lora" and lora_alpha == rank:
        tag = f"{variant}_{lln_label}_lora_r{rank}_a{rank}"
    elif parameterization == "lora":
        tag = f"{variant}_{lln_label}_lora_r{rank}_a{lora_alpha}"
    else:
        tag = f"{variant}_{lln_label}_{parameterization}_r{rank}"
    ckpt_path = os.path.join(args.ckpt_dir, f"llm_{tag}.pt")

    model = build_llm(
        variant,
        args.preset,
        lln=lln_name or "StableLiquidLN",
        parameterization=parameterization,
        rank=rank,
        lora_alpha=lora_alpha,
    ).to(device)
    n_params = num_params(model)

    print(f"\n{'=' * 60}")
    print(
        f"  {tag}  |  {n_params:,} params  |  {parameterization}  |  rank={rank}  |  alpha={lora_alpha}"
    )
    print(f"{'=' * 60}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    model.train()
    t0 = time.perf_counter()
    step = 0
    last_loss = float("nan")
    best_loss = float("inf")
    patience_counter = 0
    early_stop = False

    for epoch in range(args.epochs):
        for x in args.loader:
            x = x.to(device)
            logits = model(x)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1)
            )

            if not torch.isfinite(loss).all():
                print(f"  [step {step}] loss={loss.item():.3f} -> UNSTABLE, stopping.")
                early_stop = True
                break

            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = loss.item()
            step += 1

            if last_loss < best_loss:
                best_loss = last_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if step % args.log_every == 0:
                print(f"  [step {step:4d}] loss={last_loss:.4f} (best={best_loss:.4f})")

            if args.early_stop_patience and patience_counter >= args.early_stop_patience:
                print(
                    f"  [early stop] no improvement for {args.early_stop_patience} steps at step {step}"
                )
                early_stop = True
                break

            if args.max_steps and step >= args.max_steps:
                break
        if early_stop:
            break

    train_time = time.perf_counter() - t0
    print(f"  [done] {step} steps in {train_time:.1f}s ({train_time / step:.2f}s/step)")

    # Save checkpoint
    torch.save(
        {
            "variant": variant,
            "preset": args.preset,
            "lln": lln_name,
            "parameterization": parameterization,
            "rank": rank,
            "lora_alpha": lora_alpha,
            "state": model.state_dict(),
        },
        ckpt_path,
    )

    # Eval
    metrics = {
        "train_loss_final": float(last_loss),
        "train_loss_best": float(best_loss),
        "steps": step,
    }
    if args.eval:
        reset_momentum_buffers(model)
        print(f"  [eval] Wiki ppl ...")
        metrics["wiki_ppl"] = wiki_ppl(model, tok, device, args.seq_len, args.wiki_tokens)
        print(f"  [eval] LAMBADA ...")
        lmb_ppl, lmb_acc = lambada_eval(model, tok, device, args.seq_len, args.lambada_ex)
        metrics["lmb_ppl"] = lmb_ppl
        metrics["lmb_acc"] = lmb_acc
        print(
            f"  [eval] Wiki={metrics['wiki_ppl']:.1f}  LMB_ppl={lmb_ppl:.1f}  LMB_acc={lmb_acc:.3f}"
        )

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {
        "tag": tag,
        "variant": variant,
        "lln": lln_name or "-",
        "parameterization": parameterization,
        "rank": rank,
        "lora_alpha": lora_alpha,
        "params": n_params,
        "steps": step,
        "train_time_s": round(train_time, 1),
        "ms_per_step": round(train_time / step * 1000, 1) if step else 0,
        "early_stopped": early_stop,
        "metrics": metrics,
    }


def write_report(results, args):
    """Write JSON aggregate and markdown summary."""
    agg_path = args.out.replace(".json", "_aggregate.json")
    with open(agg_path, "w") as f:
        json.dump(results, f, indent=2)

    lines = [
        "# LLM Benchmark: LoRA Rank Sweep vs SVD",
        "",
        f"Preset `{args.preset}`, {args.tokens:,} tokens, seq_len {args.seq_len}, "
        f"batch {args.batch}, lr {args.lr}, max_steps {args.max_steps or 'all'}.",
        f"Early stop patience: {args.early_stop_patience or 'disabled'}.",
        "",
        "All LoRA configs use `alpha=rank` (scale=1.0) except `lora_r4_a1` (alpha=1, scale=0.25).",
        "Lower ppl is better; higher acc is better.",
        "",
        "| Tag | Variant | LLN | Param | Rank | α | Steps | ms/step | Train loss | Wiki ppl | LMB ppl | LMB acc | Time (s) |",
        "|-----|---------|-----|------:|-----:|--:|------:|--------:|-----------:|---------:|--------:|--------:|---------:|",
    ]
    for r in results:
        m = r["metrics"]
        lines.append(
            f"| {r['tag']} | {r['variant']} | {r['lln']} | {r['params']:,} | "
            f"{r['rank']} | {r['lora_alpha']} | {r['steps']} | {r['ms_per_step']} | "
            f"{fmt(m.get('train_loss_final'))} | {fmt(m.get('wiki_ppl'))} | "
            f"{fmt(m.get('lmb_ppl'))} | {fmt(m.get('lmb_acc'))} | "
            f"{r['train_time_s']} |"
        )

    # Speed ranking
    lines += ["", "### Speed ranking (ms/step, lower is better)", ""]
    speed_sorted = sorted(results, key=lambda r: r["ms_per_step"])
    for i, r in enumerate(speed_sorted, 1):
        lines.append(
            f"{i}. **{r['tag']}** — {r['ms_per_step']} ms/step ({r['params']:,} params, "
            f"{r['parameterization']} r={r['rank']})"
        )

    # Quality ranking by LMB ppl
    valid = [
        r
        for r in results
        if r["metrics"].get("lmb_ppl") and not math.isnan(r["metrics"]["lmb_ppl"])
    ]
    if valid:
        lines += ["", "### Quality ranking (LAMBADA ppl, lower is better)", ""]
        qual_sorted = sorted(valid, key=lambda r: r["metrics"]["lmb_ppl"])
        for i, r in enumerate(qual_sorted, 1):
            lines.append(
                f"{i}. **{r['tag']}** — LMB ppl {r['metrics']['lmb_ppl']:.1f} "
                f"({r['parameterization']} r={r['rank']}, α={r['lora_alpha']})"
            )

    # Quality ranking by Wiki ppl
    wiki_valid = [
        r
        for r in results
        if r["metrics"].get("wiki_ppl") and not math.isnan(r["metrics"]["wiki_ppl"])
    ]
    if wiki_valid:
        lines += ["", "### Quality ranking (Wiki ppl, lower is better)", ""]
        wiki_sorted = sorted(wiki_valid, key=lambda r: r["metrics"]["wiki_ppl"])
        for i, r in enumerate(wiki_sorted, 1):
            lines.append(
                f"{i}. **{r['tag']}** — Wiki ppl {r['metrics']['wiki_ppl']:.1f} "
                f"({r['parameterization']} r={r['rank']}, α={r['lora_alpha']})"
            )

    # Efficiency ranking (quality per param)
    if valid:
        lines += [
            "",
            "### Efficiency ranking (LMB ppl / params, lower = better quality per param)",
            "",
        ]
        eff_sorted = sorted(valid, key=lambda r: r["metrics"]["lmb_ppl"] / r["params"])
        for i, r in enumerate(eff_sorted, 1):
            ratio = r["metrics"]["lmb_ppl"] / r["params"]
            lines.append(
                f"{i}. **{r['tag']}** — {ratio:.4f} ppl/param "
                f"({r['params']:,} params, LMB {r['metrics']['lmb_ppl']:.1f})"
            )

    # LoRA rank scaling analysis
    lora_runs = [
        r for r in results if r["parameterization"] == "lora" and r["lora_alpha"] == r["rank"]
    ]
    if lora_runs:
        lines += ["", "### LoRA rank scaling (alpha=rank, scale=1.0)", ""]
        lines.append("| Rank | Params | ms/step | LMB ppl | Wiki ppl | Train loss |")
        lines.append("|-----:|-------:|--------:|--------:|---------:|-----------:|")
        for r in sorted(lora_runs, key=lambda x: x["rank"]):
            m = r["metrics"]
            lines.append(
                f"| {r['rank']} | {r['params']:,} | {r['ms_per_step']} | "
                f"{fmt(m.get('lmb_ppl'))} | {fmt(m.get('wiki_ppl'))} | "
                f"{fmt(m.get('train_loss_final'))} |"
            )

    # Alpha scaling analysis (rank=4, varying alpha)
    alpha_runs = [r for r in results if r["parameterization"] == "lora" and r["rank"] == 4]
    if len(alpha_runs) > 1:
        lines += ["", "### Alpha scaling (rank=4, varying alpha)", ""]
        lines.append("| α | Scale (α/r) | Params | LMB ppl | Wiki ppl | Train loss |")
        lines.append("|--:|------------:|-------:|--------:|---------:|-----------:|")
        for r in sorted(alpha_runs, key=lambda x: x["lora_alpha"]):
            m = r["metrics"]
            scale = r["lora_alpha"] / r["rank"]
            lines.append(
                f"| {r['lora_alpha']} | {scale:.2f} | {r['params']:,} | "
                f"{fmt(m.get('lmb_ppl'))} | {fmt(m.get('wiki_ppl'))} | "
                f"{fmt(m.get('train_loss_final'))} |"
            )

    lines += ["", f"Total wall time: {sum(r['train_time_s'] for r in results):.0f}s"]

    md_path = args.out.replace(".json", ".md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n[report] {md_path}")
    print(f"[aggregate] {agg_path}")


def main():
    p = argparse.ArgumentParser(description="LLM benchmark: LoRA rank sweep vs SVD")
    p.add_argument("--preset", default="tiny", choices=["tiny", "scaled", "small", "medium"])
    p.add_argument("--dataset", default="bhavnicksm/fineweb-edu-micro")
    p.add_argument("--tokens", type=int, default=200_000)
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--device", default="cpu")
    p.add_argument("--eval", action="store_true", default=True)
    p.add_argument("--no_eval", dest="eval", action="store_false")
    p.add_argument("--wiki_tokens", type=int, default=8000)
    p.add_argument("--lambada_ex", type=int, default=80)
    p.add_argument(
        "--early_stop_patience",
        type=int,
        default=50,
        help="Stop if no improvement for N steps (0 = disabled)",
    )
    p.add_argument("--out", default="benchmarks/llm_lora_rank_sweep.json")
    p.add_argument("--ckpt_dir", default="benchmarks/ckpts")
    p.add_argument("--skip_baseline", action="store_true")
    a = p.parse_args()

    os.makedirs(a.ckpt_dir, exist_ok=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    _require_internet()
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.model_max_length = int(1e30)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[data] streaming {a.dataset} up to {a.tokens} tokens ...")
    ids = build_token_buffer(tok, a.tokens, a.dataset, a.seq_len)
    print(f"[data] collected {len(ids):,} tokens")
    data = PackedTokens(ids, a.seq_len)
    a.loader = DataLoader(data, batch_size=a.batch, shuffle=True, drop_last=True)

    # Build run list: (variant, lln_name, parameterization, rank, lora_alpha)
    runs = []
    if not a.skip_baseline:
        runs.append(("baseline", None, "svd", 4, 1.0))
    # LoRA ranks with alpha=rank (scale=1.0)
    for rank in [1, 2, 4, 8, 16]:
        runs.append(("ours", "StableLiquidLN", "lora", rank, float(rank)))
    # LoRA rank=4 with alpha=1 (default, scale=0.25)
    runs.append(("ours", "StableLiquidLN", "lora", 4, 1.0))
    # SVD rank=4
    runs.append(("ours", "StableLiquidLN", "svd", 4, 1.0))

    total_runs = len(runs)
    print(f"\n[plan] {total_runs} configurations")
    for v, l, p, r, a_val in runs:
        label = l or "baseline"
        print(f"  - {v} | {label} | {p} | rank={r} | alpha={a_val}")

    results = []
    t_start = time.perf_counter()
    for i, (variant, lln_name, parameterization, rank, lora_alpha) in enumerate(runs):
        print(f"\n{'#' * 64}")
        print(f"# RUN {i + 1}/{total_runs}")
        print(f"{'#' * 64}")
        r = train_one(a, variant, lln_name, parameterization, a.device, tok, lora_alpha, rank)
        results.append(r)

    total_time = time.perf_counter() - t_start
    print(f"\n{'=' * 64}")
    print(f"ALL DONE: {total_runs} runs in {total_time:.0f}s ({total_time / 60:.1f} min)")
    print(f"{'=' * 64}")

    write_report(results, a)


if __name__ == "__main__":
    main()
