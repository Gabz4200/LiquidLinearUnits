#!/usr/bin/env python
r"""Scaled LLM benchmark: top 3 intermediary LLNs at n_embd=192, 4 layers.

Tests the best-performing SVD configs from the tiny-screen at a larger scale
where the LLU intermediary's inductive bias is more visible (shifting ~3M params
from the embedding bottleneck to the model body).
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

from llu.models.liquid_llm import build_llm, num_params, LLN_REGISTRY


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


def safe_load(model, state):
    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in state.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped.append(k)
    model.load_state_dict(filtered, strict=False)
    return skipped


def train_one(args, variant, lln_name, parameterization, device, tok, loader):
    lln_label = lln_name or "baseline"
    tag = f"{variant}_{lln_label}_p_{parameterization}"

    model = build_llm(
        variant,
        args.preset,
        lln=lln_name or "StableLiquidLN",
        parameterization=parameterization,
    ).to(device)
    n_params = num_params(model)

    print(f"\n{'=' * 60}")
    print(f"  {tag}  |  {n_params:,} params  |  {parameterization}")
    print(f"{'=' * 60}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    model.train()
    t0 = time.perf_counter()
    step = 0
    last_loss = float("nan")
    best_loss = float("inf")
    patience_counter = 0
    early_stop = False
    loss_history = []

    for epoch in range(args.epochs):
        for x in loader:
            x = x.to(device)
            logits = model(x)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1)
            )

            if not torch.isfinite(loss).all():
                print(f"  [step {step}] loss={loss.item():.3f} -> UNSTABLE")
                early_stop = True
                break

            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = loss.item()
            step += 1
            loss_history.append(last_loss)

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
    ms_per_step = train_time / step * 1000 if step else 0
    print(f"  [done] {step} steps in {train_time:.1f}s ({ms_per_step:.0f} ms/step)")

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
    return {
        "tag": tag,
        "variant": variant,
        "lln": lln_name or "-",
        "parameterization": parameterization,
        "params": n_params,
        "steps": step,
        "train_time_s": round(train_time, 1),
        "ms_per_step": round(ms_per_step, 1),
        "early_stopped": early_stop,
        "metrics": metrics,
    }


def write_report(results, args):
    lines = [
        "# LLM Benchmark: Scaled Intermediary LLN Comparison",
        "",
        f"Preset `{args.preset}` (n_embd=192, 4 layers ours / 8 layers baseline), "
        f"{args.tokens:,} tokens, seq_len {args.seq_len}, batch {args.batch}, "
        f"lr {args.lr}, max_steps {args.max_steps}.",
        f"Early stop patience: {args.early_stop_patience}.",
        "",
        "Lower ppl is better; higher acc is better.",
        "",
    ]

    # Main table sorted by LMB ppl
    valid = [
        r
        for r in results
        if r["metrics"].get("lmb_ppl") and not math.isnan(r["metrics"]["lmb_ppl"])
    ]
    sorted_results = sorted(valid, key=lambda r: r["metrics"]["lmb_ppl"]) if valid else results

    lines.append(
        "| # | LLN | Param | Steps | ms/step | Train loss (best) | Wiki ppl | LMB ppl | LMB acc | Time (s) |"
    )
    lines.append(
        "|---|-----|------:|------:|--------:|-------------------:|---------:|--------:|--------:|---------:|"
    )
    for i, r in enumerate(sorted_results, 1):
        m = r["metrics"]
        lines.append(
            f"| {i} | **{r['lln']}** ({r['parameterization']}) | {r['params']:,} | "
            f"{r['steps']} | {r['ms_per_step']:.0f} | "
            f"{fmt(m.get('train_loss_final'))} ({fmt(m.get('train_loss_best'))}) | "
            f"{fmt(m.get('wiki_ppl'))} | {fmt(m.get('lmb_ppl'))} | "
            f"{fmt(m.get('lmb_acc'))} | {r['train_time_s']:.0f} |"
        )

    lines += [
        "",
        f"Total wall time: {sum(r['train_time_s'] for r in results):.0f}s "
        f"({sum(r['train_time_s'] for r in results) / 60:.1f} min)",
    ]

    md_path = args.out.replace(".json", ".md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n[report] {md_path}")


def main():
    p = argparse.ArgumentParser(description="Scaled LLM benchmark: top intermediary LLNs")
    p.add_argument("--preset", default="scaled", help="preset name for labeling")
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
    p.add_argument("--wiki_tokens", type=int, default=10000)
    p.add_argument("--lambada_ex", type=int, default=100)
    p.add_argument("--early_stop_patience", type=int, default=75)
    p.add_argument("--out", default="benchmarks/llm_scaled_report.json")
    p.add_argument("--ckpt_dir", default="benchmarks/ckpts")
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
    loader = DataLoader(data, batch_size=a.batch, shuffle=True, drop_last=True)

    # Top 3 SVD configs from tiny screen + baseline
    configs = [
        ("baseline", None, "svd"),
        ("ours", "FactorizedBatchMomentumLiquidLN", "svd"),
        ("ours", "SharedMomentumLiquidLN", "svd"),
        ("ours", "FactorizedLiquidLN", "svd"),
        ("ours", "CrossAttnLoraLN", "svd"),
        ("ours", "StableLiquidLN", "svd"),
    ]

    print(f"\n[plan] {len(configs)} runs at scaled config (n_embd=192, 4 layers)")
    print(f"[plan] configs: {[(c[1] or 'baseline', c[2]) for c in configs]}")

    results = []
    t_start = time.perf_counter()
    for i, (variant, lln_name, param) in enumerate(configs):
        print(f"\n{'#' * 64}")
        print(f"# RUN {i + 1}/{len(configs)}")
        print(f"{'#' * 64}")
        r = train_one(a, variant, lln_name, param, a.device, tok, loader)
        results.append(r)

    total_time = time.perf_counter() - t_start
    print(f"\n{'=' * 64}")
    print(f"ALL DONE: {len(configs)} runs in {total_time:.0f}s ({total_time / 60:.1f} min)")
    print(f"{'=' * 64}")

    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[json] {a.out}")

    write_report(results, a)


if __name__ == "__main__":
    main()
