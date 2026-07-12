#!/usr/bin/env python
r"""LLM-scale training + evaluation for the Liquid / GDN-2 LLM variants.

Compares two architectures at a *shared* parameter budget:

* ``ours``     -- :class:`LiquidGDNCondLLM` (SWA as X, GDN-2 as cond,
                 StableLiquidLN intermediary MLP, SwiGLU FFN).
* ``baseline`` -- :class:`GDN2BaselineLLM` (lit_gpt-style GDN-2 mixer, no
                 attention, SwiGLU FFN) built on the same CPU gdn2.

Training data: a subset of FineWeb-Edu (``bhavnicksm/fineweb-edu-micro``,
~1M tokens). Evaluation reports the metrics requested for the comparison:

    Wiki ppl  (wikitext-2-raw-v1, lower is better)
    LMB ppl   (LAMBADA, lower is better)
    LMB acc   (LAMBADA last-token accuracy, higher is better)
    Avg acc   (mean of the accuracy metrics; here = LMB acc, single-task proxy)

Usage
-----
    python scripts/train_llm.py --variant ours --preset small --tokens 1000000
    python scripts/train_llm.py --variant baseline --preset small --tokens 1000000

The default preset (``small``) is sized to train both models within ~2h on a
weak laptop CPU; ``medium`` / ``0.5B`` are provided for GPU-class hardware.
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
    """Exit with a clear message if HuggingFace (datasets/tokenizers) is unreachable."""
    import socket

    host = "huggingface.co"
    try:
        socket.create_connection((host, 443), timeout=8)
    except OSError as e:
        sys.exit(
            f"[error] This benchmark requires an internet connection to download the "
            f"GPT-2 tokenizer, the FineWeb-Edu training data, and the Wiki/LAMBADA "
            f"evaluation sets from {host}.\n"
            f"Could not reach {host}: {e}\n"
            f"Exiting."
        )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class PackedTokens(Dataset):
    def __init__(self, ids: list[int], seq_len: int) -> None:
        self.data = torch.tensor(ids[: (len(ids) // seq_len) * seq_len], dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.data) // self.seq_len

    def __getitem__(self, i: int) -> torch.Tensor:
        s = i * self.seq_len
        return self.data[s : s + self.seq_len]


def build_token_buffer(tok, tokens: int, dataset: str, seq_len: int) -> list[int]:
    """Stream a FineWeb-Edu subset, tokenize, and collect ``tokens`` ids."""
    from datasets import load_dataset

    ids: list[int] = []
    ds = load_dataset(dataset, split="train", streaming=True)
    for ex in ds:
        text = ex.get("text") or ""
        if not text:
            continue
        ids.extend(tok(text, add_special_tokens=False).input_ids)
        if len(ids) >= tokens:
            break
    return ids[:tokens]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _wiki_ppl(model: torch.nn.Module, tok, device: str, seq_len: int, max_tokens: int = 50_000) -> float:
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids: list[int] = []
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


def _lambada_eval(model: torch.nn.Module, tok, device: str, seq_len: int, max_ex: int = 300):
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
            ctx = ids[:-1][-(seq_len - 1):]   # cap context to the training window
            last = ids[-1]
            x = torch.tensor(ctx, device=device).unsqueeze(0)
            logits = model(x)[0, -1]
            pred = int(logits.argmax().item())
            correct += int(pred == last)
            total += 1
            loss_sum += F.cross_entropy(logits.unsqueeze(0), torch.tensor([last], device=device)).item()
            n_tok += 1
            if total >= max_ex:
                break
    acc = correct / total if total else float("nan")
    ppl = math.exp(loss_sum / n_tok) if n_tok else float("nan")
    return ppl, acc


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def _run_eval(args, model, tok, device, metrics: dict) -> None:
    """Run Wiki + LAMBADA eval in place. Failures propagate (no silent fallback)."""
    if not args.eval:
        return
    print("[eval] Wiki ppl ...")
    metrics["wiki_ppl"] = _wiki_ppl(model, tok, device, args.seq_len)
    print("[eval] LAMBADA ...")
    lmb_ppl, lmb_acc = _lambada_eval(model, tok, device, args.seq_len)
    metrics["lmb_ppl"] = lmb_ppl
    metrics["lmb_acc"] = lmb_acc
    metrics["avg_acc"] = lmb_acc  # single-task proxy


def _write_result(args, variant, preset, n_params, step, train_time, metrics, ckpt_path) -> dict:
    result = dict(
        variant=variant, preset=preset, params=n_params,
        tokens=args.tokens, steps=step, train_time_s=round(train_time, 1),
        seq_len=args.seq_len, lr=args.lr, ckpt=ckpt_path, metrics=metrics,
    )
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2))
    return result


def _suffixed_path(path: str, suffix: str) -> str:
    """Insert ``suffix`` before the file extension (e.g. foo.json -> foo_lora_0.json)."""
    root, ext = os.path.splitext(path)
    return f"{root}_{suffix}{ext}"


def run_benchmark(a) -> None:
    """Run the LLM benchmark. In 'mixed' mode, trains LoRA 3x and SVD 1x (3:1)."""
    if a.parameterization != "mixed":
        train(a)
        return

    modes = ["svd", "svd", "svd", "lora"]  # 3:1 SVD:LoRA
    aggregate: list[dict] = []
    for i, mode in enumerate(modes):
        run_args = argparse.Namespace(**vars(a))
        run_args.parameterization = mode
        run_args.out = _suffixed_path(a.out, f"{mode}_{i}")
        run_args.ckpt = (
            _suffixed_path(a.ckpt, f"{mode}_{i}")
            if a.ckpt
            else f"llm_{a.variant}_{a.preset}_{mode}_{i}_ckpt.pt"
        )
        print(f"\n{'='*60}\n=== MIXED RUN {i+1}/4: parameterization={mode} ===\n{'='*60}")
        aggregate.append(train(run_args))

    agg_path = _suffixed_path(a.out, "aggregate")
    with open(agg_path, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\n[Aggregate] {len(aggregate)} runs (3 LoRA + 1 SVD) -> {agg_path}")


def train(args) -> dict:
    _require_internet()
    device = args.device
    ckpt_path = args.ckpt or f"llm_{args.variant}_{args.preset}_ckpt.pt"
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.model_max_length = int(1e30)  # Silence sequence length warning
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if args.eval_only:
        if not os.path.exists(ckpt_path):
            sys.exit(f"[error] --eval_only set but checkpoint not found: {ckpt_path}")
        print(f"[eval_only] loading checkpoint {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model = build_llm(ckpt["variant"], ckpt["preset"], parameterization=args.parameterization, **json.loads(args.overrides or "{}")).to(device)
        model.load_state_dict(ckpt["state"])
        n_params = num_params(model)
        print(f"[model] loaded {ckpt['variant']} @ preset={ckpt['preset']}: {n_params:,} params")
        metrics: dict[str, Optional[float]] = {}
        _run_eval(args, model, tok, device, metrics)
        return _write_result(args, ckpt["variant"], ckpt["preset"], n_params, 0, 0.0, metrics, ckpt_path)

    print(f"[data] streaming {args.dataset} up to {args.tokens} tokens (seq_len={args.seq_len}) ...")
    ids = build_token_buffer(tok, args.tokens, args.dataset, args.seq_len)
    print(f"[data] collected {len(ids):,} tokens")
    data = PackedTokens(ids, args.seq_len)
    loader = DataLoader(data, batch_size=args.batch, shuffle=True, drop_last=True)

    model = build_llm(args.variant, args.preset, parameterization=args.parameterization, **json.loads(args.overrides or "{}")).to(device)
    n_params = num_params(model)
    print(f"[model] {args.variant} @ preset={args.preset}: {n_params:,} params, {len(model.blocks)} layers")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    model.train()
    t0 = time.perf_counter()
    step = 0
    running_loss = 0.0
    early_stop = False
    for epoch in range(args.epochs):
        for x in loader:
            x = x.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))
            
            # Early stopping for runaway loss (nan/inf or explosion)
            if not torch.isfinite(loss).all():
                print(f"[train] step {step} loss {loss.item():.3f} -> unstable, stopping.")
                early_stop = True
                break
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += loss.item()
            step += 1
            if step % args.log_every == 0:
                print(f"[train] step {step} loss {running_loss / args.log_every:.4f}")
                running_loss = 0.0
            if args.max_steps and step >= args.max_steps:
                break
        if early_stop:
            break

    train_time = time.perf_counter() - t0

    torch.save({"variant": args.variant, "preset": args.preset, "state": model.state_dict()}, ckpt_path)
    print(f"[ckpt] saved {ckpt_path}")

    metrics: dict[str, Optional[float]] = {"train_loss_final": float(running_loss or loss.item())}
    _run_eval(args, model, tok, device, metrics)
    return _write_result(args, args.variant, args.preset, n_params, step, train_time, metrics, ckpt_path)


def main() -> None:
    p = argparse.ArgumentParser(description="Liquid / GDN-2 LLM benchmark")
    p.add_argument("--variant", choices=["ours", "baseline"], default="ours")
    p.add_argument("--preset", default="small", choices=["small", "medium", "0.5B"])
    p.add_argument("--dataset", default="bhavnicksm/fineweb-edu-micro")
    p.add_argument("--tokens", type=int, default=1_000_000)
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--block_size", type=int, default=1024)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=0, help="0 = train on all packed tokens once")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--device", default="cpu")
    p.add_argument("--eval", action="store_true", default=True)
    p.add_argument("--no_eval", dest="eval", action="store_false")
    p.add_argument("--parameterization", default="mixed", choices=["lora", "svd", "mixed"],
                   help="parameterization mode for low-rank updates (mixed = 3:1 lora:svd)")
    p.add_argument("--overrides", default="", help="JSON dict forwarded to LLMConfig")
    p.add_argument("--ckpt", default=None, help="checkpoint path; auto-derived (llm_<variant>_<preset>_ckpt.pt) if omitted")
    p.add_argument("--eval_only", action="store_true", help="load --ckpt and run eval only (skip training)")
    p.add_argument("--out", default="llm_bench_report.json")
    a = p.parse_args()
    run_benchmark(a)


if __name__ == "__main__":
    main()
