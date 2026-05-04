#!/usr/bin/env python3
"""Train one SVG language model run."""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from tqdm import trange

from common import (
    SvgTokenizer,
    append_jsonl,
    checkpoint_path,
    cosine_lr,
    load_train_config,
    memmap_split,
    now,
    pick_device,
    set_seed,
    write_json,
)
from model import GPT

try:
    from mup import MuAdamW, set_base_shapes
except Exception:  # pragma: no cover
    MuAdamW = None
    set_base_shapes = None


def get_batch(data: np.memmap, batch_size: int, block_size: int, device: torch.device):
    if len(data) <= block_size + 1:
        raise ValueError(f"Encoded split is too small for block_size={block_size}")
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(np.asarray(data[i : i + block_size], dtype=np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(np.asarray(data[i + 1 : i + block_size + 1], dtype=np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model: GPT, data: np.memmap, cfg, device: torch.device) -> float:
    model.eval()
    losses = []
    for _ in range(cfg.eval_iters):
        x, y = get_batch(data, cfg.batch_size, cfg.block_size, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def gpu_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024**2)
    return 0.0


def width_for_heads(width: int, n_heads: int) -> int:
    return max(n_heads, int(math.ceil(width / n_heads)) * n_heads)


def mup_shape_config(cfg, width: int):
    d_model = width_for_heads(width, cfg.n_heads)
    d_ff = max(cfg.n_heads, int(round(cfg.d_ff * d_model / cfg.d_model)))
    return replace(cfg, d_model=d_model, d_ff=d_ff, parameterization="mup", compile=False)


def configure_mup(model: GPT, cfg, vocab_size: int) -> None:
    if cfg.parameterization != "mup":
        return
    if set_base_shapes is None or MuAdamW is None:
        raise ImportError("parameterization='mup' requires the mup package. Install it with `pip install mup`.")
    base = GPT(mup_shape_config(cfg, cfg.mup_base_width), vocab_size)
    delta = GPT(mup_shape_config(cfg, cfg.mup_delta_width), vocab_size)
    set_base_shapes(model, base, delta=delta)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--tokenizer_path", default="data/tokenizer/tokenizer.json")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override config LR for sweeps.")
    parser.add_argument("--max_steps", type=int, default=None, help="Override config or one-epoch step count.")
    parser.add_argument("--parameterization", choices=["sp", "mup"], default=None, help="Override config parameterization.")
    parser.add_argument("--mup_base_width", type=int, default=None)
    parser.add_argument("--mup_delta_width", type=int, default=None)
    args = parser.parse_args()

    cfg = load_train_config(args.config)
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
        cfg.min_lr = min(cfg.min_lr, args.learning_rate / 10.0)
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.parameterization is not None:
        cfg.parameterization = args.parameterization
    if args.mup_base_width is not None:
        cfg.mup_base_width = args.mup_base_width
    if args.mup_delta_width is not None:
        cfg.mup_delta_width = args.mup_delta_width

    out_dir = Path(args.out_dir or f"outputs/runs/{cfg.name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    device = pick_device(args.device)
    tokenizer = SvgTokenizer(args.tokenizer_path)
    train_data = memmap_split(args.data_dir, "train")
    val_data = memmap_split(args.data_dir, "val")
    inferred_epoch_steps = max(1, len(train_data) // cfg.tokens_per_optimizer_step)
    total_steps = cfg.max_steps or inferred_epoch_steps

    model = GPT(cfg, tokenizer.vocab_size)
    configure_mup(model, cfg, tokenizer.vocab_size)
    model = model.to(device)
    if cfg.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    optimizer_cls = MuAdamW if cfg.parameterization == "mup" else torch.optim.AdamW
    optimizer = optimizer_cls(
        raw_model.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )
    use_amp = device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)
    amp_device = "cuda" if device.type == "cuda" else "cpu"

    metadata = {
        "started_at": now(),
        "config": asdict(cfg),
        "tokenizer_path": args.tokenizer_path,
        "data_dir": args.data_dir,
        "device": str(device),
        "vocab_size": tokenizer.vocab_size,
        "parameters": raw_model.parameter_count(),
        "train_tokens_available": int(len(train_data)),
        "val_tokens_available": int(len(val_data)),
        "epoch_steps_estimate": int(inferred_epoch_steps),
        "total_steps": int(total_steps),
    }
    write_json(out_dir / "metadata.json", metadata)
    log_path = out_dir / "metrics.jsonl"
    if log_path.exists():
        log_path.unlink()

    best_val = float("inf")
    start = time.time()
    tokens_seen = 0
    for step in trange(total_steps, desc=f"training {cfg.name}"):
        lr = cosine_lr(step, cfg, total_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(cfg.gradient_accumulation_steps):
            x, y = get_batch(train_data, cfg.batch_size, cfg.block_size, device)
            with autocast(device_type=amp_device, dtype=torch.float16, enabled=use_amp):
                _, loss = model(x, y)
                loss = loss / cfg.gradient_accumulation_steps
            scaler.scale(loss).backward()
            step_loss += loss.item()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        tokens_seen += cfg.tokens_per_optimizer_step

        should_eval = step == 0 or (step + 1) % cfg.eval_interval == 0 or step + 1 == total_steps
        if should_eval:
            val_loss = estimate_loss(raw_model, val_data, cfg, device)
            elapsed = time.time() - start
            record = {
                "step": step + 1,
                "train_loss": step_loss,
                "val_loss": val_loss,
                "lr": lr,
                "tokens_seen": tokens_seen,
                "tokens_per_second": tokens_seen / max(elapsed, 1e-9),
                "elapsed_seconds": elapsed,
                "gpu_memory_mb": gpu_memory_mb(device),
            }
            append_jsonl(log_path, record)
            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {
                        "model_state": raw_model.state_dict(),
                        "config": asdict(cfg),
                        "vocab_size": tokenizer.vocab_size,
                        "tokenizer_path": args.tokenizer_path,
                        "step": step + 1,
                        "val_loss": val_loss,
                        "parameters": raw_model.parameter_count(),
                    },
                    checkpoint_path(out_dir),
                )

    summary = {
        **metadata,
        "finished_at": now(),
        "best_val_loss": best_val,
        "best_val_perplexity": math.exp(best_val) if best_val < 20 else float("inf"),
        "elapsed_seconds": time.time() - start,
        "tokens_seen": tokens_seen,
        "tokens_per_second": tokens_seen / max(time.time() - start, 1e-9),
        "gpu_memory_mb": gpu_memory_mb(device),
        "checkpoint": str(checkpoint_path(out_dir)),
    }
    write_json(out_dir / "summary.json", summary)
    print(f"Finished {cfg.name}: best val loss={best_val:.4f}, checkpoint={checkpoint_path(out_dir)}")


if __name__ == "__main__":
    main()
