#!/usr/bin/env python3
"""Plot LR sweep and training curves from run directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def plot_lr_sweep(run_dirs: list[Path], out_dir: Path) -> None:
    rows = []
    for run in run_dirs:
        summary = read_json(run / "summary.json")
        rows.append((summary["config"]["learning_rate"], summary["best_val_loss"], summary["config"]["name"], run.name))
    rows.sort(key=lambda x: x[0])
    plt.figure(figsize=(6.2, 4.0))
    plt.plot([r[0] for r in rows], [r[1] for r in rows], marker="o")
    for lr, loss, _, name in rows:
        plt.annotate(name, (lr, loss), textcoords="offset points", xytext=(4, 4), fontsize=8)
    plt.xscale("log")
    plt.xlabel("Learning rate")
    plt.ylabel("Best validation loss")
    plt.title("Tiny LR Sweep")
    plt.tight_layout()
    plt.savefig(out_dir / "lr_sweep.png", dpi=200)


def plot_training_curves(run_dirs: list[Path], out_dir: Path) -> None:
    plt.figure(figsize=(7.0, 4.5))
    for run in run_dirs:
        rows = read_jsonl(run / "metrics.jsonl")
        if not rows:
            continue
        label = read_json(run / "summary.json")["config"]["name"]
        plt.plot([r["tokens_seen"] for r in rows], [r["val_loss"] for r in rows], marker="o", linewidth=1.5, label=label)
    plt.xlabel("Tokens seen")
    plt.ylabel("Validation loss")
    plt.title("Validation Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "validation_curves.png", dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out_dir", default="outputs/analysis")
    parser.add_argument("--mode", choices=["curves", "lr_sweep"], default="curves")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = [Path(r) for r in args.runs]
    if args.mode == "lr_sweep":
        plot_lr_sweep(run_dirs, out_dir)
    else:
        plot_training_curves(run_dirs, out_dir)
    print(f"Wrote {args.mode} plot to {out_dir}")


if __name__ == "__main__":
    main()
