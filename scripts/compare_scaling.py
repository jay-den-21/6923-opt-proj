#!/usr/bin/env python3
"""Plot standard-parameterization and muP scaling fits together."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def power_law(n: np.ndarray, fit: dict) -> np.ndarray:
    return fit["a"] * np.power(n, -fit["alpha"]) + fit["c"]


def add_curve(path: Path, label: str, color: str) -> None:
    fit = read_json(path)
    rows = fit["runs"]
    params = np.asarray([r["parameters"] for r in rows], dtype=np.float64)
    losses = np.asarray([r["val_loss"] for r in rows], dtype=np.float64)
    largest = max(float(params.max()), float(fit["prediction"]["parameters"]))
    xs = np.logspace(np.log10(params.min()), np.log10(largest), 200)
    plt.scatter(params, losses, color=color, label=f"{label} observed")
    plt.plot(xs, power_law(xs, fit), color=color, label=f"{label} fit alpha={fit['alpha']:.3f}")
    pred = fit["prediction"]
    plt.scatter([pred["parameters"]], [pred["loss"]], color=color, marker="x", s=80)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sp_fit", required=True)
    parser.add_argument("--mup_fit", required=True)
    parser.add_argument("--out", default="outputs/analysis/sp_vs_mup_scaling.png")
    args = parser.parse_args()

    missing = [path for path in [Path(args.sp_fit), Path(args.mup_fit)] if not path.exists()]
    if missing:
        raise SystemExit(
            "Missing scaling fit file(s): "
            + ", ".join(str(path) for path in missing)
            + ". Run the corresponding analysis stage first."
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.0, 4.5))
    add_curve(Path(args.sp_fit), "SP", "#2563eb")
    add_curve(Path(args.mup_fit), "muP", "#dc2626")
    plt.xscale("log")
    plt.xlabel("Parameters")
    plt.ylabel("Validation loss")
    plt.title("SP vs muP SVG LM Scaling")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    print(f"Wrote comparison plot to {out}")


if __name__ == "__main__":
    main()
