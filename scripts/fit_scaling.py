#!/usr/bin/env python3
"""Fit and plot validation-loss scaling curves from run summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.optimize import curve_fit
except Exception:  # pragma: no cover
    curve_fit = None


def power_law(n: np.ndarray, a: float, alpha: float, c: float) -> np.ndarray:
    return a * np.power(n, -alpha) + c


def fit_power_law(params: np.ndarray, losses: np.ndarray) -> dict:
    if curve_fit is not None and len(params) >= 4:
        try:
            a0 = max(float(abs(losses[0] - losses[-1]) * params[0] ** 0.1), 1e-6)
            p0 = [a0, 0.1, float(losses.min() * 0.95)]
            bounds = ([0.0, 0.0, 0.0], [1000.0, 5.0, float(losses.min() * 0.999)])
            popt, pcov = curve_fit(power_law, params, losses, p0=p0, bounds=bounds, maxfev=10000)
            pred = power_law(params, *popt)
            stderr = np.sqrt(np.diag(pcov)).tolist()
            return {
                "method": "scipy_curve_fit",
                "a": float(popt[0]),
                "alpha": float(popt[1]),
                "c": float(popt[2]),
                "stderr": stderr,
                "rmse": float(np.sqrt(np.mean((pred - losses) ** 2))),
            }
        except Exception as exc:
            scipy_error = repr(exc)

    best = None
    for c in np.linspace(0.0, float(losses.min() * 0.95), 200):
        y = losses - c
        if np.any(y <= 0):
            continue
        slope, intercept = np.polyfit(np.log(params), np.log(y), 1)
        alpha = -slope
        a = np.exp(intercept)
        pred = power_law(params, a, alpha, c)
        rmse = float(np.sqrt(np.mean((pred - losses) ** 2)))
        if best is None or rmse < best["rmse"]:
            best = {"method": "grid_log_linear", "a": float(a), "alpha": float(alpha), "c": float(c), "rmse": rmse}
    if best is None:
        raise RuntimeError("Could not fit scaling law")
    if "scipy_error" in locals():
        best["scipy_error"] = scipy_error
    return best


def read_summary(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="Run dirs or summary.json files.")
    parser.add_argument("--out_dir", default="outputs/analysis")
    parser.add_argument("--predict_multiplier", type=float, default=10.0)
    args = parser.parse_args()

    rows = []
    for item in args.runs:
        path = Path(item)
        summary_path = path / "summary.json" if path.is_dir() else path
        summary = read_summary(summary_path)
        rows.append(
            {
                "name": summary["config"]["name"],
                "parameters": summary["parameters"],
                "val_loss": summary["best_val_loss"],
                "tokens_per_second": summary.get("tokens_per_second"),
                "gpu_memory_mb": summary.get("gpu_memory_mb"),
                "elapsed_seconds": summary.get("elapsed_seconds"),
            }
        )
    rows = sorted(rows, key=lambda r: r["parameters"])
    params = np.asarray([r["parameters"] for r in rows], dtype=np.float64)
    losses = np.asarray([r["val_loss"] for r in rows], dtype=np.float64)
    fit = fit_power_law(params, losses)
    largest = float(params.max())
    predicted_n = largest * args.predict_multiplier
    predicted_loss = float(power_law(np.asarray([predicted_n]), fit["a"], fit["alpha"], fit["c"])[0])
    fit["prediction"] = {"parameters": predicted_n, "loss": predicted_loss, "multiplier": args.predict_multiplier}
    fit["runs"] = rows

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "scaling_fit.json", "w", encoding="utf-8") as f:
        json.dump(fit, f, indent=2, sort_keys=True)
        f.write("\n")

    xs = np.logspace(np.log10(params.min()), np.log10(predicted_n), 200)
    ys = power_law(xs, fit["a"], fit["alpha"], fit["c"])
    plt.figure(figsize=(6.5, 4.2))
    plt.scatter(params, losses, label="observed")
    for row in rows:
        plt.annotate(row["name"], (row["parameters"], row["val_loss"]), textcoords="offset points", xytext=(4, 4))
    plt.plot(xs, ys, label=f"fit alpha={fit['alpha']:.3f}")
    plt.scatter([predicted_n], [predicted_loss], marker="x", s=80, label=f"{args.predict_multiplier:g}x extrapolation")
    plt.xscale("log")
    plt.xlabel("Parameters")
    plt.ylabel("Validation loss")
    plt.title("SVG LM Scaling Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "scaling_plot.png", dpi=200)
    print(json.dumps(fit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
