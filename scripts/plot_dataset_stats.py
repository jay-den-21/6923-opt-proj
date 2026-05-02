#!/usr/bin/env python3
"""Create dataset histograms and rendered SVG example sheets."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import read_jsonl


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_histogram(stats: dict, out_dir: Path) -> None:
    plt.figure(figsize=(7.0, 4.2))
    for split, split_stats in stats["splits"].items():
        hist = split_stats.get("length_histogram")
        if hist:
            edges = hist["bin_edges"]
            counts = hist["counts"]
            centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(counts))]
            widths = [edges[i + 1] - edges[i] for i in range(len(counts))]
            plt.bar(centers, counts, width=widths, alpha=0.45, label=split, align="center")
        else:
            lengths = [
                split_stats.get("length_min", 0),
                split_stats.get("length_p50", 0),
                split_stats.get("length_p95", 0),
                split_stats.get("length_max", 0),
            ]
            plt.plot(lengths, [0, split_stats.get("files", 0) / 2, split_stats.get("files", 0) / 20, 0], marker="o", label=split)
    plt.xlabel("Tokenized SVG length")
    plt.ylabel("Number of files")
    plt.title("SVG Sequence Length Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "sequence_length_histogram.png", dpi=200)


def pick_examples(rows: list[dict], count: int) -> list[dict]:
    rows = sorted((r for r in rows if "token_length" in r), key=lambda r: r["token_length"])
    if not rows:
        return []
    if count <= 1:
        return [rows[len(rows) // 2]]
    indices = [round(i * (len(rows) - 1) / (count - 1)) for i in range(count)]
    return [rows[i] for i in indices]


def write_examples(rows: list[dict], out_dir: Path) -> None:
    cards = []
    for row in rows:
        svg = row["svg"]
        label = f"{row.get('source', 'unknown')} | {row.get('token_length', '?')} tokens"
        cards.append(
            "<article>"
            f"<div class='preview'>{svg}</div>"
            f"<p>{html.escape(label)}</p>"
            "<details><summary>SVG code</summary>"
            f"<pre>{html.escape(svg[:2000])}</pre>"
            "</details>"
            "</article>"
        )
    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>SVG dataset examples</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #1f2937; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; }}
    article {{ border: 1px solid #d1d5db; border-radius: 8px; padding: 12px; background: #fff; }}
    .preview {{ height: 160px; display: grid; place-items: center; background: #f9fafb; overflow: hidden; }}
    .preview svg {{ max-width: 140px; max-height: 140px; }}
    p {{ font-size: 13px; margin: 10px 0; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; font-size: 11px; }}
  </style>
</head>
<body>
  <h1>SVG Examples by Token Length</h1>
  <div class="grid">
    {''.join(cards)}
  </div>
</body>
</html>
"""
    (out_dir / "dataset_examples.html").write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--out_dir", default="outputs/analysis")
    parser.add_argument("--examples", type=int, default=12)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = read_json(data_dir / "tokenizer_stats.json")
    plot_histogram(stats, out_dir)
    rows = list(read_jsonl(data_dir / "train_filtered.jsonl"))
    write_examples(pick_examples(rows, args.examples), out_dir)
    print(f"Wrote dataset plots/examples to {out_dir}")


if __name__ == "__main__":
    main()
