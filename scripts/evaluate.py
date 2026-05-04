#!/usr/bin/env python3
"""Evaluate checkpoints and generated SVG samples."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from lxml import etree


def add_windows_cairo_path() -> None:
    if os.name != "nt":
        return
    candidates = [
        Path(r"C:\Program Files\GTK3-Runtime Win64\bin"),
        Path(r"C:\Program Files (x86)\GTK3-Runtime Win64\bin"),
    ]
    for path in candidates:
        if path.exists():
            os.environ["PATH"] = f"{path}{os.pathsep}{os.environ.get('PATH', '')}"
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(path))


add_windows_cairo_path()

try:
    import cairosvg
except Exception:  # pragma: no cover
    cairosvg = None

from common import SvgTokenizer, TrainConfig, memmap_split, pick_device, safe_torch_load, write_json
from model import GPT
from train import configure_mup, estimate_loss


NUMERIC_ATTRS = {
    "x",
    "y",
    "x1",
    "y1",
    "x2",
    "y2",
    "cx",
    "cy",
    "r",
    "rx",
    "ry",
    "width",
    "height",
    "opacity",
    "stroke-width",
}
NUMERIC_VALUE_RE = re.compile(r"^\s*-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?:px|%)?\s*$")


def xml_valid(svg: str) -> bool:
    try:
        root = etree.fromstring(svg.encode("utf-8"))
        return etree.QName(root).localname == "svg"
    except Exception:
        return False


def structural_valid(svg: str) -> bool:
    try:
        root = etree.fromstring(svg.encode("utf-8"))
    except Exception:
        return False
    if etree.QName(root).localname != "svg":
        return False
    for node in root.iter():
        if not isinstance(node.tag, str):
            continue
        for key, value in node.attrib.items():
            local = etree.QName(key).localname
            if local in NUMERIC_ATTRS and not NUMERIC_VALUE_RE.match(value):
                return False
            if local in {"fill", "stroke"} and value.strip() == "":
                return False
    return True


def render_valid(svg: str) -> bool:
    if cairosvg is None:
        return False
    try:
        cairosvg.svg2png(bytestring=svg.encode("utf-8"), output_width=128, output_height=128)
        return True
    except Exception:
        return False


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = safe_torch_load(ckpt_path, map_location=device)
    cfg = TrainConfig.from_dict(ckpt["config"])
    model = GPT(cfg, ckpt["vocab_size"])
    configure_mup(model, cfg, ckpt["vocab_size"])
    model = model.to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg, ckpt


def evaluate_samples(samples_path: Path) -> dict:
    rows = []
    xml_ok = 0
    render_ok = 0
    structural_ok = 0
    with open(samples_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            svg = row["svg"]
            is_xml = xml_valid(svg)
            is_structural = structural_valid(svg) if is_xml else False
            is_render = render_valid(svg) if is_xml else False
            row["xml_valid"] = is_xml
            row["structural_valid"] = is_structural
            row["render_valid"] = is_render
            rows.append(row)
            xml_ok += int(is_xml)
            structural_ok += int(is_structural)
            render_ok += int(is_render)
    annotated = samples_path.with_name(samples_path.stem + "_evaluated.jsonl")
    with open(annotated, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    n = len(rows)
    return {
        "samples": n,
        "xml_valid": xml_ok,
        "render_valid": render_ok,
        "structural_valid": structural_ok,
        "xml_valid_rate": xml_ok / n if n else 0.0,
        "render_valid_rate": render_ok / n if n else 0.0,
        "structural_valid_rate": structural_ok / n if n else 0.0,
        "annotated_samples": str(annotated),
        "cairosvg_available": cairosvg is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--samples", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = pick_device(args.device)
    model, cfg, ckpt = load_model(Path(args.ckpt), device)
    test_data = memmap_split(args.data_dir, "test")
    test_loss = estimate_loss(model, test_data, cfg, device)
    metrics = {
        "checkpoint": args.ckpt,
        "config": asdict(cfg),
        "parameters": ckpt["parameters"],
        "test_loss": test_loss,
        "test_perplexity": math.exp(test_loss) if test_loss < 20 else float("inf"),
        "samples": evaluate_samples(Path(args.samples)) if args.samples else None,
    }
    out = Path(args.out) if args.out else Path(args.ckpt).parent / "evaluation.json"
    write_json(out, metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
