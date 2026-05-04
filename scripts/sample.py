#!/usr/bin/env python3
"""Generate SVG samples from a trained checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

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

from common import SvgTokenizer, pick_device, safe_torch_load
from evaluate import load_model


DEFAULT_PREFIXES = [
    '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="8"',
    '<svg viewBox="0 0 24 24"><path d="M4 12',
    '<svg viewBox="0 0 24 24"><g><rect x="4" y="4"',
    '<svg viewBox="0 0 24 24"><circle cx="8" cy="9" r="2"/><circle',
    '<svg viewBox="0 0 24 24"><path d="M6 18 C',
]


def render(svg: str, path: Path) -> bool:
    if cairosvg is None:
        return False
    try:
        cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=str(path), output_width=256, output_height=256)
        return True
    except Exception:
        return False


def maybe_close_svg(text: str) -> str:
    start = text.find("<svg")
    if start > 0:
        text = text[start:]
    if "</svg>" in text:
        text = text[: text.find("</svg>") + len("</svg>")]
    return text.strip()


def repair_svg(text: str) -> str:
    text = maybe_close_svg(text)
    parser = etree.XMLParser(recover=True, remove_comments=True, resolve_entities=False, no_network=True)
    try:
        root = etree.fromstring(text.encode("utf-8"), parser=parser)
    except Exception:
        if "<svg" in text and "</svg>" not in text:
            text = text + "</svg>"
            root = etree.fromstring(text.encode("utf-8"), parser=parser)
        else:
            raise
    if etree.QName(root).localname != "svg":
        raise ValueError("recovered root element is not svg")
    return etree.tostring(root, encoding="unicode", method="xml", pretty_print=False).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--unconditional", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperatures", default="0.5,0.8,1.0")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--repair", action="store_true", help="Attempt lxml recovery before writing/evaluating samples.")
    args = parser.parse_args()

    device = pick_device(args.device)
    ckpt = safe_torch_load(args.ckpt, map_location=device)
    tokenizer_path = args.tokenizer_path or ckpt["tokenizer_path"]
    tokenizer = SvgTokenizer(tokenizer_path)
    model, _, _ = load_model(Path(args.ckpt), device)
    out_dir = Path(args.out_dir or Path(args.ckpt).parent / "samples")
    svg_dir = out_dir / "svg"
    png_dir = out_dir / "png"
    svg_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    temperatures = [float(t) for t in args.temperatures.split(",") if t.strip()]

    jobs: list[tuple[str, str, float]] = []
    for i in range(args.unconditional):
        jobs.append(("unconditional", "<svg", temperatures[i % len(temperatures)]))
    for i, prefix in enumerate(DEFAULT_PREFIXES):
        jobs.append((f"prefix_{i}", prefix, temperatures[i % len(temperatures)]))

    for i, (kind, prompt, temperature) in enumerate(jobs):
        input_ids = tokenizer.encode(prompt, add_special=False)
        idx = torch.tensor([input_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(
                idx,
                max_new_tokens=args.max_new_tokens,
                temperature=temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                eos_id=tokenizer.eos_id,
            )
        raw_svg = maybe_close_svg(tokenizer.decode(out[0], skip_special=True))
        repaired = False
        svg = raw_svg
        if args.repair:
            try:
                repaired_svg = repair_svg(raw_svg)
                repaired = repaired_svg != raw_svg
                svg = repaired_svg
            except Exception:
                pass
        stem = f"{i:03d}_{kind}_t{temperature:g}"
        svg_path = svg_dir / f"{stem}.svg"
        png_path = png_dir / f"{stem}.png"
        svg_path.write_text(svg, encoding="utf-8")
        rendered = render(svg, png_path)
        rows.append(
            {
                "id": i,
                "kind": kind,
                "prompt": prompt,
                "temperature": temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "repaired": repaired,
                "raw_svg": raw_svg if args.repair else None,
                "svg": svg,
                "svg_path": str(svg_path),
                "png_path": str(png_path) if rendered else None,
                "rendered": rendered,
            }
        )

    samples_path = out_dir / "samples.jsonl"
    with open(samples_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"Wrote {len(rows)} samples to {samples_path}")


if __name__ == "__main__":
    main()
