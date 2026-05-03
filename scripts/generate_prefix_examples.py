#!/usr/bin/env python3
"""Generate focused prefix-completion examples for the report."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import torch

from common import SvgTokenizer, pick_device, safe_torch_load
from evaluate import load_model, render_valid, structural_valid, xml_valid
from sample import maybe_close_svg, repair_svg, render


PREFIX_CASES = [
    {
        "name": "partial_face",
        "description": "Partial face: one eye is provided; the model may add related facial structure.",
        "prompt": '<svg viewBox="0 0 24 24"><circle cx="8" cy="9" r="2"/><circle',
    },
    {
        "name": "open_path",
        "description": "Open path: a path command is started and the model continues the geometry.",
        "prompt": '<svg viewBox="0 0 24 24"><path d="M4 12 C',
    },
    {
        "name": "group_shape",
        "description": "Group with one shape: one rectangle is provided and the model may add related shapes before closing the group.",
        "prompt": '<svg viewBox="0 0 24 24"><g><rect x="4" y="4" width="6" height="6"/>',
    },
    {
        "name": "center_circle",
        "description": "Simple centered shape: a circle starts the icon and the model chooses surrounding details.",
        "prompt": '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="6"',
    },
]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def make_html(rows: list[dict], out: Path, max_code_chars: int = 1800) -> None:
    cards = []
    for row in rows:
        flags = (
            f"XML {row['xml_valid']} | struct {row['structural_valid']} | "
            f"render {row['render_valid']} | repaired {row['repaired']}"
        )
        completion = row["svg"]
        if completion.startswith(row["prompt"]):
            completion = completion[len(row["prompt"]) :]
        cards.append(
            "<article>"
            f"<h2>{html.escape(row['case'])} t={row['temperature']}</h2>"
            f"<p class='desc'>{html.escape(row['description'])}</p>"
            f"<div class='preview'>{row['svg']}</div>"
            f"<p class='flags'>{html.escape(flags)}</p>"
            "<div class='cols'>"
            f"<section><h3>Prefix</h3><pre>{html.escape(row['prompt'])}</pre></section>"
            f"<section><h3>Completion / Full SVG</h3><pre>{html.escape(completion[:max_code_chars])}</pre></section>"
            "</div>"
            "</article>"
        )
    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Prefix Completion Examples</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #111827; }}
    h1 {{ margin-bottom: 6px; }}
    .note {{ color: #4b5563; max-width: 980px; line-height: 1.45; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; margin-top: 20px; }}
    article {{ border: 1px solid #d1d5db; border-radius: 8px; padding: 12px; background: white; }}
    h2 {{ font-size: 15px; margin: 0 0 6px; }}
    h3 {{ font-size: 12px; margin: 8px 0 4px; }}
    .desc, .flags {{ font-size: 12px; color: #4b5563; }}
    .preview {{ height: 190px; display: grid; place-items: center; background: #f9fafb; overflow: hidden; }}
    .preview svg {{ max-width: 170px; max-height: 170px; }}
    .cols {{ display: grid; grid-template-columns: 1fr; gap: 8px; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; font-size: 11px; max-height: 190px; overflow: auto; }}
  </style>
</head>
<body>
  <h1>Prefix Completion Examples</h1>
  <p class="note">Focused prompts for qualitative Part 4 analysis. These examples are useful for showing both successful SVG syntax continuation and remaining semantic/path-generation failure modes.</p>
  <div class="grid">{''.join(cards)}</div>
</body>
</html>
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="outputs/runs/large_lite/ckpt.pt")
    parser.add_argument("--out_dir", default="outputs/runs/large_lite/prefix_examples")
    parser.add_argument("--html_out", default="outputs/runs/large_lite/prefix_examples.html")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--temperatures", default="0.25,0.35,0.45")
    parser.add_argument("--top_k", type=int, default=15)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--repair", action="store_true")
    args = parser.parse_args()

    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    ckpt = safe_torch_load(args.ckpt, map_location=device)
    tokenizer = SvgTokenizer(ckpt["tokenizer_path"])
    model, _, _ = load_model(Path(args.ckpt), device)

    out_dir = Path(args.out_dir)
    svg_dir = out_dir / "svg"
    png_dir = out_dir / "png"
    svg_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    temperatures = [float(t) for t in args.temperatures.split(",") if t.strip()]
    rows = []
    sample_id = 0
    for case in PREFIX_CASES:
        for temperature in temperatures:
            prompt = case["prompt"]
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

            stem = f"{sample_id:03d}_{case['name']}_t{temperature:g}"
            svg_path = svg_dir / f"{stem}.svg"
            png_path = png_dir / f"{stem}.png"
            svg_path.write_text(svg, encoding="utf-8")
            rendered = render(svg, png_path)
            row = {
                "id": sample_id,
                "case": case["name"],
                "description": case["description"],
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
                "xml_valid": xml_valid(svg),
                "structural_valid": structural_valid(svg) if xml_valid(svg) else False,
                "render_valid": render_valid(svg) if xml_valid(svg) else False,
            }
            rows.append(row)
            sample_id += 1

    write_jsonl(out_dir / "samples.jsonl", rows)
    make_html(rows, Path(args.html_out))
    print(f"Wrote {len(rows)} prefix examples to {args.html_out}")


if __name__ == "__main__":
    main()
