#!/usr/bin/env python3
"""Generate focused prefix-completion examples for the report."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

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
    {
        "name": "vertical_path",
        "description": "Path continuation: a vertical stroke is started and the model completes the SVG path.",
        "prompt": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" height="200px" width="200px"><path fill="none" stroke="black" stroke-width="0.3" stroke-opacity="1" filling="0" d="M12 4 L12',
    },
]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def rel_url(from_dir: Path, target: str | None) -> str:
    if not target:
        return ""
    path = Path(target)
    try:
        return path.resolve().relative_to(from_dir.resolve()).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            return path.as_posix()


def make_html(rows: list[dict], out: Path, max_code_chars: int = 1800) -> None:
    cards = []
    for row in rows:
        ok = bool(row["xml_valid"] and row["structural_valid"] and row["render_valid"])
        flags = (
            f"XML {row['xml_valid']} | struct {row['structural_valid']} | "
            f"render {row['render_valid']} | repaired {row['repaired']}"
        )
        completion = row["svg"]
        if completion.startswith(row["prompt"]):
            completion = completion[len(row["prompt"]) :]
        if ok and row.get("png_path") and Path(row["png_path"]).exists():
            preview = (
                f"<img src='{html.escape(rel_url(out.parent, row['png_path']))}' "
                f"alt='{html.escape(row['case'])} t={row['temperature']}'>"
            )
        else:
            preview = "<div class='invalid-mark'>INVALID<br><span>no repair</span></div>"
        cards.append(
            f"<article class='{'ok' if ok else 'bad'}'>"
            f"<h2>{html.escape(row['case'])} t={row['temperature']}</h2>"
            f"<p class='desc'>{html.escape(row['description'])}</p>"
            f"<div class='preview'>{preview}</div>"
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
    article.ok {{ border-color: #86efac; box-shadow: inset 0 0 0 2px #dcfce7; }}
    article.bad {{ border-color: #fca5a5; background: #fff7f7; }}
    h2 {{ font-size: 15px; margin: 0 0 6px; }}
    h3 {{ font-size: 12px; margin: 8px 0 4px; }}
    .desc, .flags {{ font-size: 12px; color: #4b5563; }}
    .preview {{ height: 190px; display: grid; place-items: center; background: #f9fafb; overflow: hidden; }}
    .preview img {{ max-width: 170px; max-height: 170px; object-fit: contain; }}
    .invalid-mark {{ color: #991b1b; font-weight: 700; border: 2px solid #ef4444; padding: 10px 14px; border-radius: 4px; text-align: center; }}
    .invalid-mark span {{ font-weight: 500; }}
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


def open_on_white(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    background = Image.new("RGBA", image.size, "white")
    background.alpha_composite(image)
    return background.convert("RGB")


def make_contact_sheet(rows: list[dict], out: Path, *, max_cards: int | None, title: str) -> None:
    valid_rows = [row for row in rows if row["xml_valid"] and row["structural_valid"] and row["render_valid"]]
    if max_cards is None:
        selected = rows
        cols = 5
        cell_w, cell_h = 260, 200
    else:
        selected = valid_rows[:max_cards]
        if not selected:
            selected = rows[:max_cards]
        cols = 4
        cell_w, cell_h = 330, 230
    rows_n = max(1, math.ceil(len(selected) / cols))
    label_h = 42
    title_h = 52
    margin, gap = 24, 16
    width = margin * 2 + cols * cell_w + (cols - 1) * gap
    height = margin * 2 + title_h + rows_n * (cell_h + label_h) + (rows_n - 1) * gap
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        title_font = ImageFont.truetype("arial.ttf", 26)
        label_font = ImageFont.truetype("arial.ttf", 15)
        flag_font = ImageFont.truetype("arial.ttf", 12)
        invalid_font = ImageFont.truetype("arial.ttf", 22)
    except Exception:
        title_font = label_font = flag_font = invalid_font = None
    draw.text((margin, margin), title, fill=(0, 0, 0), font=title_font)
    y0 = margin + title_h
    for idx, row in enumerate(selected):
        grid_row, grid_col = divmod(idx, cols)
        x = margin + grid_col * (cell_w + gap)
        y = y0 + grid_row * (cell_h + label_h + gap)
        ok = bool(row["xml_valid"] and row["structural_valid"] and row["render_valid"])
        label = f"{row['id']:03d} {row['case']} | t={row['temperature']}"
        flags = f"XML {row['xml_valid']} | struct {row['structural_valid']} | render {row['render_valid']}"
        draw.text((x + 4, y + 2), label, fill=(0, 0, 0), font=label_font)
        draw.text((x + 4, y + 22), flags, fill=(32, 95, 45) if ok else (153, 27, 27), font=flag_font)
        border = (134, 239, 172) if ok else (252, 165, 165)
        draw.rectangle([x, y + label_h, x + cell_w, y + label_h + cell_h], outline=border, width=3)
        if ok and row.get("png_path") and Path(row["png_path"]).exists():
            image = open_on_white(Path(row["png_path"]))
            image.thumbnail((cell_w - 24, cell_h - 24), Image.Resampling.LANCZOS)
            paste_x = x + (cell_w - image.width) // 2
            paste_y = y + label_h + (cell_h - image.height) // 2
            sheet.paste(image, (paste_x, paste_y))
        else:
            draw.multiline_text(
                (x + cell_w / 2, y + label_h + cell_h / 2),
                "INVALID\n(no repair)",
                fill=(153, 27, 27),
                font=invalid_font,
                anchor="mm",
                align="center",
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


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
    parser.add_argument("--samples_per_case", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--clean", action="store_true", help="Remove old generated SVG/PNG files before sampling.")
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
    if args.clean:
        for directory in (svg_dir, png_dir):
            for path in directory.glob("*"):
                if path.is_file():
                    path.unlink()

    temperatures = [float(t) for t in args.temperatures.split(",") if t.strip()]
    jobs: list[tuple[dict, float]] = []
    if args.samples_per_case is None:
        jobs = [(case, temperature) for case in PREFIX_CASES for temperature in temperatures]
    else:
        for case in PREFIX_CASES:
            for repeat in range(args.samples_per_case):
                jobs.append((case, temperatures[repeat % len(temperatures)]))

    rows = []
    for sample_id, (case, temperature) in enumerate(jobs):
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
        is_xml_valid = xml_valid(svg)
        is_structural_valid = structural_valid(svg) if is_xml_valid else False
        is_render_valid = render_valid(svg) if is_xml_valid else False
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
            "xml_valid": is_xml_valid,
            "structural_valid": is_structural_valid,
            "render_valid": is_render_valid,
        }
        rows.append(row)

    write_jsonl(out_dir / "samples.jsonl", rows)
    make_html(rows, Path(args.html_out))
    make_contact_sheet(
        rows,
        out_dir / "preview_contact_white.png",
        max_cards=12,
        title="Selected valid no-repair prefix completions",
    )
    make_contact_sheet(
        rows,
        out_dir / "preview_contact_all_strict.png",
        max_cards=None,
        title="All no-repair prefix completions",
    )
    valid_count = sum(1 for row in rows if row["xml_valid"] and row["structural_valid"] and row["render_valid"])
    print(f"Wrote {len(rows)} prefix examples to {args.html_out}")
    print(f"Strict valid/renderable prefix examples: {valid_count}/{len(rows)}")


if __name__ == "__main__":
    main()
