#!/usr/bin/env python3
"""Build an HTML contact sheet for generated SVG samples."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_code_chars", type=int, default=1600)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.samples))
    cards = []
    for row in rows:
        svg = row.get("svg", "")
        prompt = row.get("prompt", "")
        kind = row.get("kind", "")
        temperature = row.get("temperature", "")
        flags = [
            f"XML {row.get('xml_valid')}",
            f"struct {row.get('structural_valid')}",
            f"render {row.get('render_valid', row.get('rendered'))}",
        ]
        cards.append(
            "<article>"
            f"<h2>{html.escape(str(row.get('id')))} {html.escape(kind)} t={html.escape(str(temperature))}</h2>"
            f"<div class='preview'>{svg}</div>"
            f"<p class='flags'>{html.escape(' | '.join(flags))}</p>"
            "<div class='cols'>"
            f"<section><h3>Prefix</h3><pre>{html.escape(prompt)}</pre></section>"
            f"<section><h3>SVG</h3><pre>{html.escape(svg[:args.max_code_chars])}</pre></section>"
            "</div>"
            "</article>"
        )
    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Generated SVG samples</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #111827; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
    article {{ border: 1px solid #d1d5db; border-radius: 8px; padding: 12px; background: white; }}
    h2 {{ font-size: 15px; margin: 0 0 8px; }}
    h3 {{ font-size: 12px; margin: 8px 0 4px; }}
    .preview {{ height: 180px; display: grid; place-items: center; background: #f9fafb; overflow: hidden; }}
    .preview svg {{ max-width: 160px; max-height: 160px; }}
    .flags {{ font-size: 12px; color: #4b5563; }}
    .cols {{ display: grid; grid-template-columns: 1fr; gap: 8px; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; font-size: 11px; max-height: 180px; overflow: auto; }}
  </style>
</head>
<body>
  <h1>Generated SVG Samples</h1>
  <div class="grid">{''.join(cards)}</div>
</body>
</html>
"""
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    print(f"Wrote sample sheet to {out}")


if __name__ == "__main__":
    main()
