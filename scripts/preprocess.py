#!/usr/bin/env python3
"""Download and clean SVG datasets into JSONL splits."""

from __future__ import annotations

import argparse
import json
import re
import random
import os
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from lxml import etree
from tqdm import tqdm

from common import write_json


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


NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")
WHITESPACE_RE = re.compile(r">\s+<|\s+")
CACHE_ROOT = Path.home() / ".cache" / "huggingface" / "datasets"


def first_svg_field(row: dict[str, Any]) -> str | None:
    preferred = ["svg", "Svg", "code", "text", "content"]
    for key in preferred:
        value = row.get(key)
        if isinstance(value, str) and "<svg" in value:
            return value
    for value in row.values():
        if isinstance(value, str) and "<svg" in value:
            return value
    return None


def round_number(match: re.Match[str], precision: int) -> str:
    token = match.group(0)
    if "." not in token and "e" not in token.lower():
        return token
    try:
        value = float(token)
    except ValueError:
        return token
    rounded = f"{value:.{precision}f}".rstrip("0").rstrip(".")
    if rounded == "-0":
        return "0"
    return rounded


def round_numbers_in_text(text: str, precision: int) -> str:
    return NUMBER_RE.sub(lambda m: round_number(m, precision), text)


def strip_unwanted(root: etree._Element) -> None:
    remove_names = {"metadata", "title", "desc"}
    for node in list(root.iter()):
        local = etree.QName(node).localname if isinstance(node.tag, str) else ""
        if local in remove_names and node.getparent() is not None:
            node.getparent().remove(node)


def round_numeric_attributes(root: etree._Element, precision: int) -> None:
    for node in root.iter():
        if not isinstance(node.tag, str):
            continue
        for key, value in list(node.attrib.items()):
            node.attrib[key] = round_numbers_in_text(value, precision)


def clean_svg(svg: str, precision: int) -> str:
    parser = etree.XMLParser(remove_comments=True, resolve_entities=False, no_network=True)
    root = etree.fromstring(svg.encode("utf-8"), parser=parser)
    if etree.QName(root).localname != "svg":
        raise ValueError("root element is not svg")
    strip_unwanted(root)
    round_numeric_attributes(root, precision)
    compact = etree.tostring(root, encoding="unicode", method="xml", pretty_print=False)
    compact = WHITESPACE_RE.sub(lambda m: "><" if m.group(0).startswith(">") else " ", compact)
    return compact.strip()


def render_ok(svg: str) -> bool:
    if cairosvg is None:
        return False
    try:
        cairosvg.svg2png(bytestring=svg.encode("utf-8"), output_width=64, output_height=64)
        return True
    except Exception:
        return False


def split_name(index: int, n_total: int, val_frac: float, test_frac: float) -> str:
    train_end = int(n_total * (1.0 - val_frac - test_frac))
    val_end = int(n_total * (1.0 - test_frac))
    if index < train_end:
        return "train"
    if index < val_end:
        return "val"
    return "test"


def cached_train_arrow_paths(source_name: str) -> list[Path]:
    cache_dir = CACHE_ROOT / source_name.replace("/", "___")
    if not cache_dir.exists():
        raise FileNotFoundError(f"Missing local cache for {source_name}: {cache_dir}")
    paths = sorted(cache_dir.rglob("*-train*.arrow"))
    paths = [path for path in paths if not path.name.startswith("cache-")]
    if not paths:
        raise FileNotFoundError(f"No train Arrow files found for {source_name} under {cache_dir}")
    return paths


def source_datasets(source_name: str, use_cached_starvector: bool) -> list[tuple[str, Dataset]]:
    if not use_cached_starvector:
        return [(source_name, load_dataset(source_name, split="train"))]
    return [
        (str(path), Dataset.from_file(str(path)))
        for path in cached_train_arrow_paths(source_name)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, help="Single HuggingFace dataset, kept for backward compatibility.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="One or more HuggingFace datasets to combine before splitting.",
    )
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--min_chars", type=int, default=50)
    parser.add_argument("--max_chars", type=int, default=12000, help="Pre-tokenizer length guard.")
    parser.add_argument("--precision", type=int, default=1)
    parser.add_argument("--val_frac", type=float, default=0.01)
    parser.add_argument("--test_frac", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--limit_per_dataset", type=int, default=None)
    parser.add_argument("--max_records", type=int, default=None, help="Stop after this many cleaned SVGs across all datasets.")
    parser.add_argument("--render_validate", action="store_true", help="Drop SVGs that CairoSVG cannot render.")
    parser.add_argument(
        "--use_cached_starvector",
        action="store_true",
        help="Read StarVector train Arrow files directly from the local HuggingFace cache.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_names = args.datasets or [args.dataset or "starvector/svg-icons-simple"]

    records: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "datasets": source_names,
        "raw_rows": 0,
        "kept": 0,
        "dropped_no_svg": 0,
        "dropped_parse": 0,
        "dropped_length": 0,
        "dropped_render": 0,
        "sources": {},
        "splits": {name: {"files": 0, "chars": 0} for name in ["train", "val", "test"]},
        "cleaning": {
            "remove_xml_comments": True,
            "remove_elements": ["metadata", "title", "desc"],
            "coordinate_precision": args.precision,
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
            "render_validate": args.render_validate,
        },
    }

    for source_index, source_name in enumerate(source_names):
        datasets = source_datasets(source_name, args.use_cached_starvector)
        row_limit = args.limit_per_dataset or args.limit
        total_rows = sum(len(dataset) for _, dataset in datasets)
        limited_rows = min(row_limit, total_rows) if row_limit else total_rows
        source_stats = {
            "raw_rows": limited_rows,
            "kept": 0,
            "dropped_no_svg": 0,
            "dropped_parse": 0,
            "dropped_length": 0,
            "dropped_render": 0,
            "arrow_files": [label for label, _ in datasets] if args.use_cached_starvector else [],
        }
        stats["raw_rows"] += limited_rows

        seen_rows = 0
        for shard_index, (dataset_label, dataset) in enumerate(datasets):
            remaining_limit = None if row_limit is None else row_limit - seen_rows
            if remaining_limit is not None and remaining_limit <= 0:
                break

            indices = list(range(len(dataset)))
            random.Random(args.seed + source_index * 1009 + shard_index).shuffle(indices)
            if remaining_limit is not None:
                indices = indices[:remaining_limit]
            seen_rows += len(indices)

            desc = f"cleaning {source_name}"
            if len(datasets) > 1:
                desc = f"{desc} shard {shard_index + 1}/{len(datasets)}"
            for row_index in tqdm(indices, desc=desc):
                row = dataset[int(row_index)]
                raw_svg = first_svg_field(row)
                if raw_svg is None:
                    stats["dropped_no_svg"] += 1
                    source_stats["dropped_no_svg"] += 1
                    continue
                try:
                    svg = clean_svg(raw_svg, args.precision)
                except Exception:
                    stats["dropped_parse"] += 1
                    source_stats["dropped_parse"] += 1
                    continue
                if len(svg) < args.min_chars or len(svg) > args.max_chars:
                    stats["dropped_length"] += 1
                    source_stats["dropped_length"] += 1
                    continue
                if args.render_validate and not render_ok(svg):
                    stats["dropped_render"] += 1
                    source_stats["dropped_render"] += 1
                    continue
                records.append({"id": len(records), "svg": svg, "source": source_name})
                source_stats["kept"] += 1
                if args.max_records and len(records) >= args.max_records:
                    break
            if args.max_records and len(records) >= args.max_records:
                break
        stats["sources"][source_name] = source_stats
        if args.max_records and len(records) >= args.max_records:
            break

    random.Random(args.seed).shuffle(records)
    for new_id, record in enumerate(records):
        record["id"] = new_id

    stats["kept"] = len(records)
    writers = {name: open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") for name in ["train", "val", "test"]}
    try:
        for i, record in enumerate(records):
            split = split_name(i, len(records), args.val_frac, args.test_frac)
            writers[split].write(json.dumps(record, sort_keys=True) + "\n")
            stats["splits"][split]["files"] += 1
            stats["splits"][split]["chars"] += len(record["svg"])
    finally:
        for f in writers.values():
            f.close()

    write_json(out_dir / "preprocess_stats.json", stats)
    print(f"Kept {stats['kept']} SVGs in {out_dir}")


if __name__ == "__main__":
    main()
