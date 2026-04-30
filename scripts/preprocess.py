#!/usr/bin/env python3
"""Download and clean SVG datasets into JSONL splits."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset
from lxml import etree
from tqdm import tqdm

from common import write_json


NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")
WHITESPACE_RE = re.compile(r">\s+<|\s+")


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


def strip_unwanted(root: etree._Element) -> None:
    remove_names = {"metadata", "title", "desc"}
    for node in list(root.iter()):
        local = etree.QName(node).localname if isinstance(node.tag, str) else ""
        if local in remove_names and node.getparent() is not None:
            node.getparent().remove(node)


def clean_svg(svg: str, precision: int) -> str:
    parser = etree.XMLParser(remove_comments=True, resolve_entities=False, no_network=True)
    root = etree.fromstring(svg.encode("utf-8"), parser=parser)
    if etree.QName(root).localname != "svg":
        raise ValueError("root element is not svg")
    strip_unwanted(root)
    compact = etree.tostring(root, encoding="unicode", method="xml", pretty_print=False)
    compact = NUMBER_RE.sub(lambda m: round_number(m, precision), compact)
    compact = WHITESPACE_RE.sub(lambda m: "><" if m.group(0).startswith(">") else " ", compact)
    return compact.strip()


def split_name(index: int, n_total: int, val_frac: float, test_frac: float) -> str:
    train_end = int(n_total * (1.0 - val_frac - test_frac))
    val_end = int(n_total * (1.0 - test_frac))
    if index < train_end:
        return "train"
    if index < val_end:
        return "val"
    return "test"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="starvector/svg-icons-simple")
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--min_chars", type=int, default=50)
    parser.add_argument("--max_chars", type=int, default=12000, help="Pre-tokenizer length guard.")
    parser.add_argument("--precision", type=int, default=1)
    parser.add_argument("--val_frac", type=float, default=0.01)
    parser.add_argument("--test_frac", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.dataset, split="train")
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    dataset = dataset.shuffle(seed=args.seed)

    records: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "dataset": args.dataset,
        "raw_rows": len(dataset),
        "kept": 0,
        "dropped_no_svg": 0,
        "dropped_parse": 0,
        "dropped_length": 0,
        "splits": {name: {"files": 0, "chars": 0} for name in ["train", "val", "test"]},
        "cleaning": {
            "remove_xml_comments": True,
            "remove_elements": ["metadata", "title", "desc"],
            "coordinate_precision": args.precision,
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
        },
    }

    for row in tqdm(dataset, desc="cleaning"):
        raw_svg = first_svg_field(row)
        if raw_svg is None:
            stats["dropped_no_svg"] += 1
            continue
        try:
            svg = clean_svg(raw_svg, args.precision)
        except Exception:
            stats["dropped_parse"] += 1
            continue
        if len(svg) < args.min_chars or len(svg) > args.max_chars:
            stats["dropped_length"] += 1
            continue
        records.append({"id": len(records), "svg": svg, "source": args.dataset})

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
