#!/usr/bin/env python3
"""Train a BPE tokenizer and encode JSONL splits into token memmaps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tqdm import tqdm

from common import SPECIAL_TOKENS, SvgTokenizer, read_jsonl, write_json


def iter_svgs(path: Path):
    for row in read_jsonl(path):
        yield row["svg"]


def train_tokenizer(train_path: Path, out_path: Path, vocab_size: int) -> None:
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=SPECIAL_TOKENS)
    tokenizer.train_from_iterator(iter_svgs(train_path), trainer=trainer)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_path))


def encode_split(tokenizer: SvgTokenizer, src: Path, dst: Path, max_tokens: int) -> dict:
    lengths: list[int] = []
    kept = 0
    dropped = 0
    token_chunks: list[np.ndarray] = []
    filtered_jsonl = dst.with_name(f"{dst.stem}_filtered.jsonl")
    with open(filtered_jsonl, "w", encoding="utf-8") as out_json:
        for row in tqdm(list(read_jsonl(src)), desc=f"encoding {src.stem}"):
            ids = tokenizer.encode(row["svg"], add_special=True)
            if len(ids) > max_tokens:
                dropped += 1
                continue
            lengths.append(len(ids))
            kept += 1
            row["token_length"] = len(ids)
            out_json.write(json.dumps(row, sort_keys=True) + "\n")
            token_chunks.append(np.asarray(ids, dtype=np.uint32))
    if token_chunks:
        all_ids = np.concatenate(token_chunks)
    else:
        all_ids = np.asarray([], dtype=np.uint32)
    mmap = np.memmap(dst, dtype=np.uint32, mode="w+", shape=(len(all_ids),))
    mmap[:] = all_ids[:]
    mmap.flush()
    return {
        "source": str(src),
        "encoded": str(dst),
        "filtered_jsonl": str(filtered_jsonl),
        "files": kept,
        "dropped_too_long": dropped,
        "tokens": int(len(all_ids)),
        "length_min": int(min(lengths)) if lengths else 0,
        "length_max": int(max(lengths)) if lengths else 0,
        "length_mean": float(np.mean(lengths)) if lengths else 0.0,
        "length_p50": float(np.percentile(lengths, 50)) if lengths else 0.0,
        "length_p95": float(np.percentile(lengths, 95)) if lengths else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--tokenizer_path", default="data/tokenizer/tokenizer.json")
    parser.add_argument("--vocab_size", type=int, default=2048)
    parser.add_argument("--max_tokens", type=int, default=1024)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    tokenizer_path = Path(args.tokenizer_path)
    train_tokenizer(data_dir / "train.jsonl", tokenizer_path, args.vocab_size)
    tokenizer = SvgTokenizer(tokenizer_path)

    stats = {
        "tokenizer_path": str(tokenizer_path),
        "vocab_size": tokenizer.vocab_size,
        "max_tokens_per_svg": args.max_tokens,
        "splits": {},
    }
    for split in ["train", "val", "test"]:
        stats["splits"][split] = encode_split(tokenizer, data_dir / f"{split}.jsonl", data_dir / f"{split}.bin", args.max_tokens)
    write_json(data_dir / "tokenizer_stats.json", stats)
    print(f"Tokenizer vocab={tokenizer.vocab_size}; encoded splits in {data_dir}")


if __name__ == "__main__":
    main()
