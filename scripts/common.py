#!/usr/bin/env python3
"""Shared utilities for the SVG scaling project."""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from tokenizers import Tokenizer


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def read_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def append_jsonl(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SvgTokenizer:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self.tokenizer = Tokenizer.from_file(self.path)
        self.pad_id = self.tokenizer.token_to_id("<pad>")
        self.bos_id = self.tokenizer.token_to_id("<bos>")
        self.eos_id = self.tokenizer.token_to_id("<eos>")
        self.unk_id = self.tokenizer.token_to_id("<unk>")
        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        ids = self.tokenizer.encode(text).ids
        if add_special:
            return [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int] | np.ndarray | torch.Tensor, skip_special: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special)


@dataclass
class TrainConfig:
    name: str
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    block_size: int = 1024
    dropout: float = 0.1
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_steps: int = 100
    max_steps: int | None = None
    eval_interval: int = 200
    eval_iters: int = 50
    log_interval: int = 20
    seed: int = 1337
    compile: bool = False
    notes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainConfig":
        return cls(**data)

    @property
    def tokens_per_optimizer_step(self) -> int:
        return self.batch_size * self.block_size * self.gradient_accumulation_steps


def load_train_config(path: str | Path) -> TrainConfig:
    return TrainConfig.from_dict(read_yaml(path))


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def cosine_lr(step: int, cfg: TrainConfig, total_steps: int) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / max(1, cfg.warmup_steps)
    if step >= total_steps:
        return cfg.min_lr
    ratio = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


def memmap_split(data_dir: str | Path, split: str) -> np.memmap:
    path = Path(data_dir) / f"{split}.bin"
    if not path.exists():
        raise FileNotFoundError(f"Missing encoded split: {path}")
    return np.memmap(path, dtype=np.uint32, mode="r")


def checkpoint_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "ckpt.pt"


def safe_torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
