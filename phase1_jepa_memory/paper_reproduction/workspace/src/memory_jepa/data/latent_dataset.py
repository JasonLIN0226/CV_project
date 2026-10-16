"""Explicit reconstruction of a module omitted from the SPACE release.

Sampling assumptions: every complete-target anchor, a uniform subset of the
historical prefix (equivalent to Reservoir marginals), sorted chronologically;
training-only memory dropout. Validation uses deterministic anchor-keyed draws.
These choices are documented reconstruction, not claimed author implementation.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass, fields
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class LatentCorpus:
    def __init__(self, manifest_path: str | Path):
        self.path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.path.read_text())
        if self.manifest.get("test_features_accessed") is not False:
            raise ValueError("only an explicitly test-isolated manifest is permitted")
        rows = self.manifest["videos"]
        if any(r["split"] not in {"train", "val"} for r in rows):
            raise ValueError("unexpected split in corpus")
        if len({r["video_id"] for r in rows}) != len(rows):
            raise ValueError("duplicate recording")
        self.cache = OrderedDict()
        self.cache_limit = 4

    def split_records(self, split: str) -> list[dict]:
        if split not in {"train", "val"}:
            raise ValueError("test split is sealed")
        return [r for r in self.manifest["videos"] if r["split"] == split]

    def load(self, row: dict) -> dict:
        if row not in self.manifest["videos"] or row["split"] not in {"train", "val"}:
            raise ValueError("record is outside the approved corpus")
        name = row["path"]
        lexical = self.path.parent / name
        path = lexical.resolve()
        if path.parent != self.path.parent or Path(name).is_absolute() or lexical.is_symlink():
            raise ValueError("unsafe latent path")
        if name in self.cache:
            self.cache.move_to_end(name)
            return self.cache[name]
        record = torch.load(path, map_location="cpu", weights_only=True)
        if record["video_id"] != row["video_id"] or record["valid_length"] != row["valid_length"]:
            raise ValueError("record/manifest mismatch")
        latent = record["latents"]
        if latent.ndim != 2 or latent.shape != (row["valid_length"], self.manifest["d_in"]):
            raise ValueError("latent dimensions do not match manifest")
        if not torch.isfinite(latent).all():
            raise ValueError("nonfinite latent")
        self.cache[name] = record
        if self.cache_limit is not None and len(self.cache) > self.cache_limit:
            self.cache.popitem(last=False)
        return record


@dataclass
class PredictorBatch:
    local: torch.Tensor
    memory: torch.Tensor
    memory_valid: torch.Tensor
    local_age: torch.Tensor
    memory_age: torch.Tensor
    targets: torch.Tensor
    target_valid: torch.Tensor

    def to(self, device: torch.device | str) -> "PredictorBatch":
        return PredictorBatch(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})


class PredictorTrainingDataset(Dataset):
    def __init__(self, *, manifest_path, split, local_context_len, memory_slots,
                 horizons, memory_dropout, seed):
        self.corpus = LatentCorpus(manifest_path)
        self.rows = self.corpus.split_records(split)
        if not self.rows:
            raise ValueError("empty approved split")
        self.split = split
        # Avoid reopening entire .pt recordings for randomly shuffled anchors.
        # This preserves the sampled examples; the complete official stride-15
        # corpus is audited before fitting and fits the 60 GB GPU host budget.
        self.corpus.cache_limit = None
        for row in self.rows:
            self.corpus.load(row)
        self.context = int(local_context_len)
        self.slots = int(memory_slots)
        self.capacity = self.slots - 1
        self.horizons = tuple(int(h) for h in horizons)
        self.dropout = float(memory_dropout) if split == "train" else 0.0
        self.seed = int(seed)
        if self.context < 1 or self.capacity < 1 or min(self.horizons) <= 0 or not 0 <= self.dropout < 1:
            raise ValueError("invalid sampling configuration")
        self.ends = []
        total = 0
        for row in self.rows:
            count = row["valid_length"] - max(self.horizons) - self.context + 1
            if count <= 0:
                raise ValueError("record has no complete-target anchor")
            total += count
            self.ends.append(total)

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        video = bisect_right(self.ends, index)
        offset = index - (self.ends[video - 1] if video else 0)
        row = self.rows[video]
        latent = self.corpus.load(row)["latents"].float()
        current = self.context - 1 + offset
        # The context ends at current; all historical timestamps precede it.
        historical_count = current - self.context + 1
        generator = None
        if self.split == "val":
            key = f"{self.seed}:{row['video_id']}:{current}".encode()
            key_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**63)
            generator = torch.Generator().manual_seed(key_seed)
        selected = torch.randperm(historical_count, generator=generator)[:self.capacity].sort().values
        if self.dropout and len(selected):
            selected = selected[torch.rand(len(selected)) >= self.dropout]
        memory = torch.zeros(self.slots, latent.shape[1])
        valid = torch.zeros(self.slots, dtype=torch.bool)
        memory_age = torch.zeros(self.slots)
        if len(selected):
            memory[:len(selected)] = latent[selected]
            valid[:len(selected)] = True
            memory_age[:len(selected)] = current - selected
        return PredictorBatch(
            local=latent[current - self.context + 1:current + 1].clone(),
            memory=memory, memory_valid=valid,
            local_age=torch.arange(self.context - 1, -1, -1).float(),
            memory_age=memory_age,
            targets=latent[torch.tensor(self.horizons) + current].clone(),
            target_valid=torch.ones(len(self.horizons), dtype=torch.bool),
        )


def collate_predictor(items):
    return PredictorBatch(**{f.name: torch.stack([getattr(i, f.name) for i in items]) for f in fields(PredictorBatch)})
