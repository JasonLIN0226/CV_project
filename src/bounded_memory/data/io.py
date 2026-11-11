"""File helpers shared by every environment: hashes, atomic writes, stable seeds.

Extracted from experiments/predictor/control/src/phase1/utils.py; the function bodies are unchanged.
"""
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def atomic(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".partial-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json(path, obj):
    atomic(path, lambda f: f.write(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False).encode()))


def write_npz(path, **arrays):
    atomic(path, lambda f: np.savez_compressed(f, **arrays))


def save_torch(path, obj):
    atomic(path, lambda f: torch.save(obj, f))


def stable_seed(*parts):
    return int(digest(parts)[:16], 16) % (2**32)


def require_sha256(path, expected, what="file"):
    """Refuse to read an asset whose bytes differ from the recorded checksum."""
    actual = file_hash(path)
    if actual != expected:
        raise ValueError(f"{what} checksum mismatch: {path} is {actual}, expected {expected}")
    return Path(path)
