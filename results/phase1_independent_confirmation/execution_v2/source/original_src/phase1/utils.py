import csv
import hashlib
import json
import os
import platform
import random
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


def code_hash(*names):
    root = Path(__file__).parent
    paths = [root / n for n in names] if names else sorted(root.glob("*.py"))
    return digest({p.name: file_hash(p) for p in paths})


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


def write_csv(path, rows):
    if not rows:
        return
    def writer(f):
        import io
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
        f.write(buf.getvalue().encode())
    atomic(path, writer)


def seed_all(seed, threads=2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"].cpu())
    if s["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([v.cpu() for v in s["cuda"]])


def environment(device="cpu"):
    import importlib.metadata
    packages = {}
    for name in ["torch", "torchvision", "numpy", "Pillow", "PyYAML", "matplotlib", "transformers", "huggingface-hub", "safetensors", "pytest"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
            "device_requested": device, "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda,
            "gpu": [{"name": torch.cuda.get_device_name(i), "total_bytes": torch.cuda.get_device_properties(i).total_memory}
                    for i in range(torch.cuda.device_count())], "slurm_job_id": os.getenv("SLURM_JOB_ID"),
            "code_sha256": code_hash()}


def device_of(cfg):
    device = torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; explicitly choose --device cpu for smoke.")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only CPU/CUDA are supported for reproducible diagnostics.")
    return device


def stable_seed(*parts):
    return int(digest(parts)[:16], 16) % (2**32)
