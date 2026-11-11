"""Shared plumbing for the thin experiment scripts: configs, run outputs and provenance.

Every script takes an explicit config file, seed, device and output directory. Configs are YAML;
`${VAR}` in a path is expanded from the environment so the same file serves local runs and Torch.
Each task writes one small JSON file atomically and is skipped when that file already exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .data.io import write_json

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[1]


def expand(value):
    """Expand ${VAR} in every string of a nested config; an unset variable is an error."""
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v) for v in value]
    if isinstance(value, str) and "$" in value:
        out = os.path.expandvars(value)
        if "$" in out:
            raise KeyError(f"environment variable not set in config value: {value}")
        return out
    return value


def load_config(path):
    return expand(yaml.safe_load(Path(path).read_text()))


def source_hash():
    """SHA256 over the package's Python sources: identifies the code a result came from."""
    digest = hashlib.sha256()
    for path in sorted(PACKAGE.rglob("*.py")):
        digest.update(str(path.relative_to(PACKAGE)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def parser(description, **extra):
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True, help="environment config (YAML)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, required=True, help="output directory for this task's small files")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    ap.add_argument("--force", action="store_true", help="overwrite an existing output")
    return ap


class Task:
    """One (script, seed) unit of work with a single JSON output."""

    def __init__(self, args, name):
        self.args, self.name = args, name
        self.path = Path(args.out) / f"{name}_seed{args.seed}.json"
        self.started = time.time()
        self.info = {}
        torch.set_num_threads(args.threads)

    def done(self):
        if self.path.exists() and not self.args.force:
            print(f"{self.path} exists, skipping", flush=True)
            return True
        return False

    def log(self, message):
        print(f"[{self.name} seed {self.args.seed} +{time.time() - self.started:6.0f}s] {message}", flush=True)

    def write(self, **payload):
        record = {"task": self.name, "seed": self.args.seed, "device": str(self.args.device),
                  "host": socket.gethostname(), "python": platform.python_version(), "torch": torch.__version__,
                  "numpy": np.__version__, "threads": self.args.threads, "config": str(self.args.config),
                  "source_sha256": source_hash(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                  "seconds": time.time() - self.started, "info": self.info, **payload}
        write_json(self.path, json.loads(json.dumps(record, default=_plain)))
        self.log(f"wrote {self.path}")


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value))
