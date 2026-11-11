"""What every Synergy script shares: open the stores and readers named by a config, and record a policy.

A Synergy config (configs/synergy/*.yaml) names:

    assets:
      terminal:       the terminal-study release folder (records, basis, action banks for TRAIN/VAL/DEV)
      train_features: projected TRAIN/VAL/DEV features (re-encoded; the release has only confirmation)
      confirmation:   records and projected features of the locked confirmation set
      readers:        the reader-aggregation release folder (readers/<name>/reader_selected.pt)
      reader_checksums: omit for the released readers (pinned); "reader_locked.json" for readers trained in a run
      released:       the confirmation release (stage-A arrays, student archives, predictions)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ...evaluation.metrics.synergy import score
from .assets import READERS, Store, load_reader


def confirmation_store(config):
    a = config["assets"]
    conf = Path(a["confirmation"])
    return Store(conf / "records.json", conf / "actions.npz", conf, a["terminal"])


def train_store(config):
    a = config["assets"]
    run = Path(a["terminal"])
    return Store(run / "data/records.json", run / "data/actions.npz", a["train_features"], run)


def reader(config, name, store):
    if name not in READERS:
        raise SystemExit(f"unknown reader {name}; the Synergy study uses {READERS}")
    folder = Path(config["assets"]["readers"])
    sha256 = None
    if config["assets"].get("reader_checksums") == "reader_locked.json":      # readers trained in a run
        sha256 = json.loads((folder / f"readers/{name}/reader_locked.json").read_text())["checkpoint_sha256"]
    return load_reader(name, store, folder, sha256)


def released_cost(config, name, records):
    """The released per-deletion evaluation cost on confirmation, aligned to `records`."""
    a = np.load(Path(config["assets"]["released"]) / f"conf/stageA/{name}_confirmation.npz")
    if a["ids"].tolist() != [r["id"] for r in records]:
        raise ValueError("released confirmation arrays do not align with the confirmation records")
    return a


def policy(records, cost, *, drop=None, loss=None, seed=None):
    """One policy's record in a Synergy rows file. Give `drop` (deleted card per record) or `loss`."""
    metrics = score(records, cost, drops=None if drop is None else [np.asarray(drop)],
                    losses=None if loss is None else [np.asarray(loss)])
    return {"seed": seed, "metrics": metrics,
            "drop": None if drop is None else np.asarray(drop).astype(int).tolist(),
            "loss": None if loss is None else np.asarray(loss).astype(float).tolist()}


def describe(task, name, record):
    m = record["metrics"]
    task.log(f"{name:42s} " + "  ".join(
        f"{t}: lost {'  n/a ' if m[t]['needed_evidence_lost'] is None else format(100 * m[t]['needed_evidence_lost'], '6.2f')}%"
        f" regret {m[t]['regret']:+.4f}" for t in "RDS"))
