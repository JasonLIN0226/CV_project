"""Shared state construction for an existing train/validation FeatureStore."""
from __future__ import annotations
import hashlib
import importlib.util
from pathlib import Path
import numpy as np
from .core import BEHAVIORS, behavior_update, rollout

FROZEN_ONLINE_SHA256 = '019778036d48baf8577c362d37baa57d48e06786f2e65f3cf42a452a8ac73280'


def load_frozen_behavior(path):
    """Load the audited old online rules only after exact source verification."""
    source = Path(path).read_bytes()
    if hashlib.sha256(source).hexdigest() != FROZEN_ONLINE_SHA256:
        raise ValueError("online behavior source differs from the frozen audit")
    spec = importlib.util.spec_from_file_location('gate_v2_behavior_rules', path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load audited behavior rules")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.update


def deterministic_seed(seed, episode_id, policy):
    raw = f"gate-v2\0{seed}\0{episode_id}\0{policy}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def validate_manifest(manifest):
    rows = manifest["records"]
    if not rows or any(r["split"] not in {"train", "validation"} for r in rows):
        raise ValueError("only original train/validation manifest allowed")
    if any(r["task"] != "gate" for r in rows):
        raise ValueError("Gate-only implementation")
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "validation"]
    if (len(train), len(val)) != (1024, 192):
        raise ValueError("unexpected episode counts")
    for key in ("id", "pair"):
        if {r[key] for r in train} & {r[key] for r in val}:
            raise ValueError(f"train/validation overlap in {key}")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate episode IDs")
    pairs = {}
    for rec in rows:
        pairs.setdefault((rec['split'], rec['pair']), []).append(rec)
    if any(len(p) != 2 or p[0].get('sibling') != p[1]['id']
           or p[1].get('sibling') != p[0]['id'] for p in pairs.values()):
        raise ValueError("manifest requires complete reciprocal sibling pairs")
    return rows


def iter_states(store, load_episode, update_impl, seed):
    """Yield (metadata, OnlineState) without reading labels or future targets.

    Caller controls persistence and must freeze the resulting manifest before
    teacher generation. This is not invoked automatically.
    """
    if seed not in (17, 29, 43):
        raise ValueError("unexpected model seed")
    for rec in validate_manifest(store.manifest):
        query = int(rec["query"])
        projected = store.projected_features(rec)[:query + 1].copy()
        actions = load_episode(store.c, rec, keys=("actions",))["actions"][:query].copy()
        for policy in BEHAVIORS:
            rng_seed = deterministic_seed(seed, rec["id"], policy)
            state, _ = rollout(projected, actions, query,
                behavior_update(policy, np.random.default_rng(rng_seed), update_impl), capture_final=True)
            # Metadata identifies the shared record, never a scorer input.
            metadata = {"split": rec["split"], "episode": rec["id"], "pair": rec["pair"],
                        "seed": seed, "behavior": policy, "rng_seed": rng_seed,
                        "clock": query, "candidates": list(state.candidates)}
            yield metadata, state
