"""Registered Gate scene identity checks and scene-paired route inference."""

from __future__ import annotations

import csv
import gzip
from collections import defaultdict
from pathlib import Path

import numpy as np

from ...data.io import read_json, require_sha256, stable_seed
from ...methods.deletion_cost.training import ARMS as POLICIES
from .protocol import HORIZONS, SEEDS

COHORT_DIR = Path(__file__).resolve().parents[4] / "configs/gate/independent"
SPLITS = ("test_id", "test_long_gap")
GENERATION_SEED = 4747846115943984
TRAINVAL_MANIFEST_SHA = "f32e469f2744940fc2678020fdaa0299113ce4ea027e744afb6b17f9c850e24d"


def verify_fresh_manifest(
    records,
    *,
    generation_seed,
    prior_ids,
    prior_pairs,
    prior_scene_seeds,
    model_frozen_before_generation,
    previously_unopened,
):
    """Fail closed when test identity, freeze, or exposure evidence is missing.

    Boolean provenance assertions must be independently substantiated by the
    caller; this function cannot prove truthfulness of the supplied audit data.
    """
    if (
        generation_seed != GENERATION_SEED
        or model_frozen_before_generation is not True
        or previously_unopened is not True
    ):
        raise ValueError("fresh seed, freeze and non-exposure must be verified")
    if prior_ids is None or prior_pairs is None or prior_scene_seeds is None:
        raise ValueError("complete prior identity sets are required")
    if len(records) != 256:
        raise ValueError("expected 64 sibling pairs per test condition")
    by_id, groups = {}, defaultdict(list)
    for r in records:
        if r.get("split") not in SPLITS or r.get("task") != "gate":
            raise ValueError("unexpected independent-test split or task")
        if (
            r["id"] in by_id
            or r["id"] in prior_ids
            or r["pair"] in prior_pairs
            or r["seed"] in prior_scene_seeds
        ):
            raise ValueError("duplicate or previously exposed identity")
        by_id[r["id"]] = r
        groups[(r["split"], r["pair"])].append(r)
    if any(sum(split == s for split, _ in groups) != 64 for s in SPLITS):
        raise ValueError("expected 64 scene pairs in each condition")
    if len({pair for _, pair in groups}) != 128:
        raise ValueError("scene pair identity crosses conditions")
    for group in groups.values():
        if (
            len(group) != 2
            or group[0]["sibling"] != group[1]["id"]
            or group[1]["sibling"] != group[0]["id"]
            or group[0]["seed"] != group[1]["seed"]
        ):
            raise ValueError("incomplete or mismatched sibling pair")
    if len({group[0]["seed"] for group in groups.values()}) != len(groups):
        raise ValueError("scene seed repeated across distinct pairs")
    return {"episodes": len(records), "pairs_per_condition": 64, "disjoint_id_pair_scene_seed": True}


def paired_route_inference(rows, records, *, bootstrap_seed=0, resamples=10000):
    """Paired D-minus-A scene bootstrap; 97.5% two-sided percentile CI.

    Exactly two siblings, three paired model seeds and two horizons are
    averaged per scene before resampling. Two intervals use Bonferroni family
    alpha 0.05. The only positive claim requires both lower bounds > 0.
    Call verify_fresh_manifest and the independent audit before invoking this
    on real data; output alone is not provenance evidence.
    """
    if bootstrap_seed != 0 or resamples != 10000:
        raise ValueError("bootstrap settings are fixed by protocol")
    recs = {r["id"]: r for r in records}
    if len(recs) != 256 or any(r.get("split") not in SPLITS for r in recs.values()):
        raise ValueError("expected complete fresh two-condition manifest")
    expected = {(s, e, p, h) for s in SEEDS for e in recs for p in POLICIES for h in HORIZONS}
    seen = set()
    grouped = defaultdict(list)
    for row in rows:
        key = (row["seed"], row["episode"], row["condition"], row["horizon"])
        if key not in expected or key in seen:
            raise ValueError("duplicate or unexpected inference row")
        rec = recs[row["episode"]]
        if row.get("split") != rec["split"] or row.get("pair") != rec["pair"]:
            raise ValueError("inference row identity mismatch")
        value = float(row["state_accuracy"])
        if not np.isfinite(value) or value not in (0.0, 1.0):
            raise ValueError("route accuracy must be a finite episode indicator")
        grouped[(rec["split"], rec["pair"], row["condition"])].append(value)
        seen.add(key)
    if seen != expected:
        raise ValueError("independent comparison rows incomplete")
    rng = np.random.default_rng(bootstrap_seed)
    out = {}
    for split in SPLITS:
        pairs = sorted({r["pair"] for r in recs.values() if r["split"] == split})
        if len(pairs) != 64:
            raise ValueError("independent condition requires 64 scenes")
        deltas = []
        for pair in pairs:
            left = grouped[(split, pair, POLICIES[0])]
            right = grouped[(split, pair, POLICIES[1])]
            if len(left) != 12 or len(right) != 12:
                raise ValueError("each scene needs 2 siblings x 3 seeds x 2 horizons")
            deltas.append(float(np.mean(right) - np.mean(left)))
        deltas = np.asarray(deltas)
        samples = rng.integers(0, len(deltas), size=(resamples, len(deltas)))
        boot = deltas[samples].mean(axis=1)
        interval = np.quantile(boot, [0.0125, 0.9875])
        out[split] = {
            "scenes": len(pairs),
            "mean_d_minus_a": float(deltas.mean()),
            "ci_97_5_percent": [float(v) for v in interval],
        }
    return {
        "conditions": out,
        "positive_on_both_conditions": all(out[s]["ci_97_5_percent"][0] > 0 for s in SPLITS),
        "familywise_alpha": 0.05,
        "resamples": resamples,
        "seed": bootstrap_seed,
        "original_gate_passed": False,
    }


def historical_records(trainval_records, prior_manifests):
    records = list(trainval_records)
    for path in prior_manifests:
        if path.name.endswith(".csv.gz"):
            expected = read_json(COHORT_DIR / "cohort.json")["prior_ids_sha256"]
            require_sha256(path, expected, "historical scene identities")
            with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
                historical = [
                    dict(id=row["id"], pair=row["pair"], seed=int(row["seed"]))
                    for row in csv.DictReader(stream)
                ]
        else:
            historical = read_json(path)["records"]
        if not historical:
            raise ValueError(f"Empty historical manifest: {path}")
        records.extend(historical)
    seeds = {r["seed"] for r in records}
    for seed, counts in (
        (9137, {"test_id": 128, "test_long_gap": 128, "test_distractor_shift": 128}),
        (2026102317, {"test_id": 64, "test_long_gap": 64}),
    ):
        expected = {
            stable_seed(seed, "gate", split, i, "position-noise-audit-v1")
            for split, count in counts.items()
            for i in range(count)
        }
        if not expected <= seeds:
            raise ValueError(f"Missing historical scenes for the exposed generation seed {seed}")
    return records


def audit_identities(records, prior, seed, provenance):
    for rec in records:
        scene_seed = stable_seed(seed, "gate", rec["split"], rec["scene_index"], "position-noise-audit-v1")
        if rec["seed"] != scene_seed or rec["scene_index"] not in range(64):
            raise ValueError("Scene identity does not match the registered generation seed")
    return verify_fresh_manifest(
        records,
        generation_seed=seed,
        prior_ids={r["id"] for r in prior},
        prior_pairs={r["pair"] for r in prior},
        prior_scene_seeds={r["seed"] for r in prior},
        model_frozen_before_generation=provenance["model_frozen_before_generation"],
        previously_unopened=provenance["previously_unopened"],
    )


def original_records(root):
    path = require_sha256(
        root / "dataset/manifest.json", TRAINVAL_MANIFEST_SHA, "frozen train/validation manifest"
    )
    return read_json(path)["records"]
