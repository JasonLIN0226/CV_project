"""Paired D/A inference with the released Gate assets and strict validation replay."""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ..data.io import file_hash, read_json, require_sha256, write_json
from ..environments.gate.assets import load_episode_arrays, load_gate, load_normalization, read_gate_episode
from ..environments.gate.bundle import TEAM_BASIS_SHA256, TEAM_PREDICTOR_SHA256
from ..environments.gate.holdout_protocol import (
    COHORT_DIR,
    SPLITS,
    audit_identities,
    historical_records,
    original_records,
    paired_route_inference,
)
from ..environments.gate.protocol import HORIZONS, SEEDS, predict_selection
from ..methods.deletion_cost.retention import scorer_selections
from ..methods.deletion_cost.training import ARMS, frozen_scorers
from ..predictors.phase1.data import dataset_signature
from ..predictors.phase1.evaluation import pair_diagnostic
from .metrics.gate import metric_rows

REFERENCE_SHA = "a2a6c3c2ebca33c5540e60de2e50ab474415b91ad978c850f8c864073dc40b04"
TEAM_FILES = {
    "features/index.json": "4a447041b4544aabe11fc28e5b66a1405b4bcaf80eee3fabe2288205b49012ba",
    "features/normalization.npz": "a6f71f62c8ab56d48452ebb8766ce0637bed4c33cb5dd4c029a9874933bd900c",
    "probes/gate_state_decoder.npz": "4bf6b5c82f6d43a204079607e12c223d338b444e163be3bf235fd69bed91d67f",
}
STUDENT_SHA = {
    17: "56b7e8fcbba629bd80ebf80168c71efd0bca5586402466fc555fe0d0e85dd056",
    29: "d7aead64fed565df0c7a97d14e30c03f65699f4b5fbed5727a8fada7f186f7dd",
    43: "3cdf05fc92348d511ef81f037939e3c38a2e0cec1b38323099c15c96e92f5f9f",
}


def holdout_inputs(assets, root, prior_manifests, provenance):
    manifest = read_json(require_sha256(root / "dataset/manifest.json", provenance["manifest_sha256"]))
    records = manifest["records"]
    if {r["split"] for r in records} != set(SPLITS):
        raise ValueError("Independent mode requires both holdout splits and rejects train/validation")
    config = read_json(root / "resolved_config.json")
    prior = historical_records(assets.records, prior_manifests)
    audit_identities(records, prior, config["data"]["seed"], provenance)
    audit = read_json(root / "features/pair_feature_audit.json")
    index_sha = provenance["index_sha256"]
    index = read_json(require_sha256(root / "features/index.json", index_sha))
    training_index = read_json(assets.root / "features/index.json")
    if (
        manifest["signature"] != dataset_signature(config)
        or index["dataset_signature"] != manifest["signature"]
        or index["encoder_hash"] != training_index["encoder_hash"]
        or set(index["episodes"]) != {r["id"] for r in records}
        or not audit["passed"]
        or not audit["exact_recent_feature_equality"]
        or audit["signature"] != index["signature"]
        or audit["index_sha256"] != index_sha
    ):
        raise ValueError("Holdout provenance or feature-pair audit mismatch")
    for rec in records:
        feature = index["episodes"][rec["id"]]
        if feature["split"] != rec["split"] or feature["source_sha256"] != rec["obs_sha256"]:
            raise ValueError(f"Holdout feature source mismatch: {rec['id']}")
    norm = load_normalization(assets.root, assets.c, training_index, assets.records)
    episodes = [read_gate_episode(root, assets.c, rec, index, norm, assets.basis) for rec in records]
    return records, episodes


@torch.inference_mode()
def score(assets, students, data_root, records, episodes):
    by_id = {e.id: e for e in episodes}
    states = {r["id"]: load_episode_arrays(data_root, r, "eval", ("states",))["states"] for r in records}
    rows = []
    for seed in SEEDS:
        require_sha256(students / f"student_{seed}.pt", STUDENT_SHA[seed], "frozen scorer pair")
        scorers, _ = frozen_scorers(students, seed, assets.device)
        predictor = assets.predictor(seed)
        for policy in ARMS:
            selections = scorer_selections(scorers[policy], episodes, audit=True)
            predictions = predict_selection(predictor, episodes, selections, assets.device)
            metrics = metric_rows(assets, episodes, predictions, selections, policy, seed)
            for i, (e, pred) in enumerate(zip(episodes, predictions)):
                decoded = assets.decoder.predict(pred)
                sibling = by_id[e.sibling]
                for j, h in enumerate(HORIZONS):
                    m = metrics[i * len(HORIZONS) + j]
                    t = e.q + h
                    pair_acc, margin, separation = pair_diagnostic(pred[j], e.z[t], sibling.z[sibling.q + h])
                    rows.append(
                        {
                            "condition": policy,
                            "split": e.split,
                            "episode": e.id,
                            "pair": e.pair,
                            "seed": seed,
                            "horizon": h,
                            "selected_indices": ";".join(map(str, m["kept"])),
                            "predicted_route_class": int(decoded[j, 0] > 0),
                            "true_route_class": int(e.classes[t] > 0),
                            "state_accuracy": m["route_accuracy"],
                            "event_hit": int(m["cue_kept"]),
                            "evidence_hit": int(set(e.evidence).issubset(m["kept"])),
                            "normalized_mse": m["normalized_mse"],
                            "pair_accuracy": pair_acc,
                            "pair_margin": margin,
                            "target_pair_distance": separation,
                            "state_position_mse": float(np.mean((decoded[j, 1:] - states[e.id][t]) ** 2)),
                        }
                    )
        print(f"Scored both policies on {len(episodes)} episodes for seed {seed}", flush=True)
    return rows


def keyed_validation_rows(rows):
    indexed = {}
    for r in rows:
        key = (r["condition"], r["episode"], r["seed"], r["horizon"])
        if r["split"] != "validation" or key in indexed:
            raise ValueError(f"Duplicate or non-validation row: {key}")
        indexed[key] = r
    return indexed


def compare_validation(rows, reference, records):
    actual = keyed_validation_rows(rows)
    archived = keyed_validation_rows(r for r in reference if r["condition"] in ARMS)
    expected = {(p, r["id"], s, h) for r in records for p in ARMS for s in SEEDS for h in HORIZONS}
    if len(records) != 192 or actual.keys() != expected or archived.keys() != expected:
        raise ValueError("Validation comparison requires all 2,304 matched rows")
    comparison = {}
    for policy in ARMS:
        keys = [k for k in actual if k[0] == policy]
        correct = sum(int(actual[k]["state_accuracy"]) for k in keys)
        for k in keys:
            r, ref = actual[k], archived[k]
            if ref["pair"] != r["pair"]:
                raise ValueError("Reference pair identity mismatch")
            # Published rows carry binary correctness. With the pinned true class this uniquely
            # identifies the original prediction, even for mistakes.
            r["reference_route_class"] = (
                r["true_route_class"] if ref["state_accuracy"] else 1 - r["true_route_class"]
            )
            r["reference_selected_indices"] = ref["selected_indices"]
        comparison[policy] = {
            "correct": correct,
            "rows": len(keys),
            "route_accuracy": correct / len(keys),
            "route_mismatches": sum(
                actual[k]["predicted_route_class"] != actual[k]["reference_route_class"] for k in keys
            ),
            "selection_mismatches": sum(
                actual[k]["selected_indices"] != archived[k]["selected_indices"] for k in keys
            ),
            "max_abs_normalized_mse_diff": max(
                abs(actual[k]["normalized_mse"] - archived[k]["normalized_mse"]) for k in keys
            ),
        }
    return comparison


def evaluate(args, config):
    root = Path(config["assets"]["root"])
    original_records(root)
    for name, sha in TEAM_FILES.items():
        require_sha256(root / name, sha)
    validation = args.mode == "validation"
    assets = load_gate(
        root,
        args.device,
        predictor_sha256=TEAM_PREDICTOR_SHA256,
        basis_sha256=TEAM_BASIS_SHA256,
        episode_splits=("validation",) if validation else (),
    )
    if validation:
        source = root
        records = [r for r in assets.records if r["split"] == "validation"]
        episodes = assets.episodes
    else:
        source = args.test_data
        provenance = read_json(args.provenance)
        if args.provenance.resolve() == COHORT_DIR / "cohort.json":
            # The data registration predates encoding. Use the receipt for this feature cache.
            provenance["index_sha256"] = read_json(source / "features/pair_feature_audit.json")["index_sha256"]
        records, episodes = holdout_inputs(assets, source, args.prior_manifest, provenance)
    rows = score(assets, Path(config["assets"]["students"]), source, records, episodes)
    summary = {
        "mode": args.mode,
        "rows": len(rows),
        "episodes": len(episodes),
        "pairs": len({e.pair for e in episodes}),
        "seeds": SEEDS,
        "horizons": HORIZONS,
        "device": str(assets.device),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "training": False,
        "data_generated": False,
        "dataset_manifest_sha256": file_hash(source / "dataset/manifest.json"),
        "feature_index_sha256": file_hash(source / "features/index.json"),
    }
    if validation:
        ref = require_sha256(Path(config["assets"]["reference_rows"]), REFERENCE_SHA, "published validation")
        summary["reference_sha256"] = REFERENCE_SHA
        summary["comparison"] = compare_validation(rows, read_json(ref)["rows"], records)
        summary["passed"] = all(
            not (v["route_mismatches"] or v["selection_mismatches"]) for v in summary["comparison"].values()
        )
    else:
        summary["inference"] = paired_route_inference(rows, records)
    groups = defaultdict(list)
    for r in rows:
        groups[r["split"], r["condition"], r["seed"], r["horizon"]].append(r)
    summary["per_seed_horizon"] = [
        {
            "split": split,
            "policy": p,
            "seed": s,
            "horizon": h,
            "rows": len(group),
            **{
                metric: float(np.mean([r[metric] for r in group]))
                for metric in (
                    "state_accuracy",
                    "event_hit",
                    "normalized_mse",
                    "pair_accuracy",
                    "state_position_mse",
                )
            },
        }
        for (split, p, s, h), group in sorted(groups.items())
    ]
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / "rows.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.out / "summary.json", summary)
    return summary
