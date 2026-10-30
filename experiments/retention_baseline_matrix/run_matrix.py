"""Run a fixed matched matrix of memory-retention policies on existing models.

Designed for the NYU course HPC environment. The scientific predictor,
checkpoints, feature cache and normalization are reused without training or
encoding.
"""
from __future__ import annotations

import argparse
import collections
import copy
import csv
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


POLICIES = (
    "recent_only",
    "full_history",
    "privileged_small_budget",
    "random_small_budget",
    "wrong_evidence",
    "fifo_b2",
    "reservoir_b2",
    "temporal_buckets_b2",
    "visual_diversity_b2",
    "prediction_surprise_b2",
    "attention_b2",
)
SELECTION_POLICIES = POLICIES[5:]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_policy_module(path: Path):
    spec = importlib.util.spec_from_file_location("retention_policy_rules", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load policy module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_source_lock(protocol_path: Path) -> dict:
    """Verify all frozen matrix files before loading features or checkpoints."""
    root = protocol_path.parent
    lock_path = root / "source_lock.json"
    lock = json.loads(lock_path.read_text())
    if lock.get("status") != "FROZEN_MATCHED_RETENTION_BASELINE_MATRIX_V2":
        raise ValueError("Unexpected source-lock status")
    verified = {}
    for relative, expected in lock["files"].items():
        observed = sha256(root / relative)
        verified[relative] = observed
        if observed != expected:
            raise ValueError(f"Frozen source mismatch: {relative}")
    return {"path": str(lock_path), "sha256": sha256(lock_path), "files": verified}


def sample_with_indices(
    store,
    rec: dict,
    history_indices,
    load_episode,
    selected_evidence,
) -> tuple[dict, np.ndarray, dict]:
    q = int(rec["query"])
    recent = int(store.c["data"]["recent"])
    capacity = int(store.c["data"]["budget"])
    idx = np.asarray(sorted(set(map(int, history_indices))), dtype=np.int64)
    valid = np.arange(q - recent + 1, dtype=np.int64)
    if len(idx) > capacity or any(int(v) not in valid for v in idx):
        raise ValueError("Illegal history selection")
    z = store.features(rec)
    times = np.concatenate([np.arange(q - recent + 1, q + 1), idx])
    own = load_episode(store.c, rec, keys=("actions",))
    prefix = np.concatenate([np.zeros((1, 2), np.float32), np.cumsum(own["actions"][:q], axis=0)])
    sample = {
        "z": np.concatenate([z[q - recent + 1 : q + 1], z[idx]]),
        "age": (q - times).astype(np.float32),
        "type": np.concatenate(
            [np.zeros(recent, np.int64), np.ones(len(idx), np.int64)]
        ),
        "past_actions": prefix[q] - prefix[times],
        "actions": own["actions"][q : q + max(store.c["data"]["horizons"])].copy(),
        "current": z[q].copy(),
    }
    if rec["task"] in store.bases:
        projected = store.projected_features(rec)
        sample["projected_z"] = projected[times].copy()
        sample["projected_current"] = projected[q].copy()
    target = z[q + np.asarray(store.c["data"]["horizons"])].copy()
    raw_evidence = load_episode(store.c, rec, True, keys=("evidence",))["evidence"]
    evidence = selected_evidence(store.c, rec, raw_evidence)
    evidence_hit = set(map(int, evidence)).issubset(set(map(int, idx)))
    return sample, target, {
        "indices": idx,
        "history_count": len(idx),
        "evidence_hit": evidence_hit,
    }


def recent_sample(store, rec: dict, query: int, load_episode) -> dict:
    recent = int(store.c["data"]["recent"])
    if query - recent + 1 < 0:
        raise ValueError("Recent context starts before episode")
    z = store.features(rec)
    times = np.arange(query - recent + 1, query + 1)
    own = load_episode(store.c, rec, keys=("actions",))
    prefix = np.concatenate([np.zeros((1, 2), np.float32), np.cumsum(own["actions"][:query], axis=0)])
    sample = {
        "z": z[times].copy(),
        "age": (query - times).astype(np.float32),
        "type": np.zeros(recent, dtype=np.int64),
        "past_actions": prefix[query] - prefix[times],
        "actions": own["actions"][query : query + max(store.c["data"]["horizons"])].copy(),
        "current": z[query].copy(),
    }
    if rec["task"] in store.bases:
        projected = store.projected_features(rec)
        sample["projected_z"] = projected[times].copy()
        sample["projected_current"] = projected[query].copy()
    return sample


@torch.inference_mode()
def prediction_surprises(store, rec, model, device, collate, load_episode) -> dict[int, float]:
    q = int(rec["query"])
    recent = int(store.c["data"]["recent"])
    valid = np.arange(q - recent + 1, dtype=np.int64)
    requests, owners = [], []
    z = store.features(rec)
    for timestamp in valid:
        for hi, horizon in enumerate(store.c["data"]["horizons"]):
            source_query = int(timestamp) - int(horizon)
            if source_query - recent + 1 < 0:
                continue
            requests.append(recent_sample(store, rec, source_query, load_episode))
            owners.append((int(timestamp), hi))
    if not requests:
        return {}
    predictions = model(collate(requests, device)).cpu().numpy()
    values = collections.defaultdict(list)
    for (timestamp, horizon_index), prediction in zip(owners, predictions):
        values[timestamp].append(float(np.mean((prediction[horizon_index] - z[timestamp]) ** 2)))
    return {timestamp: float(np.mean(parts)) for timestamp, parts in values.items()}


@torch.inference_mode()
def attention_indices(store, rec, model, poly, load_episode) -> list[int]:
    q = int(rec["query"])
    recent = int(store.c["data"]["recent"])
    capacity = int(store.c["data"]["budget"])
    valid = poly.legal_history(q, recent)
    if len(valid) <= capacity:
        return valid.tolist()
    projected = store.projected_features(rec)
    age = q - valid
    nearest = age.min()
    span = np.maximum(age.max() - nearest, 1)
    own = load_episode(store.c, rec, keys=("actions",))
    prefix = np.concatenate([np.zeros((1, 2), np.float32), np.cumsum(own["actions"][:q], axis=0)])
    past = prefix[q] - prefix[valid]
    position = (age - nearest) / span
    frame_key = np.concatenate(
        [
            projected[valid] - projected[q],
            position[:, None],
            past / (1.0 + age[:, None]),
        ],
        axis=1,
    )
    x = torch.tensor(frame_key[None], dtype=torch.float32, device=model.projection.device)
    mask = torch.ones((1, len(valid)), dtype=torch.bool, device=model.projection.device)
    # Selection happens before future actions are available. Use only the last
    # observed action and the observed recent-action prefix as a common context.
    observed_actions = own["actions"][max(0, q - recent) : q]
    if len(observed_actions):
        last_action = observed_actions[-1]
        action_sum = observed_actions.sum(0)
        action_mean = observed_actions.mean(0)
    else:
        last_action = np.zeros(2, dtype=np.float32)
        action_sum = np.zeros(2, dtype=np.float32)
        action_mean = np.zeros(2, dtype=np.float32)
    context = np.concatenate(
        [last_action, action_sum, action_mean, np.asarray([0.0], np.float32)]
    )
    context = torch.tensor(context[None], dtype=torch.float32, device=model.projection.device)
    # ContentWeights exposes one attention distribution per learned query.
    # The predictor's frame weight is their equal-head/query average.
    score = model.frames(x, mask, context)[0].mean(0).cpu().numpy()
    return poly.top_scores(q, recent, capacity, valid, score)


def manual_indices(c, store, poly, rec, load_episode, selected_evidence) -> list[int]:
    q = int(rec["query"])
    recent = int(store.c["data"]["recent"])
    capacity = int(store.c["data"]["budget"])
    valid = poly.legal_history(q, recent)
    raw = load_episode(c, rec, True, keys=("evidence",))["evidence"]
    evidence = [int(v) for v in selected_evidence(c, rec, raw)]
    chosen = evidence[:capacity]
    chosen.extend(int(v) for v in valid[::-1] if int(v) not in chosen)
    return poly.sorted_capacity(chosen[:capacity], capacity)


def reservoir_seed(c, rec, stable_seed, policy: str) -> int:
    return stable_seed(
        c["evaluation"]["random_seed"],
        rec["pair"],
        rec["id"],
        policy,
    )


def policy_indices(
    policy,
    c,
    store,
    rec,
    model,
    device,
    poly,
    load_episode,
    selected_evidence,
    stable_seed,
    collate_fn,
    surprise_cache,
) -> tuple[list[int], str]:
    q, recent, capacity = int(rec["query"]), int(c["data"]["recent"]), int(c["data"]["budget"])
    valid = poly.legal_history(q, recent)
    if policy == "recent_only":
        return [], "original evaluator policy"
    if policy == "full_history":
        return valid.tolist(), "original evaluator policy"
    if policy in {"privileged_small_budget", "wrong_evidence"}:
        return manual_indices(c, store, poly, rec, load_episode, selected_evidence), "original evidence-keyframe policy"
    if policy == "random_small_budget":
        rng = np.random.default_rng(
            stable_seed(c["evaluation"]["random_seed"], rec["pair"], policy)
        )
        return np.sort(rng.choice(valid, size=capacity, replace=False)).tolist(), "original evaluator policy"
    if policy == "fifo_b2":
        return poly.fifo(q, recent, capacity), "newest legal history entries"
    if policy == "reservoir_b2":
        seed = reservoir_seed(c, rec, stable_seed, policy)
        return poly.reservoir(q, recent, capacity, seed), "deterministic reservoir"
    if policy == "temporal_buckets_b2":
        return poly.temporal_buckets(q, recent, capacity), "newest timestamp in each temporal bucket"
    if policy == "visual_diversity_b2":
        projected = store.projected_features(rec)
        return poly.visual_diversity(
            q, recent, capacity, projected, projected[q]
        ), "greedy farthest point in projected feature space"
    if policy == "prediction_surprise_b2":
        key = rec["id"]
        if key not in surprise_cache:
            surprise_cache[key] = prediction_surprises(
                store, rec, model, device, collate_fn, load_episode
            )
        table = surprise_cache[key]
        scores = np.asarray([table.get(int(v), -np.inf) for v in valid])
        available = np.isfinite(scores)
        if int(available.sum()) < capacity:
            return poly.fifo(q, recent, capacity), "surprise fallback to FIFO"
        ranked = [
            int(valid[index])
            for index in sorted(
                np.flatnonzero(available),
                key=lambda i: (-float(scores[i]), -int(valid[i])),
            )
        ]
        return poly.sorted_capacity(ranked[:capacity], capacity), "causal recent-only forecast surprise"
    if policy == "attention_b2":
        return (
            attention_indices(store, rec, model, poly, load_episode),
            "frozen predictor frame attention with causal observed-action context",
        )
    raise ValueError(f"Unknown policy {policy}")


def maybe_wrong_sample(
    store, rec, idx, policy, load_episode, selected_evidence
):
    # Reuse the parent evaluator's input boundary for controls it already owns.
    # In particular, full_history is intentionally variable-capacity and must
    # not pass through the custom B=2 capacity assertion below.
    if policy in {
        "recent_only",
        "full_history",
        "privileged_small_budget",
        "random_small_budget",
        "wrong_evidence",
    }:
        return store.sample(rec, policy)
    return sample_with_indices(
        store, rec, idx, load_episode, selected_evidence
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--policy-module", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing overwrite: {args.output}")
    started = time.monotonic()
    args.output.mkdir(parents=True)
    protocol = json.loads(args.protocol.read_text())
    if protocol["status"] != "FROZEN_MATCHED_RETENTION_BASELINE_MATRIX_V2":
        raise ValueError("Unexpected protocol")
    source_lock = verify_source_lock(args.protocol)
    if args.device == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one CUDA device required")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    source = args.control / "src"
    if not source.is_dir():
        raise FileNotFoundError(source)
    sys.path.insert(0, str(source))

    from phase1.config import SPLITS
    from phase1.data import load_episode
    from phase1.evaluation import active_tasks, pair_diagnostic, summarize
    from phase1.inputs import FeatureStore, collate, selected_evidence
    from phase1.probes import load_decoder
    from phase1.repair_evaluation import all_models_completion, load_current_reader
    from phase1.stats import paired_improvement
    from phase1.utils import file_hash, read_json, write_csv, write_json, seed_all, stable_seed
    from phase1.proposal_trial import verify_control

    poly = load_policy_module(args.policy_module)
    completion = all_models_completion(args.control, args.training_root)
    if not completion["complete"] or len(completion["models"]) != 6:
        raise ValueError("All six completed models are required")
    p, lock, configs = verify_control(args.control)
    seed_all(17, 4)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    atomic_json(
        args.output / "attempt.json",
        {
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "protocol_sha256": sha256(args.protocol),
            "policy_module_sha256": sha256(args.policy_module),
            "run_matrix_sha256": sha256(Path(__file__)),
            "source_lock": source_lock,
            "all_six_models_verified": True,
            "device": str(device),
            "no_training": True,
            "no_fitting": True,
            "no_encoder_forward": True,
        },
    )

    raw = []
    latency = []
    selection_audit = []
    all_surprise = {}
    for task, c in configs.items():
        store = FeatureStore(c, max_cached=2 * max(c["data"]["pairs"][s] for s in SPLITS[1:]))
        decoders = load_decoder(c, task)
        for seed in p["seeds"]:
            model, checkpoint = load_current_reader(
                args.control, args.training_root, seed, store, completion
            )
            model = model.to(device).eval()
            surprise_cache = {}
            for split in SPLITS[1:]:
                rows = [
                    rec
                    for rec in store.manifest["records"]
                    if rec["task"] == task and rec["split"] == split
                ]
                for policy in POLICIES:
                    for start in range(0, len(rows), c["training"]["batch_size"]):
                        part = rows[start : start + c["training"]["batch_size"]]
                        samples = []
                        choices = []
                        rules = []
                        for rec in part:
                            indices, rule = policy_indices(
                                policy, c, store, rec, model, device, poly,
                                load_episode, selected_evidence, stable_seed, collate, surprise_cache
                            )
                            choices.append(indices)
                            rules.append(rule)
                            samples.append(
                                maybe_wrong_sample(
                                    store, rec, indices, policy,
                                    load_episode, selected_evidence
                                )
                            )
                        batch = collate([sample[0] for sample in samples], device)
                        with torch.inference_mode():
                            prediction = model(batch).cpu().numpy()
                        if not np.isfinite(prediction).all():
                            raise RuntimeError("Non-finite prediction")
                        for rec, chosen, rule, sample, pred in zip(
                            part, choices, rules, samples, prediction
                        ):
                            _, target, info = sample
                            meta = load_episode(c, rec, True)
                            sibling = store.records[rec["sibling"]]
                            other = store.features(sibling)[
                                rec["query"] + np.asarray(c["data"]["horizons"])
                            ]
                            state = decoders.predict(pred)
                            D, R, K = pred.shape[-1], c["data"]["recent"], len(chosen)
                            selection_audit.append(
                                {
                                    "seed": seed,
                                    "task": task,
                                    "split": split,
                                    "condition": policy,
                                    "episode": rec["id"],
                                    "pair": rec["pair"],
                                    "query": rec["query"],
                                    "indices": ";".join(map(str, chosen)),
                                    "rule": rule,
                                }
                            )
                            for j, horizon in enumerate(c["data"]["horizons"]):
                                p_raw = (
                                    pred[j] * store.norm["std"] + store.norm["mean"]
                                )
                                y_raw = (
                                    target[j] * store.norm["std"] + store.norm["mean"]
                                )
                                cosine = 1.0 - float(
                                    np.dot(p_raw, y_raw)
                                    / max(
                                        np.linalg.norm(p_raw)
                                        * np.linalg.norm(y_raw),
                                        1e-12,
                                    )
                                )
                                accuracy, margin, separation = pair_diagnostic(
                                    pred[j], target[j], other[j]
                                )
                                target_index = rec["query"] + horizon
                                raw.append(
                                    {
                                        "seed": seed,
                                        "task": task,
                                        "split": split,
                                        "condition": policy,
                                        "horizon": horizon,
                                        "episode": rec["id"],
                                        "pair": rec["pair"],
                                        "gap": rec["gap"],
                                        "query": rec["query"],
                                        "latent_mse": float(
                                            np.mean((p_raw - y_raw) ** 2)
                                        ),
                                        "normalized_mse": float(
                                            np.mean((pred[j] - target[j]) ** 2)
                                        ),
                                        "cosine_distance": cosine,
                                        "pair_accuracy": accuracy,
                                        "pair_margin": margin,
                                        "target_pair_distance": separation,
                                        "state_accuracy": float(
                                            (state[j, 0] > 0)
                                            == (
                                                meta["classes"][target_index]
                                                > 0
                                            )
                                        ),
                                        "state_position_mse": float(
                                            np.mean(
                                                (
                                                    state[j, 1:]
                                                    - meta["states"][target_index]
                                                )
                                                ** 2
                                            )
                                        ),
                                        "history_count": K,
                                        "history_feature_bytes": K * D * 4,
                                        "recent_feature_bytes": R * D * 4,
                                        "projected_history_bytes": K
                                        * model.projection.shape[1]
                                        * 4,
                                        "timestamp_type_mask_bytes": (K + R)
                                        * 13,
                                        "past_action_bytes": (K + R) * 8,
                                        "future_action_bytes": horizon * 2 * 4,
                                        "evidence_hit": float(info["evidence_hit"]),
                                        "state_pred_x": float(state[j, 1]),
                                        "state_pred_y": float(state[j, 2]),
                                        "state_true_x": float(
                                            meta["states"][target_index, 0]
                                        ),
                                        "state_true_y": float(
                                            meta["states"][target_index, 1]
                                        ),
                                    }
                                )
            atomic_json(
                args.output / "progress.json",
                {
                    "task": task,
                    "seed": seed,
                    "completed_seed": True,
                    "raw_rows": len(raw),
                    "policy_count": len(POLICIES),
                    "phase1_go": False,
                },
            )

    atomic_csv(args.output / "raw_records.csv", raw)
    summaries = {
        task: summarize(
            c,
            [row for row in raw if row["task"] == task],
        )
        for task, c in configs.items()
    }
    merged_summary = {
        "per_seed": [
            row
            for task in configs
            for row in summaries[task]["per_seed"]
        ],
        "comparisons": [
            row
            for task in configs
            for row in summaries[task]["comparisons"]
        ],
        "statistics": summaries["gate"]["statistics"],
    }
    atomic_json(
        args.output / "metrics.json",
        {
            "mode": "matched_retention_baseline_matrix",
            "raw_sha256": sha256(args.output / "raw_records.csv"),
            "source_control_sha256": sha256(args.control / "source_lock.json"),
            "checkpoints": completion["models"],
            "data_source": {
                task: configs[task]["output"] for task in configs
            },
            "reader": "CenteredTransportPredictor",
            "policies": list(POLICIES),
            "phase1_go": False,
            **merged_summary,
        },
    )
    atomic_csv(args.output / "selection_audit.csv", selection_audit)

    matrix = []
    for task, c in configs.items():
        for split in SPLITS[1:]:
            for horizon in c["data"]["horizons"]:
                for seed in p["seeds"] + ["all_seeds_mean"]:
                    data = [
                        r
                        for r in raw
                        if r["task"] == task
                        and r["split"] == split
                        and r["horizon"] == horizon
                        and (seed == "all_seeds_mean" or r["seed"] == seed)
                    ]
                    if seed == "all_seeds_mean":
                        grouped = collections.defaultdict(list)
                        for row in data:
                            grouped[(row["condition"], row["episode"])].append(row)
                        data = []
                        for rows in grouped.values():
                            merged = copy.deepcopy(rows[0])
                            for metric in (
                                "normalized_mse",
                                "latent_mse",
                                "state_position_mse",
                                "state_accuracy",
                                "pair_accuracy",
                                "history_count",
                                "history_feature_bytes",
                            ):
                                merged[metric] = float(
                                    np.mean([row[metric] for row in rows])
                                )
                            data.append(merged)
                    by = {
                        (row["condition"], row["episode"]): row
                        for row in data
                    }
                    episodes = sorted(
                        {
                            row["episode"]
                            for row in data
                            if row["condition"] == "privileged_small_budget"
                        }
                    )
                    for policy in (
                        "recent_only",
                        "random_small_budget",
                        "full_history",
                        *SELECTION_POLICIES,
                    ):
                        manual = [by[("privileged_small_budget", e)] for e in episodes]
                        alternative = [by[(policy, e)] for e in episodes]
                        for metric in (
                            "normalized_mse",
                            "state_position_mse",
                        ):
                            lower_better = True
                            a = [float(row[metric]) for row in alternative]
                            b = [float(row[metric]) for row in manual]
                            floor = (
                                c["decision"]["position_baseline_floor"]
                                if metric == "state_position_mse"
                                else 1e-8
                            )
                            absolute_gain = (
                                c["decision"]["absolute_position_mse_gain"]
                                if metric == "state_position_mse"
                                else c["decision"]["absolute_latent_gain"]
                            )
                            improvement = paired_improvement(
                                a,
                                b,
                                [row["pair"] for row in alternative],
                                c["evaluation"]["bootstrap"],
                                floor,
                                absolute_gain,
                            )
                            matrix.append(
                                {
                                    "task": task,
                                    "split": split,
                                    "horizon": horizon,
                                    "seed": seed,
                                    "policy": policy,
                                    "metric": metric,
                                    "manual_vs_policy": improvement,
                                }
                            )
    atomic_json(args.output / "manual_comparisons.json", matrix)
    atomic_json(
        args.output / "complete.json",
        {
            "status": "MATCHED_RETENTION_BASELINE_MATRIX_COMPLETE",
            "raw_rows": len(raw),
            "conditions": list(POLICIES),
            "models": 6,
            "device": str(device),
            "elapsed_seconds": time.monotonic() - started,
            "phase1_go": False,
        },
    )
    print(
        json.dumps(
            {
                "status": "MATCHED_RETENTION_BASELINE_MATRIX_COMPLETE",
                "raw_rows": len(raw),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
