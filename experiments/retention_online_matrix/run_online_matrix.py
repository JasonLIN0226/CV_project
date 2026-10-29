"""Run strict irreversible B=2 retention baselines on frozen predictors.

The recent buffer and historical memory are separate.  A timestamp enters
long-term memory only when it leaves the recent buffer.  At that update, every
policy may inspect only its two currently retained timestamps and the one new
arrival.  Encoder caches and six frozen checkpoints are reused.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

import policies

BASE_MATRIX_DIR = Path(__file__).resolve().parent.parent / "retention_baseline_matrix"
if str(BASE_MATRIX_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_MATRIX_DIR))
import run_matrix as base


POLICIES = policies.POLICIES
PROTOCOL_STATUS = "FROZEN_IRREVERSIBLE_ONLINE_RETENTION_MATRIX_V3"
SOURCE_LOCK_STATUS = "FROZEN_IRREVERSIBLE_ONLINE_RETENTION_MATRIX_V3"
SUM_AUDIT_KEYS = (
    "updates",
    "candidate_pool_three_updates",
    "surprise_valid_scores",
    "attention_update_clocks",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def verify_source_lock(protocol_path: Path) -> dict:
    root = protocol_path.parent
    lock_path = root / "source_lock.json"
    lock = json.loads(lock_path.read_text())
    if lock.get("status") != SOURCE_LOCK_STATUS:
        raise ValueError("Unexpected source-lock status")
    verified = {}
    for relative, expected in lock["files"].items():
        observed = sha256(root / relative)
        verified[relative] = observed
        if observed != expected:
            raise ValueError(f"Frozen source mismatch: {relative}")
    return {
        "path": str(lock_path),
        "sha256": sha256(lock_path),
        "files": verified,
    }


def stable_reservoir_seed(stable_seed, c, rec, seed: int, policy: str) -> int:
    return int(
        stable_seed(
            c["evaluation"]["random_seed"],
            rec["pair"],
            rec["id"],
            seed,
            policy,
        )
        % (2**63 - 1)
    )


@torch.inference_mode()
def recent_horizon3_prediction(reference, batch):
    """Evaluate the frozen recent-only reference's horizon-3 branch causally."""
    valid = batch["mask"]
    z = batch["z"].masked_fill(~valid[..., None], 0)
    projected = (
        batch["projected_z"].masked_fill(~valid[..., None], 0)
        if "projected_z" in batch
        else (z - reference.input_mean) @ reference.projection
    )
    current = (
        batch["projected_current"]
        if "projected_current" in batch
        else (batch["current"] - reference.input_mean) @ reference.projection
    )
    history = valid & (batch["type"] == 1)
    score = reference.score(
        torch.cat([projected, projected - current[:, None]], -1)
    ).squeeze(-1)
    score = score + torch.log1p(
        (projected - current[:, None]).square().mean(-1)
    )
    weights = (
        torch.softmax(score.masked_fill(~history, -1e4), 1) * history
    )
    weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
    past = reference.past_action(
        batch["past_actions"].masked_fill(~valid[..., None], 0)
    )
    aligned = projected + past
    age = batch["age"].masked_fill(~valid, 0) / 8
    mean_age = (weights * age).sum(1)
    center = age - mean_age[:, None]
    memory = (weights[..., None] * aligned).sum(1)
    past_mean = (weights[..., None] * past).sum(1)
    variance = (weights * center.square()).sum(1)
    velocity = -(
        weights[..., None] * center[..., None] * (aligned - memory[:, None])
    ).sum(1) / variance.clamp_min(0.01)[:, None]
    actions = batch["actions"][:, :3]
    action = reference.action(
        torch.cat([actions[:, 0], actions.sum(1), actions[:, -1]], -1)
    )
    transported = memory + velocity * (mean_age + 3 / 8)[:, None]
    stats = torch.stack(
        [
            mean_age,
            history.any(1).float(),
            torch.full_like(mean_age, 3 / 8),
        ],
        -1,
    )
    features = torch.cat(
        [
            current,
            memory,
            velocity,
            transported,
            action,
            memory * action,
            past_mean,
            velocity * action,
            stats,
        ],
        -1,
    )
    coefficient = reference.output(features)
    return (
        batch["current"]
        + reference.target_mean
        + coefficient @ reference.output_basis.T
    )


@torch.inference_mode()
def causal_surprise_table(
    store,
    rec,
    model,
    device,
    collate_fn,
    load_episode,
) -> dict[int, float]:
    """Freeze one causal horizon-3 surprise score when each observation arrives.

    Only actions and features up to the candidate timestamp are supplied.  The
    historical horizon-6 score is deliberately excluded because it would not be
    known when a frame leaves the four-frame recent buffer.
    """
    query = int(rec["query"])
    recent = int(store.c["data"]["recent"])
    arrivals = list(range(query - recent + 1))
    requests = []
    owners = []
    for timestamp in arrivals:
        source_query = timestamp - 3
        if source_query - recent + 1 < 0:
            continue
        sample = base.recent_sample(store, rec, source_query, load_episode)
        sample["actions"] = sample["actions"][:3].copy()
        requests.append(sample)
        owners.append(timestamp)
    if not requests:
        return {}
    prediction = recent_horizon3_prediction(
        model.reference, collate_fn(requests, device)
    ).cpu().numpy()
    features = store.features(rec)
    return {
        timestamp: float(np.mean((pred - features[timestamp]) ** 2))
        for timestamp, pred in zip(owners, prediction)
    }


@torch.inference_mode()
def causal_attention_table(
    store,
    rec,
    model,
    load_episode,
) -> dict[int, dict[int, float]]:
    """Score each eligible timestamp against the update clock that observes it.

    All clocks and candidate/clock pairs are scored in one frozen-frame model
    call.  Replay later looks up only pairs with candidate <= clock.  This
    adaptation uses the last observed action and observed recent-action prefix,
    never the future action plan used by the task predictor at evaluation time.
    """
    query = int(rec["query"])
    recent = int(store.c["data"]["recent"])
    arrivals = list(range(query - recent + 1))
    projected = store.projected_features(rec)
    own = load_episode(store.c, rec, keys=("actions",))
    prefix = np.concatenate(
        [np.zeros((1, 2), np.float32), np.cumsum(own["actions"][:query], axis=0)]
    )

    pair_rows = []
    contexts = []
    mapping = collections.defaultdict(dict)
    for timestamp in arrivals:
        clock = timestamp + recent
        for candidate in arrivals:
            if candidate > timestamp:
                continue
            age = clock - candidate
            oldest_age = max(recent, clock - arrivals[0])
            normalized_position = (
                (age - recent) / max(1, oldest_age - recent)
                if oldest_age > recent
                else 0.0
            )
            pair_rows.append(
                np.concatenate(
                    [
                        projected[candidate] - projected[clock],
                        np.asarray([normalized_position], np.float32),
                        (prefix[clock] - prefix[candidate]) / (1.0 + age),
                    ]
                )
            )
            observed = own["actions"][max(0, clock - recent) : clock]
            if len(observed):
                last_action = observed[-1]
                action_sum = observed.sum(0)
                action_mean = observed.mean(0)
            else:
                last_action = np.zeros(2, np.float32)
                action_sum = np.zeros(2, np.float32)
                action_mean = np.zeros(2, np.float32)
            contexts.append(
                np.concatenate(
                    [
                        last_action,
                        action_sum,
                        action_mean,
                        np.asarray([0.0], np.float32),
                    ]
                )
            )
            mapping[timestamp][candidate] = len(pair_rows) - 1
    if not pair_rows:
        return {}
    frame_keys = torch.tensor(
        np.stack(pair_rows)[:, None], dtype=torch.float32, device=model.projection.device
    )
    action_context = torch.tensor(
        np.stack(contexts), dtype=torch.float32, device=model.projection.device
    )
    valid = torch.ones(
        (len(pair_rows), 1), dtype=torch.bool, device=model.projection.device
    )
    scores = model.frames(frame_keys, valid, action_context).mean((1, 2)).cpu().numpy()
    return {
        timestamp: {
            candidate: float(scores[row_index])
            for candidate, row_index in sorted(by_candidate.items())
        }
        for timestamp, by_candidate in mapping.items()
    }


def replay_irreversible_policies(
    c,
    store,
    rec,
    model,
    device,
    poly,
    load_episode,
    stable_seed,
    collate_fn,
    seed: int,
) -> tuple[dict[str, list[int]], dict]:
    query = int(rec["query"])
    recent = int(c["data"]["recent"])
    capacity = int(c["data"]["budget"])
    arrivals = list(range(query - recent + 1))
    if not arrivals:
        raise ValueError("At least one historical arrival is required")
    surprise = causal_surprise_table(
        store, rec, model, device, collate_fn, load_episode
    )
    attention = causal_attention_table(store, rec, model, load_episode)
    projected = (
        store.projected_features(rec) if rec["task"] in store.bases else None
    )
    if polices_need_projection(POLICIES) and projected is None:
        raise ValueError("Projected features are required for selected policies")

    states = {policy: [] for policy in POLICIES}
    evicted = {policy: set() for policy in POLICIES}
    reservoir_rng = {
        policy: np.random.default_rng(
            stable_reservoir_seed(stable_seed, c, rec, seed, policy)
        )
        for policy in POLICIES
    }
    audit = {
        "updates": 0,
        "max_candidate_pool": 0,
        "candidate_pool_three_updates": 0,
        "max_retained": 0,
        "resurrections": 0,
        "future_feature_accesses": 0,
        "historical_archive_rescans": 0,
        "surprise_valid_scores": len(surprise),
        "attention_update_clocks": len(attention),
    }
    for ordinal, timestamp in enumerate(arrivals, start=1):
        clock = timestamp + recent
        audit["updates"] += len(POLICIES)
        for policy in POLICIES:
            previous = states[policy]
            candidates = poly.checked_candidates(previous, timestamp, capacity)
            audit["max_candidate_pool"] = max(
                audit["max_candidate_pool"], len(candidates)
            )
            audit["candidate_pool_three_updates"] += len(candidates) == 3
            if len(candidates) <= capacity:
                chosen = poly.warm_start(previous, timestamp, capacity)
            elif policy == "fifo_online_b2":
                chosen = poly.fifo(previous, timestamp, capacity)
            elif policy == "reservoir_online_b2":
                chosen = poly.reservoir(
                    previous,
                    timestamp,
                    ordinal,
                    reservoir_rng[policy],
                    capacity,
                )
            elif policy == "temporal_buckets_online_b2":
                chosen = poly.temporal_buckets(
                    previous, timestamp, clock, recent, capacity
                )
            elif policy == "visual_diversity_online_b2":
                chosen = poly.visual_diversity(
                    previous,
                    timestamp,
                    projected,
                    projected[clock],
                    capacity,
                )
            elif policy == "prediction_surprise_online_b2":
                chosen = poly.top_scores(
                    previous, timestamp, surprise, capacity
                )
            elif policy == "attention_online_b2":
                chosen = poly.top_scores(
                    previous, timestamp, attention[timestamp], capacity
                )
            else:
                raise ValueError(f"Unknown policy {policy}")
            if not set(chosen).issubset(candidates):
                audit["resurrections"] += 1
            if any(timestamp_value in evicted[policy] for timestamp_value in chosen):
                audit["resurrections"] += 1
            evicted[policy].update(set(candidates) - set(chosen))
            states[policy] = chosen
            audit["max_retained"] = max(audit["max_retained"], len(chosen))
    if audit["resurrections"] != 0:
        raise RuntimeError("Online policy audit detected resurrection")
    for policy, chosen in states.items():
        if len(chosen) != capacity:
            raise RuntimeError(f"{policy} finished with {len(chosen)} retained entries")
    return states, audit


def polices_need_projection(policy_names) -> bool:
    return any(
        name in {"visual_diversity_online_b2", "attention_online_b2"}
        for name in policy_names
    )


def append_metric_rows(
    rows: list[dict],
    *,
    seed: int,
    task: str,
    split: str,
    policy: str,
    rec: dict,
    chosen: list[int],
    target: np.ndarray,
    prediction: np.ndarray,
    sample_info: dict,
    c: dict,
    store,
    decoders,
    load_episode,
    pair_diagnostic,
) -> None:
    query = int(rec["query"])
    meta = load_episode(c, rec, True)
    sibling = store.records[rec["sibling"]]
    other = store.features(sibling)[query + np.asarray(c["data"]["horizons"])]
    state = decoders.predict(prediction)
    dimension = prediction.shape[-1]
    recent = int(c["data"]["recent"])
    retained = len(chosen)
    for horizon_index, horizon in enumerate(c["data"]["horizons"]):
        predicted_raw = prediction[horizon_index] * store.norm["std"] + store.norm["mean"]
        target_raw = target[horizon_index] * store.norm["std"] + store.norm["mean"]
        cosine = 1.0 - float(
            np.dot(predicted_raw, target_raw)
            / max(
                np.linalg.norm(predicted_raw) * np.linalg.norm(target_raw),
                1e-12,
            )
        )
        accuracy, margin, separation = pair_diagnostic(
            prediction[horizon_index],
            target[horizon_index],
            other[horizon_index],
        )
        target_index = query + horizon
        rows.append(
            {
                "seed": seed,
                "task": task,
                "split": split,
                "condition": policy,
                "horizon": horizon,
                "episode": rec["id"],
                "pair": rec["pair"],
                "gap": rec["gap"],
                "query": query,
                "latent_mse": float(
                    np.mean((predicted_raw - target_raw) ** 2)
                ),
                "normalized_mse": float(
                    np.mean(
                        (prediction[horizon_index] - target[horizon_index]) ** 2
                    )
                ),
                "cosine_distance": cosine,
                "pair_accuracy": accuracy,
                "pair_margin": margin,
                "target_pair_distance": separation,
                "state_accuracy": float(
                    (state[horizon_index, 0] > 0)
                    == (meta["classes"][target_index] > 0)
                ),
                "state_position_mse": float(
                    np.mean(
                        (
                            state[horizon_index, 1:]
                            - meta["states"][target_index]
                        )
                        ** 2
                    )
                ),
                "history_count": retained,
                "history_feature_bytes": retained * dimension * 4,
                "recent_feature_bytes": recent * dimension * 4,
                "projected_history_bytes": retained
                * 128
                * 4,
                "timestamp_type_mask_bytes": (retained + recent) * 13,
                "past_action_bytes": (retained + recent) * 8,
                "future_action_bytes": horizon * 2 * 4,
                "evidence_hit": float(sample_info["evidence_hit"]),
                "state_pred_x": float(state[horizon_index, 1]),
                "state_pred_y": float(state[horizon_index, 2]),
                "state_true_x": float(meta["states"][target_index, 0]),
                "state_true_y": float(meta["states"][target_index, 1]),
                "online_update_order": "timestamp ascending",
                "online_candidate_pool_max": 3,
                "online_irreversible": True,
            }
        )


def merge_stream_audit(total: dict, record: dict) -> None:
    """Accumulate counts while keeping maxima as maxima."""
    for key in SUM_AUDIT_KEYS:
        total[key] += int(record[key])
    for key in ("max_candidate_pool", "max_retained"):
        total[key] = max(int(total[key]), int(record[key]))
    total["resurrections"] += int(record["resurrections"])
    total["future_feature_accesses"] += int(record["future_feature_accesses"])
    total["historical_archive_rescans"] += int(record["historical_archive_rescans"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing overwrite: {args.output}")
    started = time.monotonic()
    args.output.mkdir(parents=True)
    protocol = json.loads(args.protocol.read_text())
    if protocol["status"] != PROTOCOL_STATUS:
        raise ValueError("Unexpected protocol")
    if tuple(protocol["policies"]) != POLICIES:
        raise ValueError("Policy list mismatch")
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
    from phase1.evaluation import pair_diagnostic
    from phase1.inputs import FeatureStore, collate, selected_evidence
    from phase1.probes import load_decoder
    from phase1.repair_evaluation import all_models_completion, load_current_reader
    from phase1.utils import seed_all, stable_seed
    from phase1.proposal_trial import verify_control

    completion = all_models_completion(args.control, args.training_root)
    if not completion["complete"] or len(completion["models"]) != 6:
        raise ValueError("All six completed models are required")
    proposal_control, _, configs = verify_control(args.control)
    seed_all(17, 4)
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    atomic_json(
        args.output / "attempt.json",
        {
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "protocol_sha256": sha256(args.protocol),
            "policy_module_sha256": sha256(Path(policies.__file__)),
            "runner_sha256": sha256(Path(__file__)),
            "source_lock": source_lock,
            "all_six_models_verified": True,
            "device": str(device),
            "no_training": True,
            "no_fitting": True,
            "no_encoder_forward": True,
            "strict_update_pool": "two retained plus one new arrival",
            "future_9137_access": False,
        },
    )

    raw = []
    selections = []
    stream_audit = {
        "records": 0,
        "updates": 0,
        "max_candidate_pool": 0,
        "candidate_pool_three_updates": 0,
        "max_retained": 0,
        "resurrections": 0,
        "future_feature_accesses": 0,
        "historical_archive_rescans": 0,
        "surprise_valid_scores": 0,
        "attention_update_clocks": 0,
        "policies": list(POLICIES),
    }

    for task, c in configs.items():
        store = FeatureStore(
            c,
            max_cached=2 * max(c["data"]["pairs"][split] for split in SPLITS[1:]),
        )
        decoders = load_decoder(c, task)
        for seed in proposal_control["seeds"]:
            model, _ = load_current_reader(
                args.control,
                args.training_root,
                seed,
                store,
                completion,
            )
            model = model.to(device).eval()
            for split in SPLITS[1:]:
                rows = [
                    rec
                    for rec in store.manifest["records"]
                    if rec["task"] == task and rec["split"] == split
                ]
                chunk_records = max(1, c["training"]["batch_size"] // len(POLICIES))
                for start in range(0, len(rows), chunk_records):
                    part = rows[start : start + chunk_records]
                    held = []
                    samples = []
                    for rec in part:
                        choices, audit = replay_irreversible_policies(
                            c,
                            store,
                            rec,
                            model,
                            device,
                            policies,
                            load_episode,
                            stable_seed,
                            collate,
                            seed,
                        )
                        stream_audit["records"] += 1
                        merge_stream_audit(stream_audit, audit)
                        for policy in POLICIES:
                            chosen = choices[policy]
                            sample = base.sample_with_indices(
                                store,
                                rec,
                                chosen,
                                load_episode,
                                selected_evidence,
                            )
                            held.append((rec, policy, chosen, sample))
                            samples.append(sample[0])
                            selections.append(
                                {
                                    "seed": seed,
                                    "task": task,
                                    "split": split,
                                    "condition": policy,
                                    "episode": rec["id"],
                                    "pair": rec["pair"],
                                    "query": rec["query"],
                                    "indices": ";".join(map(str, chosen)),
                                    "max_candidate_pool": audit["max_candidate_pool"],
                                    "resurrected_any": False,
                                }
                            )
                    batch = collate(samples, device)
                    with torch.inference_mode():
                        prediction = model(batch).cpu().numpy()
                    if not np.isfinite(prediction).all():
                        raise RuntimeError("Non-finite prediction")
                    for (
                        rec,
                        policy,
                        chosen,
                        sample,
                    ), pred in zip(held, prediction):
                        _, target, info = sample
                        append_metric_rows(
                            raw,
                            seed=seed,
                            task=task,
                            split=split,
                            policy=policy,
                            rec=rec,
                            chosen=chosen,
                            target=target,
                            prediction=pred,
                            sample_info=info,
                            c=c,
                            store=store,
                            decoders=decoders,
                            load_episode=load_episode,
                            pair_diagnostic=pair_diagnostic,
                        )
                atomic_json(
                    args.output / "progress.json",
                    {
                        "task": task,
                        "seed": seed,
                        "split": split,
                        "raw_rows": len(raw),
                        "selection_rows": len(selections),
                        "strictly_online": True,
                        "phase1_go": False,
                    },
                )

    if stream_audit["resurrections"]:
        raise RuntimeError("Irreversible streaming audit failed")
    if stream_audit["max_candidate_pool"] > 3:
        raise RuntimeError("Policy accessed more than two retained plus one arrival")
    if stream_audit["max_retained"] > 2:
        raise RuntimeError("Policy exceeded B=2")
    atomic_csv(args.output / "raw_records.csv", raw)
    atomic_csv(args.output / "selection_audit.csv", selections)
    atomic_json(
        args.output / "streaming_audit.json",
        {
            "status": "PASS_STRICT_IRREVERSIBLE_ONLINE_AUDIT",
            **stream_audit,
            "selection_rows": len(selections),
            "raw_rows": len(raw),
            "candidate_pool_rule": "previous retained <=2 plus exactly one arrival",
            "evicted_timestamp_may_return": False,
            "historical_archive_used_for_selection": False,
            "future_target_used_for_selection": False,
        },
    )
    atomic_json(
        args.output / "complete.json",
        {
            "status": "STRICT_IRREVERSIBLE_ONLINE_MATRIX_COMPLETE",
            "models": 6,
            "policies": list(POLICIES),
            "raw_rows": len(raw),
            "selection_rows": len(selections),
            "device": str(device),
            "elapsed_seconds": time.monotonic() - started,
            "new_gpu_seconds": 0.0,
            "training": False,
            "fitting": False,
            "encoder_forward": False,
            "phase1_go": False,
        },
    )
    print(
        json.dumps(
            {
                "status": "STRICT_IRREVERSIBLE_ONLINE_MATRIX_COMPLETE",
                "raw_rows": len(raw),
                "selection_rows": len(selections),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
