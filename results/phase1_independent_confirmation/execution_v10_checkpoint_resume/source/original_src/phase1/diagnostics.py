"""Read-only, frozen-checkpoint diagnostics; these outputs are never Phase 1 GO.

This module does not generate data, fit any component, select checkpoints, or
write into the supplied research runs.  Its small standalone CLI is deliberately
separate from the scientific training/evaluation CLI.
"""
import argparse
import copy
import csv
import hashlib
from pathlib import Path

import numpy as np
import torch

from .config import CONDITIONS, SPLITS, scientific_hash, validate
from .data import load_episode
from .evaluation import METRICS, pair_calibration, pair_diagnostic, summarize, training_completion
from .inputs import FeatureStore, collate, selected_evidence
from .probes import load_decoder
from .stats import clustered_interval, paired_improvement
from .training import load_predictor
from .utils import code_hash, file_hash, read_json, write_csv, write_json


POLICIES = ("old_start_end", "wide_zero_end")
ARMS = ("reference", "motion_wide_span")
INVARIANT_CONDITIONS = ("recent_only", "full_history", "random_small_budget")
DIAGNOSTIC_METRICS = ("latent_mse", "normalized_mse", "state_position_mse", "state_accuracy", "pair_accuracy")


def policy_view(store, policy):
    """Share immutable arrays/caches, changing only the existing sampling policy.

    Checkpoint/basis loading must use the ORIGINAL store/config, never this
    inference-only view.  A view neither changes the source config nor creates a
    new scientific run/signature.
    """
    if policy not in POLICIES:
        raise ValueError(f"Unknown frozen inference policy: {policy}")
    view = copy.copy(store)
    view.c = copy.deepcopy(store.c)
    view.c["experiment"] = {**view.c.get("experiment", {}),
        "arm": "reference" if policy == POLICIES[0] else "motion_wide_span"}
    return view


def assert_exact(a, b, context):
    """Fail closed on changed shape, dtype, non-finite values, or array bytes."""
    a, b = np.asarray(a), np.asarray(b)
    if (a.shape != b.shape or a.dtype != b.dtype or
            not np.isfinite(a).all() or not np.isfinite(b).all() or
            a.tobytes(order="C") != b.tobytes(order="C")):
        raise ValueError(f"Frozen policy invariant failed: {context}")


def assert_sample_exact(a, b, context):
    if set(a) != set(b):
        raise ValueError(f"Frozen policy input keys changed: {context}")
    for key in a:
        assert_exact(a[key], b[key], f"{context}/{key}")


def _motion_pairs(records, expected_pairs):
    groups, seen = {}, set()
    for rec in records:
        if rec["task"] != "motion" or rec["id"] in seen:
            raise ValueError("Motion diagnostic requires distinct motion episodes")
        seen.add(rec["id"])
        groups.setdefault(rec["pair"], []).append(rec)
    if len(groups) != expected_pairs:
        raise ValueError("Motion diagnostic slice has an unexpected base-pair count")
    ordered = []
    # Preserve original manifest/minibatch order, including padding lengths;
    # sorting before inference could introduce avoidable numerical differences.
    for siblings in groups.values():
        if (len(siblings) != 2 or siblings[0]["sibling"] != siblings[1]["id"] or
                siblings[1]["sibling"] != siblings[0]["id"] or
                siblings[0]["query"] != siblings[1]["query"] or
                siblings[0]["split"] != siblings[1]["split"]):
            raise ValueError("Motion diagnostic requires reciprocal, matched sibling pairs")
        ordered.extend(siblings)
    return ordered


def _assert_wrong_selected(view, rec, sample, info):
    """Check actual selected images, including the zero frame in wide policy."""
    selected = selected_evidence(view.c, rec,
        load_episode(view.c, rec, True, keys=("evidence",))["evidence"])
    if not np.array_equal(info["indices"], selected):
        raise ValueError("Frozen motion diagnostic requires exactly the two selected frames")
    R = view.c["data"]["recent"]
    sibling = view.records[rec["sibling"]]
    assert_exact(sample["z"][R:], view.features(sibling)[selected], "wrong actual raw frames")
    assert_exact(sample["projected_z"][R:], view.projected_features(sibling)[selected],
                 "wrong actual projected frames")


def evaluate_frozen_motion(c, store, model, decoder, seed, device):
    """Evaluate one fixed theta under both policies on existing development data.

    Model tensors use the unchanged FeatureStore.sample boundary.  Hidden state
    is loaded only afterwards for the existing evaluation readout metrics.
    """
    model.eval().requires_grad_(False)
    views = {policy: policy_view(store, policy) for policy in POLICIES}
    raw, calibration, invariants = [], [], []
    batch_size = c["training"]["batch_size"]
    if batch_size % 2:
        raise ValueError("Frozen motion diagnostic needs complete pairs in each batch")
    for split in SPLITS[1:]:
        records = _motion_pairs([r for r in store.manifest["records"]
            if r["task"] == "motion" and r["split"] == split], c["data"]["pairs"][split])
        for condition in CONDITIONS:
            prediction_hashes = {p: hashlib.sha256() for p in POLICIES}
            for start in range(0, len(records), batch_size):
                part = records[start:start+batch_size]
                samples = {p: [views[p].sample(r, condition) for r in part] for p in POLICIES}
                for i, rec in enumerate(part):
                    old, wide = samples[POLICIES[0]][i], samples[POLICIES[1]][i]
                    assert_exact(old[1], wide[1], f"{rec['id']}/unchanged targets")
                    if condition in INVARIANT_CONDITIONS:
                        assert_sample_exact(old[0], wide[0], f"{rec['id']}/{condition}")
                    if condition == "wrong_evidence":
                        for p in POLICIES:
                            _assert_wrong_selected(views[p], rec, samples[p][i][0], samples[p][i][2])
                    if condition == "recent_only" and i % 2:
                        # Two siblings have identical recent RGB AND all known actions.
                        for p in POLICIES:
                            assert_sample_exact(samples[p][i-1][0], samples[p][i][0],
                                                f"{rec['pair']}/paired recent")
                with torch.inference_mode():
                    predictions = {p: model(collate([s[0] for s in samples[p]], device)).cpu().numpy()
                                   for p in POLICIES}
                for p in POLICIES:
                    if not np.isfinite(predictions[p]).all():
                        raise ValueError("Non-finite frozen motion prediction")
                    prediction_hashes[p].update(predictions[p].tobytes(order="C"))
                if condition in INVARIANT_CONDITIONS:
                    assert_exact(predictions[POLICIES[0]], predictions[POLICIES[1]],
                                 f"seed {seed}/{split}/{condition}/predictions")
                for p in POLICIES:
                    for i, (rec, sample, prediction) in enumerate(zip(part, samples[p], predictions[p])):
                        _, target, info = sample
                        meta = load_episode(c, rec, True, keys=("classes", "states"))
                        sibling = store.records[rec["sibling"]]
                        other = store.features(sibling)[rec["query"]+np.asarray(c["data"]["horizons"])]
                        state = decoder.predict(prediction)
                        D, R, K = prediction.shape[-1], c["data"]["recent"], info["history_count"]
                        for j, h in enumerate(c["data"]["horizons"]):
                            pred_raw = prediction[j]*store.norm["std"]+store.norm["mean"]
                            true_raw = target[j]*store.norm["std"]+store.norm["mean"]
                            pairacc, margin, separation = pair_diagnostic(prediction[j], target[j], other[j])
                            ti = rec["query"]+h
                            raw.append({"policy": p, "seed": seed, "task": "motion", "split": split,
                                "condition": condition, "horizon": h, "episode": rec["id"], "pair": rec["pair"],
                                "gap": rec["gap"], "query": rec["query"],
                                "latent_mse": float(np.mean((pred_raw-true_raw)**2)),
                                "normalized_mse": float(np.mean((prediction[j]-target[j])**2)),
                                "cosine_distance": 1-float(np.dot(pred_raw, true_raw)/max(np.linalg.norm(pred_raw)*np.linalg.norm(true_raw), 1e-12)),
                                "pair_accuracy": pairacc, "pair_margin": margin, "target_pair_distance": separation,
                                "state_accuracy": float((state[j, 0]>0) == (meta["classes"][ti]>0)),
                                "state_position_mse": float(np.mean((state[j, 1:]-meta["states"][ti])**2)),
                                "history_count": K, "history_feature_bytes": K*D*4,
                                "projected_history_bytes": K*model.projection.shape[1]*4,
                                "recent_feature_bytes": R*D*4, "timestamp_type_mask_bytes": (K+R)*13,
                                "past_action_bytes": (K+R)*8, "future_action_bytes": h*2*4,
                                "evidence_hit": float(info["evidence_hit"]),
                                "selected_indices": ",".join(str(int(t)) for t in info["indices"]),
                                "state_pred_x": float(state[j, 1]), "state_pred_y": float(state[j, 2]),
                                "state_true_x": float(meta["states"][ti, 0]), "state_true_y": float(meta["states"][ti, 1])})
                            if i % 2:
                                previous = samples[p][i-1][1]
                                calibration.append({"policy": p, "seed": seed, "task": "motion", "split": split,
                                    "condition": condition, "horizon": h, "pair": rec["pair"],
                                    **pair_calibration(predictions[p][i-1:i+1, j], [previous[j], target[j]])})
            if condition in INVARIANT_CONDITIONS:
                hashes = {p: v.hexdigest() for p, v in prediction_hashes.items()}
                if len(set(hashes.values())) != 1:
                    raise ValueError("Invariant prediction hashes disagree")
                invariants.append({"seed": seed, "split": split, "condition": condition,
                    "episodes": len(records), "byte_identical": True, "prediction_sha256": hashes})
    return raw, calibration, invariants


def _paired_rows_for_policies(rows, seeds):
    """Retain seed-wise episodes; pool seeds before base-pair bootstrapping."""
    if not rows or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Cross-policy comparison requires nonempty distinct seeds and episodes")
    keyed = {}
    for row in rows:
        if row["seed"] not in seeds or row["policy"] not in POLICIES:
            raise ValueError("Unexpected seed/policy in cross-policy comparison")
        key = (row["seed"], row["policy"], row["episode"])
        if key in keyed:
            raise ValueError("Duplicate episode in cross-policy comparison")
        keyed[key] = row
    episodes = sorted({r["episode"] for r in rows})
    for e in episodes:
        expected = [keyed.get((s, p, e)) for s in seeds for p in POLICIES]
        if any(r is None for r in expected) or len({r["pair"] for r in expected}) != 1:
            raise ValueError("Cross-policy comparison requires every matched seed/policy/episode")
    for seed in list(seeds)+["all_seeds_mean"]:
        chosen = seeds if seed == "all_seeds_mean" else [seed]
        old, wide = [], []
        for e in episodes:
            for p, destination in zip(POLICIES, (old, wide)):
                items = [keyed[(s, p, e)] for s in chosen]
                destination.append({**items[0], **{k: float(np.mean([r[k] for r in items]))
                    for k in DIAGNOSTIC_METRICS+("pair_margin",)}})
        # Each episode must have its one other sibling.  Otherwise the bootstrap
        # could silently accept incomplete pairs and give a misleading interval.
        groups = [r["pair"] for r in old]
        if any(groups.count(g) != 2 for g in set(groups)):
            raise ValueError("Cross-policy statistics require exactly two episodes per base pair")
        yield seed, old, wide


def cross_policy_comparisons(c, raw):
    output = []
    for split in SPLITS[1:]:
        for h in c["data"]["horizons"]:
            for condition in CONDITIONS:
                subset = [r for r in raw if r["split"] == split and r["horizon"] == h and r["condition"] == condition]
                if len({r["pair"] for r in subset}) != c["data"]["pairs"][split]:
                    raise ValueError("Cross-policy slice has missing or extra base pairs")
                for seed, old, wide in _paired_rows_for_policies(subset, c["training"]["seeds"]):
                    groups = [r["pair"] for r in old]
                    results = {}
                    for metric in DIAGNOSTIC_METRICS:
                        lower = metric.endswith("mse")
                        a = [r[metric] if lower else 1-r[metric] for r in old]
                        b = [r[metric] if lower else 1-r[metric] for r in wide]
                        floor = c["decision"]["position_baseline_floor"] if metric == "state_position_mse" else 1e-8 if metric == "latent_mse" else c["decision"]["baseline_floor"]
                        gain = c["decision"]["absolute_position_mse_gain"] if metric == "state_position_mse" else c["decision"]["absolute_latent_gain"] if lower else c["decision"]["absolute_state_gain"]
                        results[metric] = {"old_mean": float(np.mean([r[metric] for r in old])),
                            "wide_mean": float(np.mean([r[metric] for r in wide])),
                            "improvement": paired_improvement(a, b, groups, c["evaluation"]["bootstrap"], floor, gain),
                            "absolute_error_reduction": clustered_interval(np.asarray(a)-b, groups, c["evaluation"]["bootstrap"])}
                    output.append({"seed": seed, "task": "motion", "split": split, "horizon": h,
                        "condition": condition, "reference_policy": POLICIES[0], "alternative_policy": POLICIES[1],
                        "metrics": results,
                        "pair_margin_shift": clustered_interval([b["pair_margin"]-a["pair_margin"] for a, b in zip(old, wide)],
                                                                 groups, c["evaluation"]["bootstrap"])})
    return output


def _source_files(root):
    fixed = ["resolved_config.json", "dataset/manifest.json", "dataset/split_audit.json", "features/index.json",
        "features/pair_feature_audit.json", "features/normalization.npz", "features/normalization.json",
        "predictor/motion/basis.npz", "predictor/motion/basis.json", "probes/metrics.json",
        "probes/motion_state_decoder.npz", "evaluation/metrics.json", "evaluation/raw_records.csv"]
    fixed += [f"predictor/motion/seed_{seed}/{name}" for seed in (17, 29, 43)
              for name in ("best.pt", "latest.pt", "training.json", "training.csv")]
    return {name: root/name for name in fixed}


def _hash_sources(roots):
    return {arm: {name: file_hash(path) for name, path in _source_files(root).items()}
            for arm, root in roots.items()}


def _guard_output(output, roots):
    output = Path(output).expanduser().resolve()
    for root in roots.values():
        protected = [root]+[(root/name).resolve() for name in ("dataset", "features", "predictor", "probes", "evaluation")]
        if any(output == p or output in p.parents or p in output.parents for p in protected):
            raise ValueError("Diagnostic output must be separate from original runs and shared artifacts")
    for suffix in ("json", "md", "raw.csv", "calibration.csv"):
        if (output/f"motion_cross_policy.{suffix}").exists():
            raise ValueError("Diagnostic output already exists; choose a new directory")
    return output


def _source_config(root, arm):
    c = read_json(root/"resolved_config.json")
    c["output"] = str(root)  # Relocation is explicitly outside scientific_hash.
    validate(c)
    if c.get("experiment") != {"stage": "development", "arm": arm, "tasks": ["motion"]}:
        raise ValueError("Expected an original v3 motion development arm")
    if (c["mode"] == "smoke" or c["data"]["seed"] != 6301 or c["data"]["budget"] != 2 or c["training"]["seeds"] != [17, 29, 43]
            or c["training"]["epochs"] != 120):
        raise ValueError("Diagnostic requires real completed three-seed v3 B=2 runs")
    return c


def _verify_source_proof(c, store):
    """No missing basis can silently trigger fit_basis through load_predictor."""
    root = Path(c["output"])
    if not (root/"predictor/motion/basis.npz").is_file() or not (root/"predictor/motion/basis.json").is_file():
        raise ValueError("Existing immutable task basis is required; fitting is forbidden")
    completion = training_completion(c, store)
    if not completion["complete"]:
        raise ValueError("Original training is incomplete/invalid: "+"; ".join(completion["issues"]))
    proof = read_json(root/"evaluation/metrics.json")
    if (proof["cache_signature"] != store.ix["signature"] or
            proof["raw_sha256"] != file_hash(root/"evaluation/raw_records.csv") or
            proof["evaluation_code_hash"] != code_hash("evaluation.py", "stats.py")):
        raise ValueError("Original evaluation proof/raw/code mismatch")
    for key, checked in completion["models"].items():
        previous = proof["training_completion"]["models"][key]
        for field in ("best_sha256", "latest_sha256", "training_json_sha256", "signature", "best_epoch"):
            if previous[field] != checked[field]:
                raise ValueError("Original checkpoint differs from retained evaluation proof")
        if proof["checkpoints"][key]["best_sha256"] != checked["best_sha256"]:
            raise ValueError("Original selected checkpoint proof mismatch")
    return completion


def verify_native_metrics(raw, source_csv, arm, atol=1e-7, rtol=1e-6):
    """Reproduce every original native-policy metric, not merely its file hash.

    Original evaluation additionally contains persistence; this frozen matrix
    compares its five evaluated conditions only.  Auxiliary coordinates are
    intentionally outside METRICS (the original state_true_y column is faulty).
    """
    if arm not in ARMS:
        raise ValueError("Unknown original checkpoint arm")
    native_policy = POLICIES[ARMS.index(arm)]
    fields = ("seed", "task", "split", "horizon", "condition", "pair", "episode")

    def keyed(rows):
        result = {}
        for row in rows:
            if row["condition"] not in CONDITIONS:
                continue
            key = tuple(int(row[k]) if k in {"seed", "horizon"} else row[k] for k in fields)
            if key in result:
                raise ValueError("Duplicate native-policy raw-record key")
            result[key] = row
        return result

    with open(source_csv, newline="", encoding="utf-8") as f:
        expected = keyed(csv.DictReader(f))
    actual = keyed(r for r in raw if r["policy"] == native_policy)
    if not expected or set(actual) != set(expected):
        raise ValueError("Native-policy raw-record coverage differs from original evaluation")
    differences = dict.fromkeys(METRICS, 0.0)
    for key, original in expected.items():
        for metric in METRICS:
            a, b = float(actual[key][metric]), float(original[metric])
            if not np.isfinite([a, b]).all() or not np.isclose(a, b, atol=atol, rtol=rtol):
                raise ValueError(f"Native-policy metric does not reproduce original: {key}/{metric}: {a} vs {b}")
            differences[metric] = max(differences[metric], abs(a-b))
    return {"reproduced": True, "native_policy": native_policy, "matched_rows": len(actual),
        "key_fields": list(fields), "compared_metrics": list(METRICS), "atol": atol, "rtol": rtol,
        "maximum_absolute_difference": max(differences.values()), "maximum_absolute_difference_by_metric": differences,
        "original_raw_sha256": file_hash(source_csv),
        "excluded": "persistence (not evaluated here) and auxiliary columns outside the original METRICS"}


def _markdown(result):
    lines = ["# Frozen motion checkpoint × frame-policy diagnostic", "",
        "This is a development diagnostic, not a Phase 1 GO result. No component was fitted or selected.", "",
        "Each row uses one unchanged checkpoint. Old = original start/end; wide = frame 0 / the same end.",
        "All recent/full/random input tensors and prediction bytes match across the two policies.",
        "Wrong evidence swaps the actual selected frames from the matched sibling.", "",
        "| Checkpoint arm | Seed | Split | Horizon | Old privileged MSE | Wide privileged MSE | Absolute reduction (95% pair CI) |",
        "|---|---|---|---:|---:|---:|---|"]
    for arm, value in result["arms"].items():
        for row in value["cross_policy_comparisons"]:
            if row["condition"] != "privileged_small_budget":
                continue
            m = row["metrics"]["normalized_mse"]
            ci = m["absolute_error_reduction"]
            lines.append(f"| {arm} | {row['seed']} | {row['split']} | {row['horizon']} | {m['old_mean']:.6f} | {m['wide_mean']:.6f} | {ci['mean']:.6f} [{ci['low']:.6f}, {ci['high']:.6f}] |")
    lines += ["", "Positive reduction means wide is better under that fixed checkpoint.",
        "Pooled intervals average the measured seeds per episode before bootstrapping base sibling pairs; they do not estimate uncertainty over new training seeds.",
        "All per-seed conditions, horizons, slices, wrong-evidence shifts, original-scale errors and pair decompositions are retained in JSON/CSV.",
        "Cross-policy inputs may be outside a checkpoint's training input distribution. These comparisons isolate inference policy at fixed weights; they do not replace retrained-arm or fresh-confirmation tests.",
        "No thresholds or checkpoints were changed and no new confirmation data were opened."]
    return "\n".join(lines)+"\n"


def motion_cross_policy(reference_output, wide_output, output, device="cpu"):
    """Run the fixed 2-checkpoint-arm × 2-policy matrix, using existing cache only.

    Invoke on allocated course compute containing the original complete runs.
    Inputs are trusted own research outputs, never externally supplied weights.
    Existing artifacts are hashed before/after; all new files use `output`.
    """
    roots = {arm: Path(p).expanduser().resolve() for arm, p in zip(ARMS, (reference_output, wide_output))}
    if len(set(roots.values())) != 2:
        raise ValueError("Two distinct original checkpoint arms are required")
    output = _guard_output(output, roots)
    before = _hash_sources(roots)  # Missing cached artifacts fail before any model loading.
    configs = {arm: _source_config(root, arm) for arm, root in roots.items()}
    base_configs = [dict(c) for c in configs.values()]
    for c in base_configs:
        c.pop("experiment")
    if scientific_hash(base_configs[0]) != scientific_hash(base_configs[1]):
        raise ValueError("Checkpoint arms differ beyond their predeclared experiment arm")
    for name in ("dataset/manifest.json", "features/index.json", "features/normalization.npz",
                 "predictor/motion/basis.npz", "probes/motion_state_decoder.npz"):
        if before[ARMS[0]][name] != before[ARMS[1]][name]:
            raise ValueError(f"Source arms do not share the fixed artifact: {name}")
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Use allocated CPU or CUDA compute")
    old_threads = torch.get_num_threads()
    torch.set_num_threads(configs[ARMS[0]]["threads"])
    all_raw, all_calibration, arms = [], [], {}
    try:
        for arm, c in configs.items():
            store = FeatureStore(c, max_cached=2*max(c["data"]["pairs"][s] for s in SPLITS[1:]))
            completion = _verify_source_proof(c, store)
            decoder = load_decoder(c, "motion")
            raw, calibration, invariants = [], [], []
            for seed in c["training"]["seeds"]:
                model, ck = load_predictor(c, seed, store, device, "motion")
                if ck["epoch"] != completion["models"][f"motion/{seed}"]["best_epoch"]:
                    raise ValueError("Loaded checkpoint is not the retained validation-selected best")
                records, pairs, checks = evaluate_frozen_motion(c, store, model, decoder, seed, device)
                raw.extend(records)
                calibration.extend(pairs)
                invariants.extend(checks)
                del model, ck
            reproduction = verify_native_metrics(raw, roots[arm]/"evaluation/raw_records.csv", arm)
            arms[arm] = {"source": str(roots[arm]), "training_completion": completion,
                "native_policy_reproduction": reproduction,
                "policy_summaries": {p: summarize(c, [r for r in raw if r["policy"] == p]) for p in POLICIES},
                "cross_policy_comparisons": cross_policy_comparisons(c, raw), "invariants": invariants}
            all_raw.extend({"checkpoint_arm": arm, **r} for r in raw)
            all_calibration.extend({"checkpoint_arm": arm, **r} for r in calibration)
    finally:
        torch.set_num_threads(old_threads)
        after = _hash_sources(roots)
        if before != after:
            raise ValueError("Original research artifacts changed during diagnostic; result invalid")
    result = {"status": "DIAGNOSTIC_ONLY_NOT_GO", "frozen_weights": True, "new_training": False,
        "scope": "existing v3 motion development validation and old test slices; no fresh confirmation",
        "policies": list(POLICIES), "arms": arms, "source_sha256": before, "source_unchanged": True,
        "diagnostic_code_hash": code_hash("diagnostics.py", "evaluation.py", "stats.py", "inputs.py"),
        "statistics": "Original metrics and 95% paired base-pair bootstrap; mean measured seeds before pooled bootstrap; every seed retained",
        "interpretation": "Within each checkpoint arm/seed, only inference-time frame policy changes. Invariant recent predictions prevent denominator drift in those paired comparisons. Between-arm weights differ and are not the frozen-policy causal comparison."}
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/"motion_cross_policy.raw.csv", all_raw)
    write_csv(output/"motion_cross_policy.calibration.csv", all_calibration)
    result["raw_sha256"] = file_hash(output/"motion_cross_policy.raw.csv")
    result["calibration_sha256"] = file_hash(output/"motion_cross_policy.calibration.csv")
    write_json(output/"motion_cross_policy.json", result)
    (output/"motion_cross_policy.md").write_text(_markdown(result), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="diagnostic", required=True)
    motion = sub.add_parser("motion-cross-policy", help="frozen motion checkpoint × frame-policy matrix")
    motion.add_argument("--reference-output", required=True)
    motion.add_argument("--wide-output", required=True)
    motion.add_argument("--output", required=True)
    motion.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args(argv)
    result = motion_cross_policy(args.reference_output, args.wide_output, args.output, args.device)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
