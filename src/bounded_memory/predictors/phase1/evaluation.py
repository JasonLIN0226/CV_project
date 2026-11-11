import ast
import time
from pathlib import Path

import numpy as np
import torch

from .config import CONDITIONS, SPLITS
from .data import load_episode
from .inputs import FeatureStore, collate
from .probes import load_decoder
from .stats import clustered_interval, paired_improvement
from .training import load_predictor, checkpoint_signature, trusted_checkpoint
from .utils import write_csv, write_json, read_json, device_of, file_hash, code_hash

METRICS = ["latent_mse", "normalized_mse", "cosine_distance", "pair_accuracy", "pair_margin",
           "target_pair_distance", "state_accuracy", "state_position_mse", "history_count",
           "history_feature_bytes", "projected_history_bytes", "recent_feature_bytes", "timestamp_type_mask_bytes", "past_action_bytes", "evidence_hit"]


def active_tasks(c):
    """A development arm selects tasks without altering shared data/cache provenance."""
    return c.get("experiment", {}).get("tasks", c["data"]["tasks"])


def _finite_tree(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all()) if np.issubdtype(value.dtype, np.number) else True
    if isinstance(value, dict):
        return all(_finite_tree(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite_tree(v) for v in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


def training_completion(c, store=None):
    """Check actual own latest/best checkpoints before reading any held-out results."""
    store = store or FeatureStore(c)
    v3 = bool(c.get("experiment"))
    required_epochs = c["training"]["epochs"]
    issues, models = [], {}
    if v3 and c["mode"] != "smoke" and (required_epochs != 120 or c["training"]["seeds"] != [17, 29, 43]):
        issues.append("v3 research requires exactly 120 epochs and the fixed seeds 17, 29, 43")
    steps_per_epoch = 4*((c["data"]["pairs"]["train"]+c["training"]["batch_size"]//2-1)//(c["training"]["batch_size"]//2))
    sample_count = 2*c["data"]["pairs"]["train"]
    conditions = CONDITIONS[:4]
    for task in active_tasks(c):
        for seed in c["training"]["seeds"]:
            key = f"{task}/{seed}"
            root = Path(c["output"])/"predictor"/task/f"seed_{seed}"
            error = []
            try:
                latest = trusted_checkpoint(root/"latest.pt", "cpu")
                best = trusted_checkpoint(root/"best.pt", "cpu")
                logs = latest["logs"]
                signature = checkpoint_signature(c, store, task)
                for name, ck in [("latest", latest), ("best", best)]:
                    if ck["signature"] != signature or ck["seed"] != seed or ck["task"] != task:
                        error.append(f"{name} checkpoint provenance mismatch")
                    if not _finite_tree(ck["model"]) or not _finite_tree(ck["optimizer"]):
                        error.append(f"{name} has non-finite model/optimizer values")
                    if not {"python", "numpy", "torch", "cuda"}.issubset(ck.get("rng", {})):
                        error.append(f"{name} RNG state is incomplete")
                    if ck["step"] != ck["epoch"]*steps_per_epoch or len(ck["logs"]) != ck["epoch"]:
                        error.append(f"{name} epoch/step/log length mismatch")
                if latest["epoch"] != required_epochs or len(logs) != required_epochs:
                    error.append(f"completed {latest['epoch']} of {required_epochs} required epochs")
                for i, row in enumerate(logs, 1):
                    counts = ast.literal_eval(row["mix_counts"]) if isinstance(row["mix_counts"], str) else row["mix_counts"]
                    if row["epoch"] != i or row["step"] != i*steps_per_epoch or counts != {k: sample_count for k in conditions}:
                        error.append(f"epoch {i} has incorrect sample counts/order/steps")
                    if not _finite_tree(row):
                        error.append(f"epoch {i} contains non-finite logged values")
                selection = latest.get("checkpoint_selection", latest.get("signature", {}).get("checkpoint_selection", "validation_loss"))
                if v3 and selection != "validation_mse":
                    error.append("v3 checkpoint selection must be validation_mse")
                metric = "validation_mse" if selection == "validation_mse" else "validation_loss"
                expected_best = min(logs, key=lambda r: r[metric])["epoch"] if logs else None
                if best["epoch"] != expected_best or best.get("checkpoint_selection", selection) != selection:
                    error.append("best checkpoint is not the earliest minimum of the prespecified validation criterion")
                if best["logs"] != logs[:best["epoch"]]:
                    error.append("best checkpoint log is not the matching prefix of completed training")
                actual_log = read_json(root/"training.json")
                if actual_log["logs"] != logs or actual_log["signature"] != signature:
                    error.append("actual training JSON differs from the latest checkpoint log/provenance")
                models[key] = {"complete": not error, "task": task, "seed": seed,
                    "completed_epochs": latest["epoch"], "final_step": latest["step"],
                    "selection_metric": selection, "best_epoch": best["epoch"],
                    "latest_sha256": file_hash(root/"latest.pt"), "best_sha256": file_hash(root/"best.pt"),
                    "training_json_sha256": file_hash(root/"training.json"), "signature": signature,
                    "expected_samples_per_condition_epoch": sample_count,
                    "expected_steps_per_epoch": steps_per_epoch, "issues": error}
            except (OSError, ValueError, KeyError, TypeError, SyntaxError) as exc:
                error.append(f"missing or invalid checkpoint/log: {exc}")
                models[key] = {"complete": False, "task": task, "seed": seed, "issues": error}
            issues.extend(f"{key}: {e}" for e in error)
    return {"complete": not issues, "required_epochs": required_epochs,
            "tasks": active_tasks(c), "seeds": c["training"]["seeds"], "models": models, "issues": issues}


def pair_calibration(prediction, target):
    """Normalized visual pair decomposition; hidden simulator state is not an input."""
    prediction, target = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[0] != 2:
        raise ValueError("Calibration requires a complete pair of equal-dimensional vectors")
    common_pred, common_true = prediction.mean(0), target.mean(0)
    contrast_pred, contrast_true = (prediction[0]-prediction[1])/2, (target[0]-target[1])/2
    denominator = float(np.dot(contrast_true, contrast_true))
    alpha = float(np.dot(contrast_pred, contrast_true)/denominator) if denominator > 1e-12 else None
    common_mse = float(np.mean((common_pred-common_true)**2))
    contrast_mse = float(np.mean((contrast_pred-contrast_true)**2))
    return {"common_mse": common_mse, "contrast_mse": contrast_mse,
            "total_mse": float(np.mean((prediction-target)**2)),
            "decomposition_residual": float(np.mean((prediction-target)**2)-common_mse-contrast_mse),
            "alpha": alpha, "target_contrast_mse": float(np.mean(contrast_true**2)),
            "orthogonal_contrast_mse": float(np.mean((contrast_pred-alpha*contrast_true)**2)) if alpha is not None else None,
            "degenerate_target_contrast": alpha is None}


def summarize_calibration(c, rows):
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in ["seed", "task", "split", "condition", "horizon"])
        groups.setdefault(key, []).append(row)
    output = []
    for key, values in groups.items():
        item = dict(zip(["seed", "task", "split", "condition", "horizon"], key))
        item["metrics"] = {}
        for metric in ["common_mse", "contrast_mse", "total_mse", "decomposition_residual", "alpha", "target_contrast_mse", "orthogonal_contrast_mse"]:
            valid = [r for r in values if r[metric] is not None]
            item["metrics"][metric] = clustered_interval([r[metric] for r in valid], [r["pair"] for r in valid], c["evaluation"]["bootstrap"]) if valid else None
        item["degenerate_target_pairs"] = sum(r["degenerate_target_contrast"] for r in values)
        output.append(item)
    return {"per_seed": output, "scope": "evaluation-only normalized visual pair common/contrast decomposition",
            "interpretation": "Alpha greater than one can indicate contrast amplification; this diagnostic does not establish the cause of an error or select a checkpoint."}


def pair_diagnostic(pred, own, other, atol=1e-7):
    d_own, d_other = np.mean((pred-own)**2), np.mean((pred-other)**2)
    scale = max(np.mean((own-other)**2), 1e-12)
    tolerance = atol * max(1, abs(d_own), abs(d_other))
    acc = .5 if abs(d_other-d_own) <= tolerance else float(d_own < d_other)
    return acc, float((d_other-d_own)/scale), float(scale)


def timed_forward(c, model, batch, device):
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    with torch.inference_mode():
        for _ in range(c["evaluation"]["latency_warmup"]):
            model(batch)
        sync()
        baseline = torch.cuda.memory_allocated(device) if device.type == "cuda" else None
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        began = time.perf_counter()
        for _ in range(c["evaluation"]["latency_repeats"]):
            model(batch)
        sync()
        ms = 1000*(time.perf_counter()-began)/c["evaluation"]["latency_repeats"]
    return {"milliseconds": ms, "device": str(device), "batch_size": len(batch["z"]),
            "context_length": int(batch["mask"].sum(1).max()), "dtype": str(batch["z"].dtype),
            "warmup": c["evaluation"]["latency_warmup"], "repeats": c["evaluation"]["latency_repeats"],
            "scope": "cached-feature all-horizon predictor forward ONLY",
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "baseline_cuda_allocated_bytes": baseline,
            "predictor_parameter_bytes": sum(p.numel()*p.element_size() for p in model.parameters()),
            "predictor_buffer_bytes": sum(p.numel()*p.element_size() for p in model.buffers())}


def evaluate(c):
    device = device_of(c)
    store = FeatureStore(c, max_cached=2*max(c["data"]["pairs"][s] for s in SPLITS[1:]))
    completion = training_completion(c, store) if c.get("experiment") else None
    if completion is not None and not completion["complete"]:
        raise ValueError("Held-out evaluation requires all prescribed training to complete: "+"; ".join(completion["issues"]))
    decoders = {task: load_decoder(c, task) for task in active_tasks(c)}
    raw, latencies, checkpoints, calibration, pending = [], [], {}, [], {}
    for seed in c["training"]["seeds"]:
        for task in active_tasks(c):
            model, ck = load_predictor(c, seed, store, device, task)
            checkpoints[f"{task}/{seed}"] = {"task": task, "seed": seed, "best_epoch": ck["epoch"], "training_seconds": ck["seconds"],
                "best_sha256": file_hash(Path(c["output"])/"predictor"/task/f"seed_{seed}"/"best.pt")}
            for split in SPLITS[1:]:
                rows = [r for r in store.manifest["records"] if r["task"] == task and r["split"] == split]
                for condition in CONDITIONS+["persistence"]:
                    if condition != "persistence":
                        sample = store.sample(rows[0], condition)[0]
                        latency = timed_forward(c, model, collate([sample], device), device)
                        latencies.append({"seed": seed, "task": task, "split": split, "condition": condition, **latency})
                    for start in range(0, len(rows), c["training"]["batch_size"]):
                        part = rows[start:start+c["training"]["batch_size"]]
                        samples = [store.sample(r, condition if condition != "persistence" else "recent_only") for r in part]
                        if condition == "persistence":
                            predictions = np.stack([np.tile(s[0]["current"], (len(c["data"]["horizons"]), 1)) for s in samples])
                        else:
                            with torch.inference_mode():
                                predictions = model(collate([s[0] for s in samples], device)).cpu().numpy()
                        for rec, sample, prediction in zip(part, samples, predictions):
                            _, target, info = sample
                            meta = load_episode(c, rec, True)
                            sibling = store.records[rec["sibling"]]
                            other = store.features(sibling)[rec["query"]+np.array(c["data"]["horizons"])]
                            state = decoders[task].predict(prediction)
                            D, R, K = prediction.shape[-1], c["data"]["recent"], info["history_count"]
                            for j, h in enumerate(c["data"]["horizons"]):
                                if task == "gate":
                                    key = (seed, task, split, condition, h, rec["pair"])
                                    pending.setdefault(key, {})[rec["id"]] = (prediction[j], target[j])
                                    if len(pending[key]) == 2:
                                        vectors = [pending[key][eid] for eid in sorted(pending[key])]
                                        calibration.append({"seed": seed, "task": task, "split": split, "condition": condition, "horizon": h,
                                                            "pair": rec["pair"], **pair_calibration([p for p, _ in vectors], [y for _, y in vectors])})
                                        del pending[key]
                                p, y = prediction[j]*store.norm["std"]+store.norm["mean"], target[j]*store.norm["std"]+store.norm["mean"]
                                cosine = 1-float(np.dot(p, y)/(max(np.linalg.norm(p)*np.linalg.norm(y), 1e-12)))
                                pairacc, margin, separation = pair_diagnostic(prediction[j], target[j], other[j])
                                ti = rec["query"]+h
                                raw.append({"seed": seed, "task": task, "split": split, "condition": condition, "horizon": h,
                                    "episode": rec["id"], "pair": rec["pair"], "gap": rec["gap"], "query": rec["query"],
                                    "latent_mse": float(np.mean((p-y)**2)), "normalized_mse": float(np.mean((prediction[j]-target[j])**2)),
                                    "cosine_distance": cosine, "pair_accuracy": pairacc, "pair_margin": margin,
                                    "target_pair_distance": separation, "state_accuracy": float((state[j, 0]>0) == (meta["classes"][ti]>0)),
                                    "state_position_mse": float(np.mean((state[j, 1:]-meta["states"][ti])**2)),
                                    "history_count": K, "history_feature_bytes": K*D*4, "recent_feature_bytes": R*D*4,
                                    "projected_history_bytes": K*model.projection.shape[1]*4,
                                    "timestamp_type_mask_bytes": (K+R)*13, "past_action_bytes": (K+R)*8,
                                    "future_action_bytes": h*2*4, "evidence_hit": float(info["evidence_hit"]),
                                    "state_pred_x": float(state[j, 1]), "state_pred_y": float(state[j, 2]),
                                    "state_true_x": float(meta["states"][ti, 0]), "state_true_y": float(meta["states"][ti, 1])})
    root = Path(c["output"])/"evaluation"
    if pending:
        raise ValueError("Calibration encountered an incomplete gate sibling pair")
    write_csv(root/"raw_records.csv", raw)
    summary = summarize(c, raw)
    write_json(root/"metrics.json", {"mode": c["mode"], "cache_signature": store.ix["signature"],
        "raw_sha256": file_hash(root/"raw_records.csv"), "evaluation_code_hash": code_hash("evaluation.py", "stats.py"),
        "checkpoints": checkpoints, **({"training_completion": completion, "experiment": c["experiment"]} if completion is not None else {}), **summary})
    write_json(root/"latency_memory.json", latencies)
    if calibration:
        write_json(root/"calibration_summary.json", {"cache_signature": store.ix["signature"], **summarize_calibration(c, calibration)})
    return summary


def summarize(c, raw):
    groups = {}
    for r in raw:
        key = tuple(r[k] for k in ["seed", "task", "split", "condition", "horizon"])
        groups.setdefault(key, []).append(r)
    summary = []
    for key, rows in groups.items():
        row = dict(zip(["seed", "task", "split", "condition", "horizon"], key))
        row["metrics"] = {metric: clustered_interval([r[metric] for r in rows], [r["pair"] for r in rows], c["evaluation"]["bootstrap"])
                          for metric in METRICS}
        row["target_pair_distance_quantiles"] = np.quantile([r["target_pair_distance"] for r in rows], [.0, .1, .5, .9, 1.]).tolist()
        row["almost_indistinguishable_fraction"] = float(np.mean([r["target_pair_distance"] < c["decision"]["pair_distance_floor"] for r in rows]))
        summary.append(row)
    # Across seeds, average each (pair, sibling) first, then bootstrap base pairs.
    # This is conditional on the measured seeds; ALSO retain every seed and require each seed for GO.
    pooled = {}
    for r in raw:
        key = tuple(r[k] for k in ["task", "split", "condition", "horizon", "episode"])
        pooled.setdefault(key, []).append(r)
    averaged = [{**rows[0], **{metric: float(np.mean([r[metric] for r in rows])) for metric in METRICS}} for rows in pooled.values()]
    comparisons = []
    for task in active_tasks(c):
        for split in SPLITS[1:]:
            for h in c["data"]["horizons"]:
                for seed in c["training"]["seeds"]+["all_seeds_mean"]:
                    data = averaged if seed == "all_seeds_mean" else [r for r in raw if r["seed"] == seed]
                    subset = [r for r in data if r["task"] == task and r["split"] == split and r["horizon"] == h]
                    by = {(r["condition"], r["episode"]): r for r in subset}
                    episodes = sorted({r["episode"] for r in subset})
                    for alt in ["privileged_small_budget", "full_history", "random_small_budget"]:
                        base, better = [by[("recent_only", e)] for e in episodes], [by[(alt, e)] for e in episodes]
                        results = {}
                        for metric in ["normalized_mse", "latent_mse", "state_position_mse", "state_accuracy", "pair_accuracy"]:
                            lower_better = metric.endswith("mse")
                            a = [r[metric] if lower_better else 1-r[metric] for r in base]
                            b = [r[metric] if lower_better else 1-r[metric] for r in better]
                            floor = c["decision"]["position_baseline_floor"] if metric == "state_position_mse" else 1e-8 if metric == "latent_mse" else c["decision"]["baseline_floor"]
                            absolute_gain = c["decision"]["absolute_position_mse_gain"] if metric == "state_position_mse" else c["decision"]["absolute_latent_gain"] if metric.endswith("mse") else c["decision"]["absolute_state_gain"]
                            results[metric] = paired_improvement(a, b, [r["pair"] for r in base], c["evaluation"]["bootstrap"],
                                floor, absolute_gain)
                        comparisons.append({"seed": seed, "task": task, "split": split, "horizon": h, "alternative": alt,
                                            "reference": "recent_only", "improvement": results})
                    priv = [by[("privileged_small_budget", e)] for e in episodes]
                    wrong = [by[("wrong_evidence", e)] for e in episodes]
                    shift = [p["pair_margin"]-w["pair_margin"] for p, w in zip(priv, wrong)]
                    comparisons.append({"seed": seed, "task": task, "split": split, "horizon": h,
                        "alternative": "wrong_evidence", "reference": "privileged_small_budget",
                        "toward_wrong_future": clustered_interval(shift, [r["pair"] for r in priv], c["evaluation"]["bootstrap"])})
    return {"per_seed": summary, "comparisons": comparisons,
            "statistics": "95% percentile paired bootstrap over base pairs; all_seeds_mean CI conditional on seeds; per-seed results retained"}
