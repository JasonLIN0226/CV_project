import copy
from pathlib import Path

import yaml

from .utils import digest, write_json, read_json

SPLITS = ["train", "validation", "test_id", "test_long_gap", "test_distractor_shift"]
CONDITIONS = ["recent_only", "full_history", "privileged_small_budget", "random_small_budget", "wrong_evidence"]
DEFAULT = {
    "mode": "smoke", "output": "runs/smoke", "device": "cpu", "threads": 2,
    "data": {"seed": 2271, "tasks": ["gate", "motion"], "image_size": 96, "recent": 4,
             "horizons": [3, 6], "budget": 2, "gap": [5, 10], "long_gap": [20, 32],
             "cue_start": [2, 7], "cue_duration": [1, 3], "motion_dt": [1, 3],
             "pairs": {"train": 12, "validation": 4, "test_id": 6, "test_long_gap": 6, "test_distractor_shift": 6}},
    "encoder": {"backend": "smoke_pixels", "repository": "facebook/ijepa_vith14_1k", "revision": "main",
                "pooling": "grid2", "dtype": "float32", "batch_size": 2, "smoke_size": 12,
                "local_files_only": False, "cache_dir": None},
    "normalization": {"std_floor": 0.05},
    "probe": {"ridge": [0.01, 0.1, 1.0, 10.0]},
    "model": {"hidden": 64, "heads": 4, "layers": 2, "dropout": 0.05, "rank": 32},
    "training": {"seeds": [17], "epochs": 4, "batch_size": 16, "lr": 0.001, "weight_decay": 0.0001,
                 "clip": 1.0, "lr_step": 40, "lr_gamma": 0.5, "pair_weight": 0.5, "rank_weight": 0.0},
    "evaluation": {"random_seed": 901, "bootstrap": 500, "latency_warmup": 3, "latency_repeats": 10},
    "decision": {"min_seeds": 3, "min_pairs": 32, "probe_accuracy": 0.75, "readout_accuracy": 0.8, "readout_position_rmse": 0.05,
                 "latent_relative_improvement": 0.1, "state_relative_improvement": 0.1,
                 "absolute_state_gain": 0.05, "absolute_latent_gain": 0.01, "baseline_floor": 0.02,
                 "position_baseline_floor": 0.00001, "absolute_position_mse_gain": 0.000001, "wrong_evidence_gain": 0.05,
                 "full_history_tolerance": 0.05, "pair_distance_floor": 0.000001}
}


def merge(base, update, prefix=""):
    result = copy.deepcopy(base)
    for k, v in update.items():
        if k not in base:
            raise ValueError(f"Unknown config key {prefix}{k}")
        result[k] = merge(base[k], v, prefix + k + ".") if isinstance(v, dict) else v
    return result


def validate(c):
    d = c["data"]
    if c["mode"] not in {"smoke", "pilot", "confirm"}:
        raise ValueError("mode must be smoke, pilot, or confirm")
    if (c["encoder"]["backend"].startswith("smoke")) != (c["mode"] == "smoke"):
        raise ValueError("Smoke backend must use smoke mode; research runs require ijepa.")
    if c["encoder"]["backend"] not in {"smoke_pixels", "ijepa"}:
        raise ValueError("Unsupported encoder")
    if c["encoder"]["pooling"] not in {"mean", "grid2"}:
        raise ValueError("Unsupported pooling")
    if d["tasks"] != ["gate", "motion"] or set(d["pairs"]) != set(SPLITS):
        raise ValueError("Both tasks and all five splits are required")
    if d["recent"] < 1 or d["budget"] < 2 or d["image_size"] < 64:
        raise ValueError("Require recent>=1, B>=2, image_size>=64")
    if len(d["horizons"]) < 2 or min(d["horizons"]) < 3 or sorted(set(d["horizons"])) != d["horizons"]:
        raise ValueError("Use >=2 increasing distinct horizons >=3 (both tasks visible)")
    for name in ["gap", "long_gap", "cue_start", "cue_duration", "motion_dt"]:
        if len(d[name]) != 2 or not 1 <= d[name][0] <= d[name][1]:
            raise ValueError(f"Invalid range {name}")
    if d["long_gap"][0] <= d["gap"][1]:
        raise ValueError("long_gap must be strictly outside training gap range")
    if min(d["pairs"].values()) < 2 or d["cue_start"][0] + min(d["cue_duration"][0], d["motion_dt"][0]) + d["gap"][0] < d["budget"]:
        raise ValueError("Insufficient pairs or history for B")
    if c["model"]["rank"] < 1 or c["model"]["hidden"] % c["model"]["heads"] or not 0 <= c["model"]["dropout"] <= 0.1:
        raise ValueError("Invalid attention dimension/dropout")
    if c["training"]["epochs"] < 1 or c["training"]["batch_size"] < 1 or c["encoder"]["batch_size"] < 1:
        raise ValueError("Batch sizes and epochs must be positive")
    if c["training"]["batch_size"] % 2 or min(c["training"]["pair_weight"], c["training"]["rank_weight"]) < 0:
        raise ValueError("Paired training requires an even batch size and nonnegative pair weight")
    if not c["training"]["seeds"] or len(set(c["training"]["seeds"])) != len(c["training"]["seeds"]):
        raise ValueError("Training seeds must be distinct")
    if c["normalization"]["std_floor"] <= 0 or c["evaluation"]["bootstrap"] < 100:
        raise ValueError("Require positive std floor and >=100 bootstrap replicates")


def load_config(path, output=None, device=None, seed=None):
    with open(path, encoding="utf-8") as f:
        c = merge(DEFAULT, yaml.safe_load(f) or {})
    if output is not None:
        c["output"] = str(output)
    if device is not None:
        c["device"] = device
    if seed is not None:
        c["training"]["seeds"] = [seed]
    c["output"] = str(Path(c["output"]).expanduser().resolve())
    validate(c)
    return c


def scientific_hash(c):
    return digest({k: v for k, v in c.items() if k not in {"output", "device", "threads"}})


def prepare(c):
    root = Path(c["output"])
    root.mkdir(parents=True, exist_ok=True)
    p = root / "resolved_config.json"
    if p.exists() and scientific_hash(read_json(p)) != scientific_hash(c):
        raise ValueError("Run configuration mismatch. Choose a NEW output directory; no stale cache/checkpoint reuse.")
    write_json(p, c)
    return root
