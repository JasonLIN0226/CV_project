"""Audit only an already generated v2.1 train/validation dataset.

No generation, encoder, model training, test split, or legacy dataset access occurs
here. The sole classifier is the preregistered hidden-state leakage diagnostic.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np


SPLIT_PAIRS = {"train": 512, "validation": 96}
OBSERVATION_KEYS = {"rgb", "actions", "query"}
RECORD_KEYS = {"id", "pair", "sibling", "split", "task", "seed", "scene_index",
               "query", "gap", "frames", "obs", "eval", "obs_sha256", "eval_sha256"}
EVALUATION_KEYS = {"states", "velocity", "classes", "visible", "sufficient", "label",
                   "evidence", "t_on", "t_c", "end", "L0", "L1", "v0", "v1",
                   "speed", "x_end", "x_pre"}
LEAKAGE_LIMITS = {"end_only": [0.47, 0.53], "one_post_non_end": [0.0, 0.58],
                  "stale_plus_end": [0.47, 0.53], "two_post": [0.95, 1.0]}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def close(a, b, message):
    check(np.allclose(a, b, atol=2e-7, rtol=0.0), message)


def scalar(meta, key):
    check(np.asarray(meta[key]).shape == (), f"{key} must be a scalar")
    return np.asarray(meta[key]).item()


def _reflect(x, velocity):
    while not 0.08 <= x <= 0.92:
        if x < 0.08:
            x, velocity = 0.16 - x, -velocity
        else:
            x, velocity = 1.84 - x, -velocity
    return x, velocity


def _read_episode(root, record):
    # Validate the split and complete path before opening either NPZ.
    check(record["split"] in SPLIT_PAIRS, "Forbidden or unknown split in manifest")
    check(set(record) == RECORD_KEYS, "Unexpected manifest record fields")
    values = []
    for field, directory in (("obs", "observations"), ("eval", "evaluation_only")):
        expected = Path(directory) / record["split"] / f"{record['id']}.npz"
        check(Path(record[field]) == expected, f"Unexpected {field} path")
        path = (root / expected).resolve()
        check(path.is_relative_to(root.resolve()), "Dataset path escapes dataset root")
        check(file_sha(path) == record[field + "_sha256"], f"{field} hash mismatch")
        with np.load(path, allow_pickle=False) as data:
            values.append({key: data[key] for key in data.files})
    observation, meta = values
    check(set(observation) == OBSERVATION_KEYS, "Hidden metadata or extra keys in model observation")
    check(set(meta) == EVALUATION_KEYS, "Unexpected evaluation-only schema")
    return observation, meta


def audit_episode(record, observation, meta):
    q = int(record["query"])
    length = int(record["frames"])
    t_on, t_c, end, l0, l1 = (int(scalar(meta, key)) for key in ("t_on", "t_c", "end", "L0", "L1"))
    label = int(scalar(meta, "label"))
    check(label in (0, 1), "Invalid sibling label")
    check(0 <= t_on <= 6 and 4 <= l0 <= 10 and 3 <= l1 <= 7, "Timing range violation")
    check(t_c == t_on + l0 and end == t_c + l1 - 1, "Timing relation violation")
    check(5 <= record["gap"] <= 10, "Train/validation must use ordinary gap")
    check(q == end + record["gap"] + 4, "Query/gap relation violation")
    check(length == q + 7, "Horizon/length violation")
    check(np.asarray(observation["query"]).shape == () and int(observation["query"]) == q, "Observation query mismatch")
    rgb, actions = observation["rgb"], observation["actions"]
    check(rgb.dtype == np.uint8 and rgb.shape == (length, 96, 96, 3), "RGB shape/type mismatch")
    check(actions.dtype == np.float32 and actions.shape == (length - 1, 2), "Action shape/type mismatch")
    check(np.all(np.isfinite(actions)) and np.all(np.abs(actions) <= 1), "Invalid actions")
    state, velocity, visible, sufficient = (meta[k] for k in ("states", "velocity", "visible", "sufficient"))
    check(state.dtype == np.float32 and state.shape == (length, 2), "State shape/type mismatch")
    check(velocity.dtype == np.float32 and velocity.shape == (length,), "Velocity shape/type mismatch")
    check(visible.dtype == np.bool_ and visible.shape == (length,), "Visibility shape/type mismatch")
    check(sufficient.dtype == np.bool_ and sufficient.shape == (length,), "Sufficient mask shape/type mismatch")
    check(np.all(np.isfinite(state)) and np.all(np.isfinite(velocity)), "Non-finite state")
    check(meta["classes"].dtype == np.int64 and meta["classes"].shape == (length,), "Class schema mismatch")
    check(not visible[:t_on].any(), "Pre-roll is not fully occluded")
    check(visible[t_on:t_c].sum() >= 2, "Fewer than two visible old-segment frames")
    check(visible[t_c:end + 1].sum() >= 2 and visible[end], "Insufficient post-change visibility")
    check(not visible[end + 1:q + 2].any(), "Long shutter does not extend through query+1")
    check(visible[q + 2:].all(), "Future target frames must be visible")
    expected_sufficient = visible & (np.arange(length) >= t_c) & (np.arange(length) <= end)
    check(np.array_equal(sufficient, expected_sufficient), "Incorrect sufficient-frame mask")
    ident = np.flatnonzero(sufficient)
    check(np.array_equal(meta["evidence"], [ident[0], ident[-1]]), "Evidence should be first/last visible new frame")
    check(np.all(meta["evidence"] < q - 3), "Evidence enters recent window")
    check(not visible[q - 3:q + 1].any(), "Recent four frames are not all shutters")
    check(np.all(meta["classes"][~visible] == -1), "Occluded frame class should be absent")

    x_end, x_pre = float(scalar(meta, "x_end")), float(scalar(meta, "x_pre"))
    v0, v1, speed = (float(scalar(meta, k)) for k in ("v0", "v1", "speed"))
    check(0.20 <= x_end <= 0.80 and 0.20 <= x_pre <= 0.80, "Position sampling bounds violation")
    check(0.005 <= abs(v0) <= 0.007 and 0.008 <= speed <= 0.012, "Speed sampling bounds violation")
    check(v1 == (2 * label - 1) * speed, "Signed final velocity does not match sibling label")
    expected_state, expected_velocity = np.empty((length, 2)), np.empty(length)
    x, v = x_pre, v0
    for t in range(t_c):
        expected_state[t], expected_velocity[t] = [x, 0.40], v
        x, v = _reflect(x + v, v)
    for t in range(t_c, end + 1):
        expected_state[t], expected_velocity[t] = [x_end + v1 * (t - end), 0.62], v1
    x, v = x_end, v1
    for t in range(end + 1, length):
        a = actions[t - 1]
        x, v = _reflect(x + v + 0.0015 * float(a[0]) + 0.001 * float(a[1]), v)
        expected_state[t], expected_velocity[t] = [x, 0.62], v
    close(state, expected_state, "Trajectory or Phase 1 action-transition convention mismatch")
    close(velocity, expected_velocity, "Stored intrinsic velocity mismatch")
    check(np.all((state[:, 0] >= 0.08 - 2e-7) & (state[:, 0] <= 0.92 + 2e-7)), "Position outside reflecting bounds")
    check(np.all((state[t_c:end + 1, 0] >= 0.128 - 2e-7) & (state[t_c:end + 1, 0] <= 0.872 + 2e-7)), "New segment outside non-reflecting bounds")
    # Validate a region strictly inside the shutter, away from the striped lines.
    # Pixel (row=48, col=20) is never a stripe in the Phase 1 96px renderer.
    check(np.all(rgb[~visible, 48, 20] == np.array([80, 89, 102], np.uint8)), "Invisible frames lack an actual RGB shutter")
    check(any(not np.array_equal(rgb[0, 80:], rgb[t, 80:]) for t in range(1, length)), "Scene distractors never move")
    return {"q": q, "t_on": t_on, "t_c": t_c, "end": end, "visible_old": int(visible[:t_c].sum()),
            "visible_new": int(sufficient.sum())}


def _policy_success(meta, q, name):
    stop = q - 3
    memory = [0, 1]
    for t in range(2, stop):
        candidates = memory + [t]
        if name == "fifo":
            memory = candidates[1:]
        elif name == "keep_first":
            pass
        elif name == "latest_visible_reference":
            keep = [i for i in candidates if meta["visible"][i]]
            if len(keep) >= 2:
                memory = keep[-2:]
            else:
                memory = keep + [i for i in candidates if i not in keep][-(2 - len(keep)):]
        else:
            raise ValueError(name)
    return bool(meta["sufficient"][memory].all())


def leakage_audit(train_pairs):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold, cross_val_score
    rng = np.random.default_rng(7)
    rows = {name: [] for name in LEAKAGE_LIMITS}
    groups, labels, choices = [], [], []
    for group, (pair_id, q, metas) in enumerate(train_pairs):
        a, b = sorted(metas, key=lambda meta: int(meta["label"]))
        ident = np.flatnonzero(a["sufficient"])
        end, t_c = int(a["end"]), int(a["t_c"])
        old = np.flatnonzero(a["visible"][:t_c])
        one = int(rng.choice(ident[ident != end]))
        two = sorted(int(v) for v in rng.choice(ident, 2, replace=False))
        stale = int(rng.choice(old))
        selections = {"end_only": [end], "one_post_non_end": [one], "stale_plus_end": [stale, end], "two_post": two}
        choices.append({"pair": pair_id, **selections})
        for meta in (a, b):
            for name, times in selections.items():
                row = []
                for t in times:
                    row.extend([t - q, float(meta["states"][t, 0]), float(meta["states"][t, 1]), float(meta["visible"][t])])
                rows[name].append(row)
            labels.append(int(meta["label"]))
            groups.append(group)
    groups, labels = np.asarray(groups), np.asarray(labels)
    results = {}
    for name, values in rows.items():
        X = np.asarray(values, dtype=np.float64)
        if name in ("end_only", "stale_plus_end"):
            check(np.array_equal(X[0::2], X[1::2]), f"{name} must be sibling-identical")
        scores = cross_val_score(HistGradientBoostingClassifier(max_iter=200, random_state=0), X, labels,
                                 groups=groups, cv=GroupKFold(5), scoring="accuracy", n_jobs=1)
        value = float(np.mean(scores))
        low, high = LEAKAGE_LIMITS[name]
        results[name] = {"mean_accuracy": value, "fold_accuracies": scores.tolist(), "allowed_interval": [low, high],
                         "passed": low <= value <= high, "feature_dimensions": int(X.shape[1])}
    return {"status": "passed" if all(r["passed"] for r in results.values()) else "failed", "source_split": "train",
            "scenes": len(train_pairs), "examples": len(labels), "folds": 5, "group_unit": "sibling_pair",
            "frame_choice_seed": 7, "classifier": {"name": "HistGradientBoostingClassifier", "max_iter": 200, "random_state": 0},
            "selection_sha256": digest(choices), "results": results}


def audit_dataset(dataset):
    started = time.perf_counter()
    root = Path(dataset).resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    config_path = root.parent / "resolved_config.json"
    config = json.loads(config_path.read_text())
    expected_data = {"seed": 20261030, "tasks": ["motion"], "image_size": 96, "recent": 4, "budget": 2,
                     "horizons": [3, 6], "gap": [5, 10], "long_gap": [20, 32], "pairs": SPLIT_PAIRS}
    check(config.get("generator_version") == "motion_v2.1" and config.get("data") == expected_data,
          "Resolved generation settings differ from the registered configuration")
    check(manifest.get("generator_version") == "motion_v2.1" and manifest.get("mode") == "train_validation_only",
          "Unexpected generator version or dataset mode")
    source_dir = Path(__file__).parent
    source_hashes = {name: file_sha(source_dir / name) for name in ("envs_v2.py", "generate_v2.py")}
    check(manifest.get("source_hashes") == source_hashes, "Manifest generator source hashes differ")
    check(manifest.get("config_hash") == digest(expected_data), "Manifest data configuration hash differs")
    check(manifest.get("code_hash") == digest(source_hashes), "Manifest code hash differs")
    check(manifest.get("signature") == digest({"data": expected_data, "generator_version": "motion_v2.1",
                                               "implementation": digest(source_hashes)}), "Manifest signature differs")
    records = manifest["records"]
    check(all(r.get("split") in SPLIT_PAIRS for r in records), "Manifest contains a test/unknown split; refusing all sample reads")
    counts = Counter(r["split"] for r in records)
    check(dict(counts) == {s: 2 * n for s, n in SPLIT_PAIRS.items()}, "Required train/validation dataset sizes differ")
    check(all(r.get("task") == "motion" for r in records), "Dataset is not motion-only")
    check(len({r["id"] for r in records}) == len(records), "Repeated episode id")
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["split"], record["pair"])].append(record)
    pair_rows, failures, train_pairs = [], [], []
    online = {split: {name: [] for name in ("fifo", "keep_first", "hindsight_oracle", "latest_visible_reference")} for split in SPLIT_PAIRS}
    for split in SPLIT_PAIRS:
        groups = [(pair_id, pair) for (sp, pair_id), pair in grouped.items() if sp == split]
        groups.sort(key=lambda p: p[1][0]["scene_index"])
        check(len(groups) == SPLIT_PAIRS[split], f"Pair count mismatch for {split}")
        check([pair[0]["scene_index"] for _, pair in groups] == list(range(SPLIT_PAIRS[split])), "Scene indices missing or duplicated")
        for pair_id, pair in groups:
            try:
                check(len(pair) == 2, "Pair must have exactly two siblings")
                index = int(pair[0]["scene_index"])
                check(pair[1]["scene_index"] == index, "Sibling scene indices differ")
                expected_pair = digest([20261030, "motion", split, index])[:24]
                check(pair_id == expected_pair, "Pair ID formula mismatch")
                expected_ids = {digest([pair_id, j, "opaque-id"])[:24] for j in range(2)}
                check({r["id"] for r in pair} == expected_ids, "Opaque sibling ID formula mismatch")
                for record in pair:
                    check(record["sibling"] in expected_ids - {record["id"]}, "Sibling pointer mismatch")
                    expected_seed = int(digest((20261030, "motion", split, index))[:16], 16) % (2**32)
                    check(record["seed"] == expected_seed, "Scene seed formula mismatch")
                (oa, ma), (ob, mb) = [_read_episode(root, rec) for rec in pair]
                details = [audit_episode(rec, obs, meta) for rec, obs, meta in zip(pair, (oa, ob), (ma, mb))]
                check(details[0] == details[1], "Shared timing/visibility counts differ")
                check({int(ma["label"]), int(mb["label"])} == {0, 1}, "Sibling labels not balanced")
                q, t_c, end = details[0]["q"], details[0]["t_c"], details[0]["end"]
                for key in ("visible", "sufficient", "t_on", "t_c", "end", "L0", "L1", "v0", "speed", "x_end", "x_pre", "evidence"):
                    check(np.array_equal(ma[key], mb[key]), f"Shared metadata {key} differs")
                check(np.array_equal(oa["actions"], ob["actions"]), "Actions reveal sibling label")
                check(np.array_equal(oa["rgb"][:t_c], ob["rgb"][:t_c]), "Pre-change RGB differs")
                check(np.array_equal(ma["states"][:t_c], mb["states"][:t_c]), "Old-state trajectory differs")
                check(np.array_equal(ma["velocity"][:t_c], mb["velocity"][:t_c]), "Old velocity differs")
                check(np.array_equal(oa["rgb"][q - 3:q + 1], ob["rgb"][q - 3:q + 1]), "Recent RGB differs")
                check(np.array_equal(ma["states"][end], mb["states"][end]), "End position differs")
                check(np.array_equal(oa["rgb"][end], ob["rgb"][end]), "End RGB differs")
                differed = np.any(oa["rgb"][:q + 1] != ob["rgb"][:q + 1], axis=(1, 2, 3))
                check(not np.any(differed & ~ma["sufficient"][:q + 1]), "History differs outside allowed visible new segment")
                check(differed.any(), "No post-change visual evidence differs")
                for h in (3, 6):
                    check(ma["visible"][q + h] and mb["visible"][q + h], "Target occluded")
                    check(not np.array_equal(ma["states"][q + h], mb["states"][q + h]), "Sibling target positions identical")
                for meta in (ma, mb):
                    for name in ("fifo", "keep_first", "latest_visible_reference"):
                        online[split][name].append(_policy_success(meta, q, name))
                    online[split]["hindsight_oracle"].append(bool(meta["sufficient"][:q - 3].sum() >= 2))
                if split == "train":
                    train_pairs.append((pair_id, q, (ma, mb)))
                pair_rows.append({"pair": pair_id, "split": split, "scene_index": index, "passed": True, **details[0]})
            except (AssertionError, KeyError, OSError, ValueError) as exc:
                failure = {"pair": pair_id, "split": split, "error": f"{type(exc).__name__}: {exc}"}
                failures.append(failure)
                pair_rows.append({**failure, "passed": False})
    report = {"schema_version": 1, "status": "running", "dataset_manifest_sha256": file_sha(manifest_path),
              "resolved_config_sha256": file_sha(config_path), "generator_version": manifest["generator_version"],
              "dataset": str(root), "data_seed": 20261030, "split_pairs": SPLIT_PAIRS, "raw_test_access": False,
              "environment": {"python": sys.version, "platform": platform.platform(),
                              "packages": {p: importlib.metadata.version(p) for p in ("numpy", "Pillow", "scikit-learn", "scipy", "pytest")}},
              "structure": {"status": "failed" if failures else "passed", "pairs_checked": len(pair_rows),
                            "pairs_passed": len(pair_rows) - len(failures), "failures": failures, "pairs": pair_rows}}
    if failures:
        report.update(status="failed", leakage={"status": "not_run_structure_failed"}, online={"status": "not_run_structure_failed"})
    else:
        online_results = {}
        for split, policy in online.items():
            rates = {name: float(np.mean(values)) for name, values in policy.items()}
            passed = rates["fifo"] <= 0.02 and rates["keep_first"] <= 0.02 and rates["hindsight_oracle"] == 1.0 and rates["latest_visible_reference"] == 1.0
            online_results[split] = {"episodes": 2 * SPLIT_PAIRS[split], "rates": rates, "passed": passed}
        report["online"] = {"status": "passed" if all(s["passed"] for s in online_results.values()) else "failed", "splits": online_results,
                            "criterion": "Both retained history frames are sufficient; no claim about pixel/feature decoding", "thresholds": {"fifo_max": 0.02, "keep_first_max": 0.02, "hindsight_oracle": 1.0, "latest_visible_reference": 1.0}}
        if report["online"]["status"] == "passed":
            report["leakage"] = leakage_audit(train_pairs)
        else:
            report["leakage"] = {"status": "not_run_online_failed"}
        report["status"] = "passed" if report["online"]["status"] == "passed" and report["leakage"]["status"] == "passed" else "failed"
    report["cpu_wall_seconds"] = time.perf_counter() - started
    report["gpu_seconds"] = 0
    return report


def write_audit(dataset, output):
    output = Path(output)
    check(not output.exists(), "Refusing to overwrite an existing audit result")
    report = audit_dataset(dataset)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as destination:
        json.dump(report, destination, indent=2)
        destination.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = write_audit(args.dataset, args.output)
    print(json.dumps({"status": result["status"], "pairs_passed": result["structure"]["pairs_passed"],
                      "online": result["online"], "leakage": result["leakage"], "cpu_wall_seconds": result["cpu_wall_seconds"]}, indent=2))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
