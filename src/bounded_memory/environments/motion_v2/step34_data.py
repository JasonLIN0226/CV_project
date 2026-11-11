"""Registered validation-only long-gap data and CPU leakage diagnostics.

Importing this module does not generate/read episodes or fit classifiers. The
ordinary dataset is immutable; its train split is the only probe fitting source.
All output writes are exclusive creations under a new caller-supplied directory.
"""

from collections import Counter
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import re
import sys
import time

import numpy as np


DATA_SEED = 20261101
RANDOM_POST_SEED = 20261102
PROBE_FRAME_SEED = 7
PAIR_COUNT = 96
GENERATOR_VERSION = "motion_v2.1_validation_longgap_v1"
CONFIG_VERSION = "motion_step34_longgap_v1"
ORDINARY_MANIFEST_SHA256 = "9ac43b81eb75456e17f3542e64e8a0ab58282207752ae3712836f6d2ea20032a"
ENVS_SHA256 = "6949d94c78e7fd87606d0552d0b5f292e49c2238029f2ba13de902ef4164987f"
DATA_CONFIG = {
    "seed": DATA_SEED, "tasks": ["motion"], "image_size": 96,
    "recent": 4, "budget": 2, "horizons": [3, 6],
    "gap": [20, 32], "pairs": {"validation": PAIR_COUNT},
}
RECORD_KEYS = {"id", "pair", "sibling", "split", "task", "seed", "scene_index",
               "query", "gap", "frames", "obs", "eval", "obs_sha256", "eval_sha256"}
OBSERVATION_KEYS = {"rgb", "actions", "query"}
EVALUATION_KEYS = {"states", "velocity", "classes", "visible", "sufficient", "label",
                   "evidence", "t_on", "t_c", "end", "L0", "L1", "v0", "v1",
                   "speed", "x_end", "x_pre"}
PROBE_KEYS = ("states", "visible", "sufficient", "label", "t_c", "end")
LEAKAGE_LIMITS = {"end_only": (0.47, 0.53), "one_post_non_end": (0.0, 0.58),
                  "stale_plus_end": (0.47, 0.53), "two_post": (0.95, 1.0)}
_ENVIRONMENT_MODULE = None


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def _array_sha256(value):
    value = np.ascontiguousarray(value)
    checksum = hashlib.sha256(json.dumps({"shape": list(value.shape), "dtype": value.dtype.str},
                                        sort_keys=True, separators=(",", ":")).encode())
    checksum.update(value.tobytes())
    return checksum.hexdigest()


def _write_json_new(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def _write_npz_new(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)


def _envs():
    global _ENVIRONMENT_MODULE
    path = Path(__file__).with_name("envs_v2.py")
    _require(_sha256(path) == ENVS_SHA256, "Frozen v2.1 environment source changed")
    if _ENVIRONMENT_MODULE is None:
        spec = importlib.util.spec_from_file_location("motion_step34_frozen_envs", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _ENVIRONMENT_MODULE = module
    return _ENVIRONMENT_MODULE


def _sources():
    return {"envs_v2.py": _sha256(Path(__file__).with_name("envs_v2.py")),
            "step34_data.py": _sha256(Path(__file__))}


def _scene_seed(seed, index):
    return int(_digest((seed, "motion", "validation", index))[:16], 16) % (2**32)


def _pair_id(seed, split, index):
    return _digest([seed, "motion", split, index])[:24]


def _validate_record(record, split):
    """Positive allowlist applied before opening any observation/evaluator NPZ."""
    _require(split in ("train", "validation"), "Unauthorized requested split")
    _require(isinstance(record, dict) and set(record) == RECORD_KEYS, "Unexpected public record schema")
    _require(record["split"] == split and record["task"] == "motion", "Unauthorized episode split/task")
    for key in ("id", "pair", "sibling"):
        _require(isinstance(record[key], str) and re.fullmatch(r"[0-9a-f]{24}", record[key]),
                 "Invalid opaque identity")
    for key in ("seed", "scene_index", "query", "gap", "frames"):
        _require(type(record[key]) is int and record[key] >= 0, "Invalid integer record field: " + key)
    for kind, directory in (("obs", "observations"), ("eval", "evaluation_only")):
        _require(record[kind] == str(Path(directory) / split / (record["id"] + ".npz")),
                 "Episode path is outside the authorized split layout")
        _require(isinstance(record[kind + "_sha256"], str)
                 and re.fullmatch(r"[0-9a-f]{64}", record[kind + "_sha256"]), "Invalid episode hash")


def _read_arrays(root, record, split, kind, keys=None):
    _validate_record(record, split)
    _require(kind in ("obs", "eval"), "Unknown archive type")
    root = Path(root).resolve()
    path = (root / record[kind]).resolve()
    _require(path.is_relative_to(root), "Episode path escapes the dataset root")
    _require(_sha256(path) == record[kind + "_sha256"], "Episode SHA256 mismatch")
    expected = OBSERVATION_KEYS if kind == "obs" else EVALUATION_KEYS
    requested = expected if keys is None else set(keys)
    _require(requested <= expected and requested, "Unexpected requested archive members")
    with np.load(path, allow_pickle=False) as archive:
        _require(set(archive.files) == expected, "Archive schema violates the observation/evaluator boundary")
        return {key: archive[key] for key in sorted(requested)}


def _ordinary_manifest(dataset):
    root = Path(dataset).resolve()
    path = root / "manifest.json"
    _require(_sha256(path) == ORDINARY_MANIFEST_SHA256, "Original train/validation manifest changed")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _require(manifest.get("generator_version") == "motion_v2.1"
             and manifest.get("mode") == "train_validation_only", "Wrong original dataset provenance")
    records = manifest["records"]
    _require(len(records) == 1216, "Original manifest must enumerate exactly 1216 episodes")
    _require(Counter(r.get("split") for r in records) == {"train": 1024, "validation": 192},
             "Original manifest contains a forbidden split or wrong count")
    for record in records:
        _validate_record(record, record["split"])
    _require(len({r["id"] for r in records}) == len(records), "Original episode identities repeat")
    return root, manifest


def _scene_longgap(index):
    """Same v2.1 RNG draw order and scene rules, with new seed and gap range."""
    _require(type(index) is int and 0 <= index < PAIR_COUNT, "Long-gap scene index outside 0..95")
    seed = _scene_seed(DATA_SEED, index)
    rng = np.random.default_rng(seed)
    gap = int(rng.integers(20, 33))
    while True:
        t_on = int(rng.integers(0, 7))
        length_old = int(rng.integers(4, 11))
        length_new = int(rng.integers(3, 8))
        t_c = t_on + length_old
        end = t_c + length_new - 1
        query = end + gap + 4
        visible = np.array([(t_on <= t <= end) and rng.random() > 0.25
                            for t in range(query + 1)], dtype=bool)
        visible[end] = True
        if visible[t_c:end + 1].sum() >= 2 and visible[:t_c].sum() >= 2:
            break
    x_end = float(rng.uniform(0.20, 0.80))
    speed = float(rng.uniform(0.008, 0.012))
    x_pre = float(rng.uniform(0.20, 0.80))
    v0 = float(rng.choice([-1, 1]) * rng.uniform(0.005, 0.007))
    length = query + 7
    full_visibility = np.zeros(length, dtype=bool)
    full_visibility[:query + 1] = visible
    full_visibility[query + 2:] = True
    actions = rng.uniform(-1, 1, (length - 1, 2)).astype(np.float32)
    label_order = rng.permutation([0, 1]).tolist()
    distractors = np.empty((length, 8, 3), dtype=np.float64)
    nuisance = rng.uniform(0, 1, (8, 3))
    for t in range(length):
        nuisance = np.clip(nuisance + rng.normal(0, 0.08, nuisance.shape), 0, 1)
        distractors[t] = nuisance
    return {"seed": seed, "task": "motion", "split": "validation", "query": query,
            "gap": gap, "length": length, "size": 96, "t_on": t_on, "t_c": t_c,
            "end": end, "L0": length_old, "L1": length_new, "x_end": x_end,
            "speed": speed, "x_pre": x_pre, "v0": v0, "visible": full_visibility,
            "actions": actions, "label_order": label_order, "distractors": distractors, "shift": False}


def generate_longgap(output: Path, ordinary_dataset: Path):
    """Create exactly 96 validation sibling pairs in new output/dataset."""
    output = Path(output).absolute()
    _require(not output.exists() and not output.is_symlink(), "Output already exists; refusing overwrite/retry")
    ordinary_root, ordinary = _ordinary_manifest(ordinary_dataset)
    resolved_output = output.resolve()
    _require(not resolved_output.is_relative_to(ordinary_root)
             and not ordinary_root.is_relative_to(resolved_output), "New output overlaps immutable ordinary data")
    envs = _envs()
    sources = _sources()
    config = {"config_version": CONFIG_VERSION, "generator_version": GENERATOR_VERSION,
              "data": DATA_CONFIG, "ordinary_manifest_sha256": ORDINARY_MANIFEST_SHA256}
    output.mkdir(parents=True, exist_ok=False)
    root = output / "dataset"
    root.mkdir()
    _write_json_new(output / "resolved_config.json", config)
    began = time.perf_counter()
    records = []
    seen_pairs = {r["pair"] for r in ordinary["records"]}
    seen_ids = {r["id"] for r in ordinary["records"]}
    seen_seeds = {r["seed"] for r in ordinary["records"]}
    for index in range(PAIR_COUNT):
        shared = _scene_longgap(index)
        pair = _pair_id(DATA_SEED, "validation", index)
        ids = [_digest([pair, slot, "opaque-id"])[:24] for slot in range(2)]
        _require(pair not in seen_pairs and shared["seed"] not in seen_seeds
                 and len(set(ids)) == 2 and not (set(ids) & seen_ids), "Data identity/scene-seed collision")
        seen_pairs.add(pair)
        seen_ids.update(ids)
        seen_seeds.add(shared["seed"])
        for slot, eid in enumerate(ids):
            episode = envs.episode(shared, shared["label_order"][slot])
            obs, meta = episode["observations"], episode["evaluation_only"]
            _require(set(obs) == OBSERVATION_KEYS and set(meta) == EVALUATION_KEYS,
                     "Frozen episode observation/evaluator schema changed")
            obs_path = root / "observations" / "validation" / (eid + ".npz")
            eval_path = root / "evaluation_only" / "validation" / (eid + ".npz")
            _write_npz_new(obs_path, obs)
            _write_npz_new(eval_path, meta)
            records.append({"id": eid, "pair": pair, "sibling": ids[1 - slot], "split": "validation",
                            "task": "motion", "seed": shared["seed"], "scene_index": index,
                            "query": shared["query"], "gap": shared["gap"], "frames": shared["length"],
                            "obs": str(obs_path.relative_to(root)), "eval": str(eval_path.relative_to(root)),
                            "obs_sha256": _sha256(obs_path), "eval_sha256": _sha256(eval_path)})
    manifest = {"generator_version": GENERATOR_VERSION, "config_version": CONFIG_VERSION,
                "mode": "validation_longgap_only", "config_hash": _digest(DATA_CONFIG),
                "source_hashes": sources, "code_hash": _digest(sources),
                "signature": _digest({"data": DATA_CONFIG, "generator_version": GENERATOR_VERSION,
                                      "implementation": _digest(sources)}),
                "ordinary_manifest_sha256": ORDINARY_MANIFEST_SHA256,
                "generation_seconds": time.perf_counter() - began, "records": records}
    _write_json_new(root / "manifest.json", manifest)
    _write_json_new(output / "generation_receipt.json", {
        "status": "generated_awaiting_registered_audit", "dataset_manifest_sha256": _sha256(root / "manifest.json"),
        "source_hashes": sources, "pairs": {"validation": PAIR_COUNT}, "episodes": len(records),
        "frames": sum(r["frames"] for r in records), "ordinary_manifest_sha256": ORDINARY_MANIFEST_SHA256,
        "ordinary_episode_arrays_read": 0, "test_or_confirmation_generated_or_opened": False,
    })
    return manifest


def random_post_pair(visible, tc, end, pair_id, seed=RANDOM_POST_SEED):
    """One fixed sorted sample from new visible frames, shared by both siblings."""
    visible = np.asarray(visible)
    _require(visible.dtype == np.bool_ and visible.ndim == 1, "Visibility must be a one-dimensional bool array")
    _require(isinstance(tc, (int, np.integer)) and not isinstance(tc, (bool, np.bool_))
             and isinstance(end, (int, np.integer)) and not isinstance(end, (bool, np.bool_))
             and 0 <= tc < end < len(visible), "Invalid new-segment bounds")
    _require(type(seed) is int and seed >= 0, "Frame-selection seed must be a nonnegative integer")
    _require(isinstance(pair_id, str) and re.fullmatch(r"[0-9a-f]{24}", pair_id), "Invalid pair identity")
    post = int(tc) + np.flatnonzero(visible[int(tc):int(end) + 1])
    _require(len(post) >= 2 and bool(visible[int(end)]), "New segment lacks two visible frames or visible end")
    draw_seed = int(_digest((seed, pair_id, "random_post_pair"))[:16], 16) % (2**32)
    rng = np.random.default_rng(draw_seed)
    return np.sort(rng.choice(post, size=2, replace=False)).astype(np.int64)


def _scalar(meta, key):
    value = np.asarray(meta[key])
    _require(value.shape == (), "Evaluator field must be scalar: " + key)
    return value.item()


def _reflect(x, velocity):
    while not 0.08 <= x <= 0.92:
        if x < 0.08:
            x, velocity = 0.16 - x, -velocity
        else:
            x, velocity = 1.84 - x, -velocity
    return x, velocity


def _audit_episode(record, observation, meta):
    """Independent counterpart of step1 audit, with registered long-gap bounds."""
    _validate_record(record, "validation")
    _require(set(observation) == OBSERVATION_KEYS and set(meta) == EVALUATION_KEYS,
             "Hidden fields entered observations or evaluator schema changed")
    q, length = record["query"], record["frames"]
    t_on, tc, end, l0, l1 = (int(_scalar(meta, k)) for k in ("t_on", "t_c", "end", "L0", "L1"))
    label = int(_scalar(meta, "label"))
    _require(label in (0, 1) and 0 <= t_on <= 6 and 4 <= l0 <= 10 and 3 <= l1 <= 7,
             "Timing or sibling label outside registered bounds")
    _require(tc == t_on + l0 and end == tc + l1 - 1, "Timing identities differ")
    _require(20 <= record["gap"] <= 32 and q == end + record["gap"] + 4 and length == q + 7,
             "Long-gap query/horizon relation differs")
    rgb, actions = observation["rgb"], observation["actions"]
    _require(np.asarray(observation["query"]).shape == () and int(observation["query"]) == q,
             "Observation query differs")
    _require(rgb.dtype == np.uint8 and rgb.shape == (length, 96, 96, 3), "Invalid RGB array")
    _require(actions.dtype == np.float32 and actions.shape == (length - 1, 2)
             and np.isfinite(actions).all() and (np.abs(actions) <= 1).all(), "Invalid action array")
    state, velocity, visible, sufficient = (meta[k] for k in ("states", "velocity", "visible", "sufficient"))
    _require(state.dtype == np.float32 and state.shape == (length, 2) and np.isfinite(state).all(), "Invalid states")
    _require(velocity.dtype == np.float32 and velocity.shape == (length,) and np.isfinite(velocity).all(), "Invalid velocities")
    _require(visible.dtype == np.bool_ and visible.shape == (length,)
             and sufficient.dtype == np.bool_ and sufficient.shape == (length,), "Invalid visibility masks")
    _require(meta["classes"].dtype == np.int64 and meta["classes"].shape == (length,), "Invalid class array")
    _require(not visible[:t_on].any() and visible[t_on:tc].sum() >= 2, "Old-segment visibility constraint failed")
    _require(visible[tc:end + 1].sum() >= 2 and visible[end], "New-segment visibility constraint failed")
    _require(not visible[end + 1:q + 2].any() and visible[q + 2:].all(), "Shutter/target visibility differs")
    expected_sufficient = visible & (np.arange(length) >= tc) & (np.arange(length) <= end)
    _require(np.array_equal(sufficient, expected_sufficient), "Sufficient-frame mask differs")
    identifying = np.flatnonzero(sufficient)
    _require(meta["evidence"].dtype == np.int64 and np.array_equal(meta["evidence"], [identifying[0], end])
             and np.all(meta["evidence"] < q - 3), "Evidence indices differ or enter recent window")
    _require(not visible[q - 3:q + 1].any(), "Recent four frames must all be shutters")
    # The renderer computes classes before float32 state serialization; do not
    # introduce an extra rounded-state threshold comparison at exactly x=0.5.
    _require(np.all(meta["classes"][~visible] == -1)
             and np.isin(meta["classes"][visible], [0, 1]).all(), "Evaluator class/visibility values differ")
    x_end, x_pre, v0, v1, speed = (float(_scalar(meta, k)) for k in ("x_end", "x_pre", "v0", "v1", "speed"))
    _require(0.20 <= x_end <= 0.80 and 0.20 <= x_pre <= 0.80
             and 0.005 <= abs(v0) <= 0.007 and 0.008 <= speed <= 0.012
             and v1 == (2 * label - 1) * speed, "Position/speed distribution or sibling sign differs")
    expected_state, expected_velocity = np.empty((length, 2)), np.empty(length)
    x, v = x_pre, v0
    for t in range(tc):
        expected_state[t], expected_velocity[t] = (x, 0.40), v
        x, v = _reflect(x + v, v)
    for t in range(tc, end + 1):
        expected_state[t], expected_velocity[t] = (x_end + v1 * (t - end), 0.62), v1
    x, v = x_end, v1
    for t in range(end + 1, length):
        a = actions[t - 1]
        x, v = _reflect(x + v + 0.0015 * float(a[0]) + 0.001 * float(a[1]), v)
        expected_state[t], expected_velocity[t] = (x, 0.62), v
    _require(np.allclose(state, expected_state, atol=2e-7, rtol=0)
             and np.allclose(velocity, expected_velocity, atol=2e-7, rtol=0), "Physical transition/action mismatch")
    _require(np.all((state[:, 0] >= 0.08 - 2e-7) & (state[:, 0] <= 0.92 + 2e-7)), "Outside reflecting field")
    _require(np.all((state[tc:end + 1, 0] >= 0.128 - 2e-7) & (state[tc:end + 1, 0] <= 0.872 + 2e-7)),
             "New segment outside original non-reflecting bounds")
    _require(np.all(rgb[~visible, 48, 20] == np.array([80, 89, 102], np.uint8)), "Invisible RGB lacks shutter")
    _require(any(not np.array_equal(rgb[0, 80:], rgb[t, 80:]) for t in range(1, length)), "Distractors never move")
    return {"q": q, "t_on": t_on, "t_c": tc, "end": end,
            "visible_old": int(visible[:tc].sum()), "visible_new": int(sufficient.sum())}


def _audit_pair(records, observations, metas):
    _require(len(records) == len(observations) == len(metas) == 2, "A pair must contain exactly two siblings")
    a, b = records
    _require(a["pair"] == b["pair"] and a["sibling"] == b["id"] and b["sibling"] == a["id"]
             and a["scene_index"] == b["scene_index"], "Sibling identity link differs")
    first, second = [_audit_episode(r, o, m) for r, o, m in zip(records, observations, metas)]
    _require(first == second, "Sibling timing/visibility differs")
    ma, mb = metas
    oa, ob = observations
    _require({int(ma["label"]), int(mb["label"])} == {0, 1}, "Sibling labels not balanced")
    for key in ("visible", "sufficient", "t_on", "t_c", "end", "L0", "L1", "v0", "speed", "x_end", "x_pre", "evidence"):
        _require(np.array_equal(ma[key], mb[key]), "Shared evaluator field differs: " + key)
    q, tc, end = first["q"], first["t_c"], first["end"]
    _require(np.array_equal(oa["actions"], ob["actions"]), "Actions reveal sibling")
    _require(np.array_equal(oa["rgb"][:tc], ob["rgb"][:tc])
             and np.array_equal(ma["states"][:tc], mb["states"][:tc])
             and np.array_equal(ma["velocity"][:tc], mb["velocity"][:tc]), "Old segment differs")
    _require(np.array_equal(oa["rgb"][q - 3:q + 1], ob["rgb"][q - 3:q + 1]), "Recent four RGB frames differ")
    _require(np.array_equal(ma["states"][end], mb["states"][end])
             and np.array_equal(oa["rgb"][end], ob["rgb"][end]), "Matched end position/RGB differs")
    differed = np.any(oa["rgb"][:q + 1] != ob["rgb"][:q + 1], axis=(1, 2, 3))
    _require(differed.any() and not np.any(differed & ~ma["sufficient"][:q + 1]),
             "History differs outside visible new segment or has no direction evidence")
    for h in (3, 6):
        _require(ma["visible"][q + h] and mb["visible"][q + h]
                 and not np.array_equal(ma["states"][q + h], mb["states"][q + h]), "Invalid paired future target")
    return first


def _online_success(meta, q, policy):
    memory = [0, 1]
    for t in range(2, q - 3):
        candidates = memory + [t]
        if policy == "fifo":
            memory = candidates[1:]
        elif policy == "keep_first":
            pass
        elif policy == "latest_visible_reference":
            visible = [i for i in candidates if meta["visible"][i]]
            memory = visible[-2:] if len(visible) >= 2 else visible + [i for i in candidates if i not in visible][-(2 - len(visible)):]
        else:
            raise ValueError("Unknown online reference policy")
    return bool(meta["sufficient"][memory].all())


def _group_pairs(records, split, expected_count, data_seed):
    selected = [r for r in records if r["split"] == split]
    groups = {}
    for record in selected:
        _validate_record(record, split)
        groups.setdefault(record["scene_index"], []).append(record)
    _require(len(selected) == expected_count * 2 and set(groups) == set(range(expected_count)), "Incomplete pair/index coverage")
    ordered = []
    for index in range(expected_count):
        pair = groups[index]
        pair_id = _pair_id(data_seed, split, index)
        ids = [_digest([pair_id, slot, "opaque-id"])[:24] for slot in range(2)]
        expected_seed = int(_digest((data_seed, "motion", split, index))[:16], 16) % (2**32)
        _require(len(pair) == 2 and {r["id"] for r in pair} == set(ids)
                 and all(r["pair"] == pair_id and r["seed"] == expected_seed for r in pair), "Pair/episode/scene-seed formula differs")
        pair.sort(key=lambda r: ids.index(r["id"]))
        _require(pair[0]["sibling"] == pair[1]["id"] and pair[1]["sibling"] == pair[0]["id"]
                 and pair[0]["query"] == pair[1]["query"] and pair[0]["frames"] == pair[1]["frames"], "Sibling links/time differ")
        ordered.append(pair)
    return ordered


def _probe_features(pairs):
    """Exactly the old seed7 subset scheme and (t-q,x,y,visible) feature order."""
    rng = np.random.default_rng(PROBE_FRAME_SEED)
    rows = {name: [] for name in LEAKAGE_LIMITS}
    labels, choices = [], []
    for pair_id, q, metas in pairs:
        a, b = sorted(metas, key=lambda m: int(m["label"]))
        _require({int(a["label"]), int(b["label"])} == {0, 1}, "Probe sibling labels not balanced")
        for key in ("visible", "sufficient", "t_c", "end"):
            _require(np.array_equal(a[key], b[key]), "Probe selection metadata differs between siblings")
        post = np.flatnonzero(a["sufficient"])
        end, tc = int(a["end"]), int(a["t_c"])
        old = np.flatnonzero(a["visible"][:tc])
        _require(len(post) >= 2 and post[-1] == end and len(old) >= 2, "Probe lacks sufficient visible frames")
        one = int(rng.choice(post[post != end]))
        two = sorted(int(v) for v in rng.choice(post, 2, replace=False))
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
    values = {key: np.asarray(rows[key], dtype=np.float64) for key in rows}
    for key, x in values.items():
        _require(x.ndim == 2 and np.isfinite(x).all(), "Nonfinite probe inputs")
        if key in ("end_only", "stale_plus_end"):
            _require(np.array_equal(x[0::2], x[1::2]), key + " must be sibling-identical")
    return values, np.asarray(labels, dtype=np.int64), choices


def _leakage_audit(ordinary_root, ordinary, evaluation_pairs):
    from sklearn.ensemble import HistGradientBoostingClassifier

    training_records = _group_pairs(ordinary["records"], "train", 512, 20261030)
    fitting_pairs, fitting_episode_ids = [], []
    for pair in training_records:
        metas = [_read_arrays(ordinary_root, r, "train", "eval", PROBE_KEYS) for r in pair]
        fitting_pairs.append((pair[0]["pair"], pair[0]["query"], metas))
        fitting_episode_ids.extend(r["id"] for r, meta in sorted(zip(pair, metas), key=lambda item: int(item[1]["label"])))
    fit_x, fit_y, fit_choices = _probe_features(fitting_pairs)
    eval_x, eval_y, eval_choices = _probe_features(evaluation_pairs)
    _require(len(fit_y) == 1024 and len(eval_y) == 192, "Probe fitting/evaluation counts differ")
    _require(not ({p[0] for p in fitting_pairs} & {p[0] for p in evaluation_pairs}), "Probe fit/evaluation overlap")
    results = {}
    for name, (low, high) in LEAKAGE_LIMITS.items():
        classifier = HistGradientBoostingClassifier(max_iter=200, random_state=0)
        classifier.fit(fit_x[name], fit_y)
        prediction = classifier.predict(eval_x[name])
        accuracy = float(np.mean(prediction == eval_y))
        results[name] = {"accuracy": accuracy, "mean_accuracy": accuracy, "correct": int(np.sum(prediction == eval_y)),
                         "examples": len(eval_y), "allowed_interval": [low, high], "passed": low <= accuracy <= high,
                         "feature_dimensions": fit_x[name].shape[1], "classifier_parameters": classifier.get_params(),
                         "fit_input_sha256": _array_sha256(fit_x[name]), "evaluation_input_sha256": _array_sha256(eval_x[name]),
                         "prediction_sha256": _array_sha256(prediction)}
    passed = all(value["passed"] for value in results.values())
    return {"passed": passed, "status": "passed" if passed else "failed", "fit_split": "ordinary_train",
            "evaluation_split": "new_validation_longgap", "fit_pairs": 512, "evaluation_pairs": PAIR_COUNT,
            "fit_examples": 1024, "evaluation_examples": 192, "classifier": "HistGradientBoostingClassifier",
            "max_iter": 200, "random_state": 0, "frame_choice_seed": PROBE_FRAME_SEED,
            "input_schema_per_frame": ["t-query", "x", "y", "visible"],
            "input_dtype": "float64", "decision": "unrounded accuracy on all 192 held-out long-gap siblings",
            "cross_validation_or_hyperparameter_selection": False, "validation_used_for_fitting": False,
            "fit_labels_sha256": _array_sha256(fit_y), "evaluation_labels_sha256": _array_sha256(eval_y),
            "fit_episode_ids": fitting_episode_ids, "fit_row_order": "scene_index then sibling label 0,1",
            "fit_pair_ids": [p[0] for p in fitting_pairs], "evaluation_pair_ids": [p[0] for p in evaluation_pairs],
            "fit_choices": fit_choices, "evaluation_choices": eval_choices,
            "fit_selection_sha256": _digest(fit_choices), "evaluation_selection_sha256": _digest(eval_choices),
            "ordinary_manifest_sha256": ORDINARY_MANIFEST_SHA256,
            "ordinary_evaluation_arrays_opened": 1024, "ordinary_validation_arrays_opened": 0,
            "results": results}


def audit_longgap(dataset_root, ordinary_dataset):
    """Audit saved long-gap episodes, then train four train-only leakage probes.

    Returns a JSON-serializable report including passed. It never writes reports,
    regenerates data, fits a predictor/decoder, or accesses test/confirmation data.
    """
    began = time.perf_counter()
    root = Path(dataset_root).resolve()
    ordinary_root, ordinary = _ordinary_manifest(ordinary_dataset)
    _require(root != ordinary_root and not root.is_relative_to(ordinary_root)
             and not ordinary_root.is_relative_to(root), "Long-gap and ordinary data roots overlap")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_path = root.parent / "resolved_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_config = {"config_version": CONFIG_VERSION, "generator_version": GENERATOR_VERSION,
                       "data": DATA_CONFIG, "ordinary_manifest_sha256": ORDINARY_MANIFEST_SHA256}
    sources = _sources()
    _require(sources["envs_v2.py"] == ENVS_SHA256 and config == expected_config, "Generation config/source differs")
    _require(manifest.get("generator_version") == GENERATOR_VERSION and manifest.get("config_version") == CONFIG_VERSION
             and manifest.get("mode") == "validation_longgap_only", "Not a registered validation-only long-gap manifest")
    _require(manifest.get("config_hash") == _digest(DATA_CONFIG) and manifest.get("source_hashes") == sources
             and manifest.get("code_hash") == _digest(sources)
             and manifest.get("ordinary_manifest_sha256") == ORDINARY_MANIFEST_SHA256
             and manifest.get("signature") == _digest({"data": DATA_CONFIG, "generator_version": GENERATOR_VERSION,
                                                        "implementation": _digest(sources)}), "Manifest provenance differs")
    records = manifest["records"]
    _require(len(records) == 192 and all(r.get("split") == "validation" for r in records),
             "Refusing sample reads: only 192 validation long-gap episodes are authorized")
    _require(len({r["id"] for r in records}) == len(records), "Repeated long-gap identity")
    for key in ("id", "pair", "seed"):
        _require(not ({r[key] for r in records} & {r[key] for r in ordinary["records"]}),
                 "Long-gap data overlaps ordinary data: " + key)
    groups = _group_pairs(records, "validation", PAIR_COUNT, DATA_SEED)
    report = {"schema_version": 1, "passed": False, "status": "running", "dataset_manifest_sha256": _sha256(manifest_path),
              "resolved_config_sha256": _sha256(config_path), "source_hashes": sources, "data_seed": DATA_SEED,
              "split_pairs": {"validation": PAIR_COUNT}, "ordinary_manifest_sha256": ORDINARY_MANIFEST_SHA256,
              "test_or_confirmation_opened_or_generated": False, "gpu_seconds": 0,
              "environment": {"python": sys.version, "platform": platform.platform(),
                              "packages": {p: importlib.metadata.version(p) for p in ("numpy", "Pillow", "scikit-learn", "scipy")}}}
    pair_rows, failures, evaluation_pairs = [], [], []
    successes = {name: [] for name in ("fifo", "keep_first", "hindsight_oracle", "latest_visible_reference")}
    for pair in groups:
        try:
            observations = [_read_arrays(root, r, "validation", "obs") for r in pair]
            metas = [_read_arrays(root, r, "validation", "eval") for r in pair]
            details = _audit_pair(pair, observations, metas)
            q = pair[0]["query"]
            for meta in metas:
                for policy in ("fifo", "keep_first", "latest_visible_reference"):
                    successes[policy].append(_online_success(meta, q, policy))
                successes["hindsight_oracle"].append(bool(meta["sufficient"][:q - 3].sum() >= 2))
            evaluation_pairs.append((pair[0]["pair"], q, metas))
            pair_rows.append({"pair": pair[0]["pair"], "scene_index": pair[0]["scene_index"], "passed": True, **details})
        except (ValueError, KeyError, OSError, IndexError, TypeError) as exc:
            row = {"pair": pair[0]["pair"], "scene_index": pair[0]["scene_index"], "passed": False,
                   "error": type(exc).__name__ + ": " + str(exc)}
            pair_rows.append(row)
            failures.append(row)
    report["structure"] = {"passed": not failures, "pairs_checked": len(pair_rows),
                           "pairs_passed": len(pair_rows) - len(failures), "failures": failures, "pairs": pair_rows}
    if failures:
        report.update(status="failed", online={"status": "not_run_structure_failed"},
                      leakage={"status": "not_run_structure_failed"})
    else:
        rates = {name: float(np.mean(values)) for name, values in successes.items()}
        online_passed = (rates["fifo"] <= 0.02 and rates["keep_first"] <= 0.02
                         and rates["hindsight_oracle"] == 1.0 and rates["latest_visible_reference"] == 1.0)
        report["online"] = {"passed": online_passed, "status": "passed" if online_passed else "failed",
                            "episodes": 192, "rates": rates,
                            "criterion": "Both retained frames are evaluator-marked sufficient",
                            "thresholds": {"fifo_max": 0.02, "keep_first_max": 0.02,
                                           "hindsight_oracle": 1.0, "latest_visible_reference": 1.0}}
        if online_passed:
            report["leakage"] = _leakage_audit(ordinary_root, ordinary, evaluation_pairs)
        else:
            report["leakage"] = {"status": "not_run_online_failed", "passed": False}
        report["passed"] = online_passed and report["leakage"]["passed"]
        report["status"] = "passed" if report["passed"] else "failed"
    report["total_frames"] = sum(r["frames"] for r in records)
    report["cpu_wall_seconds"] = time.perf_counter() - began
    return report
