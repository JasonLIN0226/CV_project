"""Train/validation-only feature adapter for the registered Motion v2.2 reader.

Importing this file does not open a dataset, import a model, or launch work.
Hidden metadata is read only by the explicit evaluator/privileged selection
boundary. Model samples contain the same observable tensors as Phase 1.
"""

import copy
import hashlib
import importlib
import importlib.machinery
import json
from pathlib import Path
import re
import sys
from types import ModuleType

import numpy as np


# The frozen predictor source, vendored byte for byte in this package (predictors/phase1). The
# original file pointed at results/phase1_independent_confirmation/.../source/original_src/phase1,
# which holds the same bytes; the SHA256 checks below are unchanged and still enforced.
_REFERENCE_ROOT = Path(__file__).resolve().parents[2] / "predictors/phase1"
_REFERENCE_PACKAGE = "_motion_v2_frozen_phase1"
_REFERENCE_HASHES = {
    "model": "dca27f514900f90950b1536a49b9610b779ddbc312579814c022ac4160bece66",
    "inputs": "b4766e845c10400929fd0f84f8b6f447182008ba8c21e0404fddfd3ea6cbffbd",
    "repair_trial": "a3bba5e0b0737eb01585e716c485aa11fb64bded5d970ed8b5d89167757d2890",
    "training": "0e30f8c45e84a4dc966d3020066c0686b8c4b951a6447c52ecc85d309a14a235",
    "probes": "50219b0167cf6fdd829962db566f14808ea1cdad6c5ebf9628992054e2c53988",
    "encoders": "7a407ad8f03ecf58bac6dd0bd0fceca0aaabae1201298083fe7ccf3cb5aee871",
    "cache": "5935e62802ef2bf8a5590f559948eb1c1fa1d3f9c4826780ef4ce188d7a96bc0",
    "stats": "767912a6a1617f7e932c1c807031c4ae1809e007045007e818990f2e3d6d9d08",
    "utils": "03b0c1fb7f52fc822523125e5fe813cf48883459d48a038ffb22aa7ff87aef14",
    "config": "67b38fea253659bea37cd5b3726674bc2ad9cfb659b712ae05c0eb53d9034084",
}
_REFERENCE_VERIFIED = False
_REFERENCE_MODULES = {}
_RECORD_KEYS = {
    "id", "pair", "sibling", "split", "task", "seed", "scene_index", "query",
    "gap", "frames", "obs", "eval", "obs_sha256", "eval_sha256",
}
_MODEL_KEYS = {"z", "age", "type", "past_actions", "actions", "current"}
_PROJECTED_KEYS = {"projected_z", "projected_current"}
_EVALUATION_KEYS = {"states", "visible", "evidence", "t_c", "end", "sufficient"}
_CONDITIONS = {
    "a": "recent_only", "recent_only": "recent_only",
    "b": "privileged_small_budget", "privileged_small_budget": "privileged_small_budget",
    "c": "stale_two", "d": "end_plus_stale",
    "full_history": "full_history", "random_small_budget": "random_small_budget",
}


def _sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def load_reference(name):
    """Load an unchanged v10 Phase 1 module under a private package alias.

    No repository-wide sys.path edit or replacement of the real phase1 package
    is performed. Normal relative imports resolve inside the frozen source tree.
    The run protocol additionally binds all transitive source files by SHA256.
    """
    global _REFERENCE_VERIFIED
    if name not in _REFERENCE_HASHES:
        raise ValueError("Reference module is outside the registered allowlist")
    if not _REFERENCE_VERIFIED:
        for module, expected in _REFERENCE_HASHES.items():
            path = _REFERENCE_ROOT / (module + ".py")
            _require(path.is_file() and _sha256(path) == expected,
                     "Frozen Phase 1 source changed: " + module)
        _REFERENCE_VERIFIED = True
    if _REFERENCE_PACKAGE not in sys.modules:
        package = ModuleType(_REFERENCE_PACKAGE)
        package.__package__ = _REFERENCE_PACKAGE
        package.__path__ = [str(_REFERENCE_ROOT)]
        package.__spec__ = importlib.machinery.ModuleSpec(
            _REFERENCE_PACKAGE, loader=None, is_package=True
        )
        sys.modules[_REFERENCE_PACKAGE] = package
    package = sys.modules[_REFERENCE_PACKAGE]
    _require(list(getattr(package, "__path__", [])) == [str(_REFERENCE_ROOT)],
             "Private reference package was replaced")
    if name not in _REFERENCE_MODULES:
        module = importlib.import_module(_REFERENCE_PACKAGE + "." + name)
        _require(Path(module.__file__).resolve() == (_REFERENCE_ROOT / (name + ".py")).resolve(),
                 "Reference module resolved outside the frozen source tree")
        _REFERENCE_MODULES[name] = module
    return _REFERENCE_MODULES[name]


class V2FeatureStore:
    """Read the new, fully audited 512/96-pair dataset and its frozen features.

    dataset_root is the directory containing manifest.json and observations/.
    features_root is config['output']/features. Feature NPZs contain only the
    float32 array 'features'; index.json records each path, SHA256 and frame
    count. The normalization provenance must enumerate exactly the train IDs.
    """

    def __init__(self, config, dataset_root, features_root, manifest):
        self.c = config
        self.dataset_root = Path(dataset_root).resolve()
        self.features_root = Path(features_root).resolve()
        self.manifest = copy.deepcopy(manifest)
        _require(self.features_root == (Path(config["output"]) / "features").resolve(),
                 "fit_basis requires normalization at output/features")
        _require(config["data"]["seed"] == 20261030
                 and config["data"]["tasks"] == ["motion"]
                 and config["data"]["pairs"] == {"train": 512, "validation": 96}
                 and config["data"]["recent"] == 4
                 and config["data"]["budget"] == 2
                 and config["data"]["horizons"] == [3, 6],
                 "Only the registered Motion v2 train/validation scope is permitted")
        manifest_path = self.dataset_root / "manifest.json"
        with manifest_path.open(encoding="utf-8") as source:
            _require(json.load(source) == self.manifest, "Manifest argument differs from disk")
        _require(self.manifest.get("mode") == "train_validation_only"
                 and self.manifest.get("generator_version") == "motion_v2.1"
                 and self.manifest.get("config_hash") == _digest(config["data"]),
                 "Wrong generator/config provenance")
        rows = self.manifest["records"]
        _require(len(rows) == 1216, "Expected exactly 1216 train/validation episodes")
        self.records = {}
        for rec in rows:
            _require(set(rec) == _RECORD_KEYS, "Unexpected public manifest fields")
            _require(rec["split"] in ("train", "validation") and rec["task"] == "motion",
                     "Held-out or unrelated episode is forbidden")
            _require(all(re.fullmatch(r"[0-9a-f]{24}", rec[key]) for key in ("id", "pair", "sibling")),
                     "Malformed opaque identity")
            _require(rec["id"] not in self.records, "Duplicate episode ID")
            _require(type(rec["query"]) is int and type(rec["frames"]) is int
                     and rec["frames"] == rec["query"] + 7,
                     "Invalid query/target frame boundary")
            self.records[rec["id"]] = rec
        for split, expected in (("train", 1024), ("validation", 192)):
            selected = [rec for rec in rows if rec["split"] == split]
            _require(len(selected) == expected
                     and len({rec["pair"] for rec in selected}) == expected // 2,
                     "Incomplete registered split")
        for left, right in zip(rows[::2], rows[1::2]):
            _require(left["sibling"] == right["id"] and right["sibling"] == left["id"]
                     and left["pair"] == right["pair"] and left["split"] == right["split"]
                     and left["query"] == right["query"] and left["frames"] == right["frames"],
                     "fit_basis requires adjacent complete sibling pairs")
        with (self.features_root / "index.json").open(encoding="utf-8") as source:
            self.ix = json.load(source)
        _require(self.ix.get("feature_dim") == 5120
                 and self.ix.get("manifest_sha256") == _sha256(manifest_path)
                 and isinstance(self.ix.get("signature"), str)
                 and re.fullmatch(r"[0-9a-f]{64}", self.ix["signature"])
                 and set(self.ix["episodes"]) == set(self.records),
                 "Feature cache is incomplete or belongs to another dataset")
        with np.load(self.features_root / "normalization.npz", allow_pickle=False) as archive:
            self.norm = {key: archive[key] for key in archive.files}
        _require({"mean", "std", "train_ids", "signature", "floor"} <= set(self.norm),
                 "Normalization train-only provenance is missing")
        for key in ("mean", "std"):
            _require(self.norm[key].dtype == np.float32 and self.norm[key].shape == (5120,)
                     and np.isfinite(self.norm[key]).all(), "Invalid normalization " + key)
        floor = float(config["normalization"]["std_floor"])
        _require(str(np.asarray(self.norm["signature"]).item()) == self.ix["signature"]
                 and np.asarray(self.norm["floor"]).shape == ()
                 and np.isclose(float(self.norm["floor"]), floor, atol=1e-8, rtol=0)
                 and np.all(self.norm["std"] >= floor), "Normalization provenance/floor differs")
        train_ids = {rec["id"] for rec in rows if rec["split"] == "train"}
        _require(self.norm["train_ids"].ndim == 1 and len(self.norm["train_ids"]) == 1024
                 and set(self.norm["train_ids"].tolist()) == train_ids,
                 "Normalization must fit all and only train episodes")
        self.bases = {}
        self.lru = {}
        self.projected = {}
        self.max_cached = len(rows)
        self._actions = {}
        self._selection = {}
        self._file_checks = {}
        self.training = False
        self.train_dropout = 0.25
        self._verified_file(manifest_path, self.ix["manifest_sha256"])
        for path in (self.features_root / "index.json", self.features_root / "normalization.npz"):
            self._verified_file(path, _sha256(path))

    def _record(self, rec):
        _require(isinstance(rec, dict) and rec.get("id") in self.records,
                 "Unknown or held-out record")
        canonical = self.records[rec["id"]]
        _require(rec == canonical, "Caller changed public record metadata")
        return canonical

    def _verified_file(self, path, sha):
        path = Path(path)
        stat = path.stat()
        identity = (stat.st_size, stat.st_mtime_ns, stat.st_ino, sha)
        if path not in self._file_checks:
            _require(re.fullmatch(r"[0-9a-f]{64}", sha) and _sha256(path) == sha,
                     "Artifact checksum mismatch: " + str(path))
            self._file_checks[path] = identity
        else:
            _require(self._file_checks[path] == identity, "Artifact changed after verification")
        return path

    def _episode_path(self, rec, kind):
        rec = self._record(rec)
        directory = "observations" if kind == "obs" else "evaluation_only"
        expected = Path(directory) / rec["split"] / (rec["id"] + ".npz")
        _require(Path(rec[kind]) == expected, "Episode path is outside the permitted layout")
        path = (self.dataset_root / expected).resolve()
        _require(path.is_relative_to(self.dataset_root), "Episode path escaped dataset root")
        return self._verified_file(path, rec[kind + "_sha256"])

    def read_evaluation(self, rec, keys):
        """Explicit evaluator-only access, never called for labels or velocities.

        Decoder fitting/evaluation can request states. The predictor sampler
        requests only visibility and timing to select privileged indices.
        """
        keys = tuple(keys)
        _require(bool(keys) and len(set(keys)) == len(keys) and set(keys) <= _EVALUATION_KEYS,
                 "Evaluator keys must be explicit and within the allowlist")
        path = self._episode_path(rec, "eval")
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in keys}

    def _selection_times(self, rec):
        rec = self._record(rec)
        if rec["id"] not in self._selection:
            meta = self.read_evaluation(rec, keys=("visible", "t_c", "end"))
            visible = meta["visible"]
            _require(visible.dtype == np.bool_ and visible.shape == (rec["frames"],)
                     and meta["t_c"].shape == () and meta["end"].shape == (),
                     "Invalid privileged selection metadata")
            tc, end = int(meta["t_c"]), int(meta["end"])
            stop = rec["query"] - 4 + 1
            _require(0 < tc < end < stop and bool(visible[end]), "Invalid legal evidence interval")
            old = np.flatnonzero(visible[:tc]).astype(np.int64)
            post = (tc + np.flatnonzero(visible[tc:end + 1])).astype(np.int64)
            _require(len(old) >= 2 and len(post) >= 2 and post[-1] == end,
                     "Generator audit visibility constraints no longer hold")
            self._selection[rec["id"]] = {
                "b": np.array([post[0], end], dtype=np.int64),
                "c": np.array([old[0], old[-1]], dtype=np.int64),
                "d": np.array([old[-1], end], dtype=np.int64),
            }
        return self._selection[rec["id"]]

    def features(self, rec):
        """Normalized feature sequence; no observation RGB is decompressed."""
        rec = self._record(rec)
        if rec["id"] not in self.lru:
            entry = self.ix["episodes"][rec["id"]]
            expected = Path("episodes") / (rec["id"] + ".npz")
            _require(Path(entry["path"]) == expected and entry["frames"] == rec["frames"],
                     "Feature index entry mismatch")
            path = (self.features_root / expected).resolve()
            _require(path.is_relative_to(self.features_root), "Feature path escaped its root")
            self._verified_file(path, entry["sha256"])
            with np.load(path, allow_pickle=False) as archive:
                _require(set(archive.files) == {"features"}, "Unexpected feature archive fields")
                z = archive["features"]
            _require(z.dtype == np.float32 and z.shape == (rec["frames"], 5120)
                     and np.isfinite(z).all(), "Invalid full-dimensional frozen features")
            normalized = ((z - self.norm["mean"]) / self.norm["std"]).astype(np.float32)
            _require(np.isfinite(normalized).all(), "Nonfinite normalized features")
            normalized.flags.writeable = False
            self.lru[rec["id"]] = normalized
        return self.lru[rec["id"]]

    def projected_features(self, rec):
        rec = self._record(rec)
        _require(rec["task"] in self.bases, "Train-only visual basis is not installed")
        basis = self.bases[rec["task"]]
        key = (rec["id"], id(basis["input_mean"]), id(basis["projection"]))
        if key not in self.projected:
            projected = load_reference("model").project(self.features(rec), basis)
            _require(np.isfinite(projected).all(), "Nonfinite projected features")
            projected.flags.writeable = False
            self.projected[key] = projected
        return self.projected[key]

    def _observed_actions(self, rec):
        rec = self._record(rec)
        if rec["id"] not in self._actions:
            path = self._episode_path(rec, "obs")
            with np.load(path, allow_pickle=False) as archive:
                _require(set(archive.files) == {"rgb", "actions", "query"},
                         "Unexpected observation fields")
                # Listing the RGB member does not read/decompress its contents.
                actions, query = archive["actions"], archive["query"]
            _require(query.shape == () and int(query) == rec["query"]
                     and actions.dtype == np.float32
                     and actions.shape == (rec["frames"] - 1, 2)
                     and np.isfinite(actions).all() and np.all(np.abs(actions) <= 1),
                     "Invalid observed action/query arrays")
            actions.flags.writeable = False
            self._actions[rec["id"]] = actions
        return self._actions[rec["id"]]

    def sample(self, rec, condition, rng=None, dropout=0):
        """Return (observable model tensors, separate feature target, public info).

        Whole-memory dropout is training-only and only applies when the caller
        supplied a training RNG. The original train_epoch does not pass its
        dropout argument, so training=True uses train_dropout in that case.
        Identically seeded sibling RNGs make exactly the same selection/draw.
        """
        rec = self._record(rec)
        _require(condition in _CONDITIONS, "Unknown condition; legacy wrong_evidence is not supported")
        canonical = _CONDITIONS[condition]
        supplied_rng = rng is not None
        if self.training:
            _require(rec["split"] == "train", "Training sampler cannot consume validation rows")
        if rng is None:
            seed = int(_digest((self.c["evaluation"]["random_seed"], rec["pair"], canonical))[:16], 16) % (2**32)
            rng = np.random.default_rng(seed)
        q, recent = rec["query"], 4
        stop = q - recent + 1
        valid = np.arange(stop, dtype=np.int64)
        if canonical == "recent_only":
            indices = np.empty(0, dtype=np.int64)
        elif canonical == "full_history":
            indices = valid.copy()
        elif canonical == "random_small_budget":
            indices = np.sort(rng.choice(valid, size=2, replace=False)).astype(np.int64)
        else:
            group = {"privileged_small_budget": "b", "stale_two": "c", "end_plus_stale": "d"}[canonical]
            indices = self._selection_times(rec)[group].copy()
        probability = float(dropout) if dropout else float(self.train_dropout)
        _require(np.isfinite(probability) and 0 <= probability <= 1, "Invalid memory dropout probability")
        if self.training and supplied_rng and len(indices) and rng.random() < probability:
            indices = indices[:0]
        _require(indices.dtype == np.int64 and np.all((indices >= 0) & (indices < stop))
                 and np.all(np.diff(indices) > 0), "Illegal or duplicate history index")
        times = np.concatenate([np.arange(q - recent + 1, q + 1), indices])
        z = self.features(rec)
        actions = self._observed_actions(rec)
        prefix = np.concatenate([np.zeros((1, 2), np.float32), np.cumsum(actions[:q], axis=0)])
        sample = {
            "z": z[times].copy(),
            "age": (q - times).astype(np.float32),
            "type": np.concatenate([np.zeros(recent, np.int64), np.ones(len(indices), np.int64)]),
            "past_actions": prefix[q] - prefix[times],
            "actions": actions[q:q + 6].copy(),
            "current": z[q].copy(),
        }
        if rec["task"] in self.bases:
            projected = self.projected_features(rec)
            sample["projected_z"] = projected[times].copy()
            sample["projected_current"] = projected[q].copy()
        allowed = _MODEL_KEYS | (_PROJECTED_KEYS if rec["task"] in self.bases else set())
        _require(set(sample) == allowed and all(np.isfinite(value).all() for value in sample.values()),
                 "Unexpected or nonfinite model tensor")
        # Targets stay outside the model-input dictionary. No future RGB is read.
        target = z[q + np.asarray([3, 6])].copy()
        info = {"indices": indices.copy(), "history_count": len(indices)}
        return sample, target, info

    def verify_boundaries(self):
        """Cheap check that already-hashed source artifacts have not changed."""
        for path, expected in self._file_checks.items():
            stat = path.stat()
            _require((stat.st_size, stat.st_mtime_ns, stat.st_ino, expected[3]) == expected,
                     "Input artifact changed during the run")


def assert_a_c_d_inputs_identical(store, pairs):
    """Audit complete sibling inputs without requiring b features to differ."""
    checked, by_split = 0, {"train": 0, "validation": 0}
    was_training = store.training
    store.training = False
    try:
        for pair in pairs:
            _require(len(pair) == 2, "Input audit needs complete pairs")
            left, right = (store._record(rec) for rec in pair)
            _require(left["pair"] == right["pair"] and left["split"] == right["split"]
                     and left["sibling"] == right["id"] and right["sibling"] == left["id"],
                     "Input audit received unrelated records")
            for condition in ("a", "c", "d"):
                a, _, a_info = store.sample(left, condition)
                b, _, b_info = store.sample(right, condition)
                _require(set(a) == set(b), "Sibling model tensor fields differ")
                for key in a:
                    _require(a[key].dtype == b[key].dtype and np.array_equal(a[key], b[key]),
                             "Sibling inputs differ for condition " + condition + ": " + key)
                _require(a_info["history_count"] == b_info["history_count"] == (0 if condition == "a" else 2)
                         and np.array_equal(a_info["indices"], b_info["indices"]),
                         "Sibling history selection differs")
            b_indices = []
            for rec in (left, right):
                _, _, info = store.sample(rec, "b")
                selected = info["indices"]
                _require(info["history_count"] == 2 and np.array_equal(selected, store._selection_times(rec)["b"])
                         and selected[0] < selected[1] < rec["query"] - 3,
                         "Condition b did not select the registered legal pair")
                b_indices.append(selected)
            _require(np.array_equal(*b_indices), "Condition b timestamps differ between siblings")
            checked += 1
            by_split[left["split"]] += 1
        _require(checked > 0, "Input audit received no pairs")
        store.verify_boundaries()
    finally:
        store.training = was_training
    return {
        "passed": True, "pairs": checked, "pairs_by_split": by_split,
        "a_c_d_exact_input_comparisons": 3 * checked,
        "b_legal_two_frame_checks": 2 * checked,
        "b_feature_inequality_required": False,
        "hidden_state_in_model_inputs": False,
    }
