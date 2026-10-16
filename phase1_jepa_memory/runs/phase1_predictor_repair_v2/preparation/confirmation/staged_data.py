"""Preparation only: strict canonical staged data/cache boundary.

No fresh data is accessed by import. Production entry points require a separately
frozen permit proving both development tasks and their independent audit passed.
Original scientific functions are locally bound to explicit stage dependencies;
their code objects and the original module globals are never modified.
"""
import copy
from collections import OrderedDict
from pathlib import Path
from types import FunctionType
import shutil

import numpy as np

from phase1 import cache, data
from phase1.config import SPLITS
from phase1.inputs import FeatureStore
from phase1.exact_runtime_cache import CachedFeatureStore
from phase1.utils import digest, file_hash, read_json, write_json

TRAIN_SPLITS = ("train", "validation")


def bind(function, **dependencies):
    namespace = dict(function.__globals__)
    namespace.update(dependencies)
    result = FunctionType(function.__code__, namespace, function.__name__,
                          function.__defaults__, function.__closure__)
    result.__kwdefaults__ = copy.deepcopy(function.__kwdefaults__)
    return result


def expected_rows(c, splits):
    """Opaque IDs/order exactly as original data.generate, without simulation."""
    rows = []
    for task in c["data"]["tasks"]:
        for split in SPLITS:
            if split not in splits:
                continue
            for i in range(c["data"]["pairs"][split]):
                pid = digest([c["data"]["seed"], task, split, i])[:24]
                ids = [digest([pid, j, "opaque-id"])[:24] for j in range(2)]
                rows.extend((eid, pid, ids[1-j], task, split, i)
                            for j, eid in enumerate(ids))
    return rows


def check_scope(c, m, splits):
    if tuple(splits) not in (TRAIN_SPLITS, tuple(SPLITS)):
        raise ValueError("Only complete train/validation or complete final scope")
    if m["signature"] != data.dataset_signature(c):
        raise ValueError("Canonical full configuration signature mismatch")
    actual = [(r["id"], r["pair"], r["sibling"], r["task"], r["split"],
               r["scene_index"]) for r in m["records"]]
    if actual != expected_rows(c, splits):
        raise ValueError("Missing/extra/reordered canonical pair or split")


def verify_permit(c, permit, *, models_locked=False, fixture=False):
    if fixture:
        if c["mode"] != "smoke" or c["data"]["seed"] == 9137:
            raise ValueError("Fixtures may not access research/fresh data")
        return
    if (c["mode"] != "confirm" or c["data"]["seed"] != 9137
            or c["data"]["pairs"] != {"train": 512, "validation": 96,
                "test_id": 256, "test_long_gap": 256, "test_distractor_shift": 256}
            or "experiment" in c):
        raise ValueError("Canonical original full confirmation config required")
    p = read_json(permit)
    if (p.get("status") != "FROZEN_CONFIRMATION_AUTHORIZED"
            or p.get("development_tasks") != {"gate": "SUPPORTED", "motion": "SUPPORTED"}
            or p.get("independent_development_audit") != "PASS"
            or p.get("GPU_cap_seconds") != 28800
            or p.get("canonical_config_digest") != digest(c)
            or not p.get("evidence_files") or not p.get("source_files")):
        raise ValueError("Complete frozen development/confirmation permit absent")
    for name, sha in {**p["evidence_files"], **p["source_files"]}.items():
        if file_hash(name) != sha:
            raise ValueError("Frozen evidence/source changed: " + name)
    if models_locked:
        lock = read_json(p["all_models_lock"])
        if (lock.get("status") != "ALL_TWELVE_MODELS_LOCKED"
                or len(lock.get("models", {})) != 12
                or lock.get("required_epochs") != 120
                or lock.get("test_opened") is not False):
            raise ValueError("All12 full models must be locked before tests")
        for name, sha in lock["files"].items():
            if file_hash(name) != sha:
                raise ValueError("Locked model/transform changed")


def stage_audit_path(c):
    return Path(c["output"])/"dataset/train_validation_stage_audit.json"


def audit_stage(c):
    m = data.manifest(c)
    check_scope(c, m, TRAIN_SPLITS)
    def write_stage(path, value):
        if Path(path).name != "split_audit.json":
            raise ValueError("Unexpected structural audit write")
        write_json(stage_audit_path(c), {**value, "stage": "TRAIN_VALIDATION_ONLY",
                   "canonical_final_complete": False,
                   "manifest_sha256": file_hash(Path(c["output"])/"dataset/manifest.json")})
    # Exact original assertions and deterministic regeneration, restricted only
    # in the final split-summary loop. No original full audit is fabricated.
    return bind(data.validate_data, SPLITS=TRAIN_SPLITS, write_json=write_stage)(c)


def require_stage(c):
    m = data.manifest(c)
    check_scope(c, m, TRAIN_SPLITS)
    a = read_json(stage_audit_path(c))
    if (not a["passed"] or a["signature"] != m["signature"]
            or a.get("canonical_final_complete") is not False
            or a["manifest_sha256"] != file_hash(Path(c["output"])/"dataset/manifest.json")):
        raise ValueError("Complete current train/validation stage audit required")
    for rec in m["records"]:
        for kind in ("obs", "eval"):
            if file_hash(Path(c["output"])/"dataset"/rec[kind]) != rec[kind+"_sha256"]:
                raise ValueError("Stage source checksum changed")
    return m


def generate_train_validation(c, permit=None, *, fixture=False):
    verify_permit(c, permit, fixture=fixture)
    if (Path(c["output"])/"dataset/manifest.json").exists():
        raise ValueError("No implicit generation retry/overwrite")
    bind(data.generate, SPLITS=TRAIN_SPLITS, validate_data=audit_stage)(c)
    audit_stage(c)
    return require_stage(c)


def stage_load_index(c, complete=True):
    m = require_stage(c)
    p = Path(c["output"])/"features/index.json"
    ix = read_json(p)
    if ix["signature"] != cache.cache_signature(c, m):
        raise ValueError("Stage cache signature changed")
    if complete:
        if set(ix["episodes"]) != {r["id"] for r in m["records"]}:
            raise ValueError("Stage cache incomplete or includes heldout")
        a = read_json(p.parent/"train_validation_feature_audit.json")
        if not a["passed"] or a["index_sha256"] != file_hash(p):
            raise ValueError("Current stage feature-pair proof required")
    return ix


def audit_stage_features(c, m, ix):
    check_scope(c, m, TRAIN_SPLITS)
    def write_stage(path, value):
        if Path(path).name != "pair_feature_audit.json":
            raise ValueError("Unexpected feature audit write")
        write_json(Path(path).with_name("train_validation_feature_audit.json"),
                   {**value, "stage": "TRAIN_VALIDATION_ONLY",
                    "canonical_final_complete": False})
    return bind(cache.audit_features, write_json=write_stage)(c, m, ix)


def encode_train_validation(c, permit=None, *, fixture=False):
    verify_permit(c, permit, fixture=fixture)
    if not fixture and (c["device"] != "cuda" or c["encoder"]["batch_size"] != 2):
        raise ValueError("Original actualGPU batch2 encoder required")
    return bind(cache.encode, require_audit=require_stage, load_index=stage_load_index,
                audit_features=audit_stage_features)(c)


def stage_normalization(c):
    return bind(cache.load_normalization, load_index=stage_load_index,
                require_audit=require_stage)(c)


def fit_stage_normalization(c):
    # Original float64 summation loop/order/floor and train-ID provenance exact.
    return bind(cache.fit_normalization, require_audit=require_stage,
                load_index=stage_load_index, load_normalization=stage_normalization)(c)


class StageFeatureStore(FeatureStore):
    """Original sampler/projection methods, explicit strictly checked constructor."""
    def __init__(self, c, max_cached=1216):
        self.c, self.manifest, self.ix = c, require_stage(c), stage_load_index(c)
        self.norm = stage_normalization(c)
        self.records = {r["id"]: r for r in self.manifest["records"]}
        self.lru, self.max_cached = OrderedDict(), max_cached
        self.bases, self.projected = {}, OrderedDict()


class StageCachedFeatureStore(CachedFeatureStore, StageFeatureStore):
    """Original exact metadata/validation cache with explicit stage constructor.

The cooperative initializer calls StageFeatureStore before FeatureStore. No
global full-cache audit is disabled and CachedFeatureStore.sample stays exact.
The original per-task1216 metadata count remains enforced.
"""
    def verify_all_inputs(self):
        method = bind(CachedFeatureStore.verify_all_inputs, FeatureStore=StageFeatureStore)
        return method(self)

    def verify_validation_inputs(self, pairs):
        method = bind(CachedFeatureStore.verify_validation_inputs, FeatureStore=StageFeatureStore)
        return method(self, pairs)


def stage_subset_lock(c):
    """Immutable train/validation feature provenance to recheck after test append."""
    m, ix = require_stage(c), stage_load_index(c)
    return {"dataset_signature": m["signature"], "cache_signature": ix["signature"],
            "records": copy.deepcopy(m["records"]), "episodes": copy.deepcopy(ix["episodes"]),
            "normalization_sha256": file_hash(Path(c["output"])/"features/normalization.npz")}


def append_locked_tests(c, subset, permit=None, *, fixture=False):
    verify_permit(c, permit, models_locked=True, fixture=fixture)
    if stage_subset_lock(c) != subset:
        raise ValueError("Training records/features/transforms changed before test opening")
    root = Path(c["output"])
    attempt = root/"confirmation_test_generation_attempt.json"
    if attempt.exists():
        raise ValueError("Unknown/prior test generation attempt; no implicit retry")
    write_json(attempt, {"status": "STARTED", "training_subset_digest": digest(subset)})
    staging = root/"locked_test_generation"
    if staging.exists():
        raise ValueError("Refuse stale test staging directory")
    test_c = copy.deepcopy(c)
    test_c["output"] = str(staging)
    tests = tuple(s for s in SPLITS if s not in TRAIN_SPLITS)
    new = bind(data.generate, SPLITS=tests)(test_c)
    old = require_stage(c)
    records = {r["id"]: r for r in old["records"]+new["records"]}
    for kind in ("observations", "evaluation_only"):
        for split in tests:
            destination = root/"dataset"/kind/split
            if destination.exists():
                raise ValueError("Heldout destination already materialized")
            shutil.move(str(staging/"dataset"/kind/split), destination)
    complete = {**old, "records": [records[e[0]] for e in expected_rows(c, SPLITS)]}
    check_scope(c, complete, tuple(SPLITS))
    write_json(root/"dataset/manifest.json", complete)
    audit = data.validate_data(c)
    verify_training_subset(c, subset)
    write_json(root/"confirmation_test_generation_complete.json",
               {"passed": True, "training_subset_digest": digest(subset),
                "final_manifest_sha256": file_hash(root/"dataset/manifest.json")})
    return audit


def verify_training_subset(c, subset):
    m = data.manifest(c)
    ix = read_json(Path(c["output"])/"features/index.json")
    if (m["signature"] != subset["dataset_signature"]
            or ix["signature"] != subset["cache_signature"]
            or [r for r in m["records"] if r["split"] in TRAIN_SPLITS] != subset["records"]
            or {eid: ix["episodes"][eid] for eid in subset["episodes"]} != subset["episodes"]
            or file_hash(Path(c["output"])/"features/normalization.npz") != subset["normalization_sha256"]):
        raise ValueError("Frozen training subset changed after test append")
    for rec in subset["records"]:
        for kind in ("obs", "eval"):
            if file_hash(Path(c["output"])/"dataset"/rec[kind]) != rec[kind+"_sha256"]:
                raise ValueError("Training source changed after test append")
        cache.load_feature(c, rec, ix)
    return True
