"""Software fixtures only, seed6301/smoke_pixels; no research/fresh access."""
import copy
from pathlib import Path
import numpy as np
import pytest
from phase1 import data, cache
from phase1.config import DEFAULT
from phase1.inputs import FeatureStore
from phase1.utils import file_hash
from staged_data import (generate_train_validation, encode_train_validation,
    fit_stage_normalization, StageFeatureStore, stage_subset_lock,
    append_locked_tests, verify_training_subset, verify_permit)


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    c = copy.deepcopy(DEFAULT)
    c.update(output=str(tmp_path_factory.mktemp("stage")), device="cpu")
    c["data"]["seed"] = 6301
    c["data"]["pairs"] = {s: 2 for s in c["data"]["pairs"]}
    c["evaluation"]["bootstrap"] = 100
    generate_train_validation(c, fixture=True)
    encode_train_validation(c, fixture=True)
    fit_stage_normalization(c)
    original = copy.deepcopy(c)
    original["output"] = str(tmp_path_factory.mktemp("full"))
    data.generate(original)
    data.validate_data(original)
    cache.encode(original)
    cache.fit_normalization(original)
    return c, original


def test_exact_original_data_features_normalization_and_samples(staged):
    c, original = staged
    a, b = StageFeatureStore(c), FeatureStore(original)
    for rec in a.manifest["records"]:
        other = b.records[rec["id"]]
        for meta in (False, True):
            left, right = data.load_episode(c, rec, meta), data.load_episode(original, other, meta)
            assert left.keys() == right.keys()
            assert all(np.array_equal(left[k], right[k]) for k in left)
        assert np.array_equal(a.features(rec), b.features(other))
        for condition in ("recent_only", "full_history", "privileged_small_budget", "random_small_budget", "wrong_evidence"):
            ra, rb = np.random.default_rng(91), np.random.default_rng(91)
            sa, sb = a.sample(rec, condition, ra), b.sample(other, condition, rb)
            assert ra.bit_generator.state == rb.bit_generator.state
            for left, right in ((sa[0], sb[0]), (sa[2], sb[2])):
                assert left.keys() == right.keys()
                assert all(np.array_equal(left[k], right[k]) for k in left)
            assert np.array_equal(sa[1], sb[1])
    for k in a.norm:
        assert np.array_equal(a.norm[k], b.norm[k])


def test_incomplete_stage_cannot_open_original_full_store(staged):
    c, _ = staged
    assert not (Path(c["output"])/"dataset/split_audit.json").exists()
    with pytest.raises(FileNotFoundError):
        FeatureStore(c)


def test_append_preserves_training_and_full_audits(staged):
    c, original = staged
    subset = stage_subset_lock(c)
    norm_sha = file_hash(Path(c["output"])/"features/normalization.npz")
    append_locked_tests(c, subset, fixture=True)
    cache.encode(c)
    assert verify_training_subset(c, subset)
    assert file_hash(Path(c["output"])/"features/normalization.npz") == norm_sha
    a, b = FeatureStore(c), FeatureStore(original)
    for rec in a.manifest["records"]:
        assert np.array_equal(a.features(rec), b.features(b.records[rec["id"]]))
    with pytest.raises(ValueError):
        append_locked_tests(c, subset, fixture=True)


def test_fixture_cannot_open_fresh_or_research(tmp_path):
    c = copy.deepcopy(DEFAULT)
    c["output"] = str(tmp_path)
    c["data"]["seed"] = 9137
    with pytest.raises(ValueError):
        verify_permit(c, None, fixture=True)
    c["data"]["seed"] = 6301
    c["mode"] = "confirm"
    with pytest.raises(ValueError):
        verify_permit(c, None, fixture=True)
