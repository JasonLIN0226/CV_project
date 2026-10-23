"""Actual pixel-cache fixtures for the sample pipeline, never official I-JEPA."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.phase2 import prepare_controlled_data as data
from experiments.phase2 import prepare_encoded_samples as encoded


def test_encoder_config_is_pinned_separate_and_rebased_without_mutation(tmp_path):
    prepared = data.make_config(json.loads(data.REFERENCE.read_text()), tmp_path / "data")
    before = copy.deepcopy(prepared)
    actual = encoded.make_encoder_config(prepared, tmp_path / "moved-data", tmp_path / "model-cache", device="cuda")
    assert prepared == before
    assert actual["data"] == prepared["data"]
    assert actual["encoder"]["revision"] == encoded.OFFICIAL_REVISION
    assert actual["encoder"]["dtype"] == "float32" and actual["encoder"]["pooling"] == "grid2"
    assert actual["output"] == str((tmp_path / "moved-data").resolve())
    wrong = copy.deepcopy(prepared)
    wrong["encoder"]["backend"] = "smoke_pixels"
    with pytest.raises(ValueError, match="official pinned"):
        encoded.make_encoder_config(wrong, tmp_path, tmp_path, device="cpu")
    with pytest.raises(ValueError, match="positive integer"):
        encoded.make_encoder_config(prepared, tmp_path, tmp_path, device="cpu", batch_size=0)


@pytest.fixture(scope="module")
def pixel_cache(tmp_path_factory):
    # Exercise actual retained cache machinery without fetching any model.
    from phase1.cache import encode, fit_normalization
    import torch
    torch.set_num_threads(2)
    root = tmp_path_factory.mktemp("phase2-pixel-cache")
    preparation = data.run(root, seed=20261023, train_pairs=2, eval_pairs=2)
    original_config_hash = encoded.sha256(root / "resolved_config.json")
    config = json.loads((root / "resolved_config.json").read_text())
    config["mode"] = "smoke"
    config["encoder"]["backend"] = "smoke_pixels"
    config["encoder"]["smoke_size"] = 2
    first = encode(config, max_episodes=3)
    assert len(first["episodes"]) == 3
    prefix_hashes = {key: value["sha256"] for key, value in first["episodes"].items()}
    complete = encode(config)
    assert len(complete["episodes"]) == 40
    assert all(complete["episodes"][key]["sha256"] == value for key, value in prefix_hashes.items())
    norm = fit_normalization(config)
    return dict(root=root, config=config, index=complete, norm=norm,
                clean=json.loads((root / "observation_index.json").read_text()),
                roles=json.loads((root / "split_roles.json").read_text())["roles"],
                original_config_hash=original_config_hash, preparation=preparation)


def test_real_pixel_cache_train_normalization_and_sample_audit(pixel_cache, tmp_path):
    fixture = pixel_cache
    result = encoded.summarize_encoded_samples(
        fixture["config"], fixture["clean"], fixture["roles"], fixture["index"], fixture["norm"], tmp_path,
        epochs=2, base_seed=17, dropout=.25, expected_dimension=12)
    assert result["encoder_backend"] == "smoke_pixels"
    assert result["encoder_name"] == "smoke_pixels_NOT_JEPA"
    assert result["feature_episodes_loaded"] == 40
    assert result["sample_rows"] == 8 * 2 * 6 + 8 * 6
    assert result["normalization_train_episodes"] == 8
    assert result["normalization_train_only_verified"]
    assert result["normalization_moments_recomputed_exactly"]
    assert result["test_examples_generated"] == 0 and not result["predictor_training"]
    assert result["deterministic_validation_replays"] == 8 * 6
    rows = [json.loads(line) for line in (tmp_path / "sample_rows.jsonl").read_text().splitlines()]
    assert {row["split"] for row in rows} == {"train", "validation"}
    assert any(row["dropout_removed_entries"] > 0 for row in rows if row["split"] == "train")
    assert all(not row["dropout_event"] for row in rows if row["split"] == "validation")
    assert encoded.sha256(fixture["root"] / "resolved_config.json") == fixture["original_config_hash"]


def test_independent_train_moment_check_rejects_wrong_normalization(pixel_cache, tmp_path):
    fixture = pixel_cache
    norm = dict(fixture["norm"])
    norm["mean"] = norm["mean"].copy() + np.float32(.01)
    with pytest.raises(ValueError, match="moments"):
        encoded.summarize_encoded_samples(fixture["config"], fixture["clean"], fixture["roles"],
                                          fixture["index"], norm, tmp_path, expected_dimension=12)


def test_preparation_verification_accepts_relocation_but_rejects_stale_bytes(pixel_cache):
    fixture = pixel_cache
    # Runtime encoder fields are absent from the dataset signature, so the
    # fixture can verify prepared bytes while exercising the pixel cache.
    native, clean, roles = encoded.verify_prepared_inputs(fixture["root"], fixture["config"])
    assert native["signature"] == clean["dataset_signature"]
    assert roles["test_id"] == "development_smoke"
    proof = encoded.verify_retained_source()
    assert len(proof["files"]) == 6
    index_path = fixture["root"] / "observation_index.json"
    original = index_path.read_bytes()
    try:
        index_path.write_bytes(original + b"\n")
        with pytest.raises(ValueError, match="checksum differs"):
            encoded.verify_prepared_inputs(fixture["root"], fixture["config"])
    finally:
        index_path.write_bytes(original)


def test_invalid_sampling_options_fail_before_any_encode(tmp_path):
    for kwargs in ({"sample_epochs": 0}, {"base_seed": -1}, {"dropout": 1.1}):
        with pytest.raises(ValueError):
            encoded.run(tmp_path / "missing", tmp_path / "cache", tmp_path / "report", device="cpu", **kwargs)
