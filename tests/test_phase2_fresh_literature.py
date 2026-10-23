"""Fresh-data provenance and frozen evaluation contracts; no GPU/model run."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.phase2 import encode_fresh_gate_diagnostic as encoding
from experiments.phase2 import run_fresh_literature_diagnostic as fresh
from experiments.phase2.literature_baselines import SalienceStudent

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("condition", list(encoding.CONDITIONS))
def test_fixed_fresh_generation_is_paired_and_old_test_names_are_conditions_only(condition):
    envs, hashes = encoding.verify_sources(ROOT)
    assert hashes == encoding.EXPECTED_SOURCE_SHA256
    records, episodes = encoding.make_pair(envs, condition, 0)
    a, b = records
    q = a["query"]
    assert a["split"] == b["split"] == "fresh_development"
    assert a["generation_condition"] == encoding.CONDITIONS[condition]
    assert a["sibling"] == b["id"] and a["pair"] == b["pair"]
    np.testing.assert_array_equal(episodes[0]["rgb"][q-3:q+1], episodes[1]["rgb"][q-3:q+1])
    np.testing.assert_array_equal(episodes[0]["actions"], episodes[1]["actions"])
    assert not np.array_equal(episodes[0]["rgb"][episodes[0]["evidence"]], episodes[1]["rgb"][episodes[1]["evidence"]])
    if condition == "long_gap":
        assert 20 <= a["gap"] <= 32
    else:
        assert 5 <= a["gap"] <= 10
    again, regenerated = encoding.make_pair(envs, condition, 0)
    assert again == records
    np.testing.assert_array_equal(regenerated[0]["rgb"], episodes[0]["rgb"])


def test_embedded_fresh_config_preserves_inherited_dynamics_except_new_seed_and_gate_scope():
    reference = json.loads((ROOT / "results/reference/gate/resolved_config.json").read_text())
    for key in ("image_size", "recent", "horizons", "budget", "gap", "long_gap", "cue_start", "cue_duration", "motion_dt"):
        assert encoding.DATA_CONFIG[key] == reference["data"][key]
    assert encoding.DATA_CONFIG["seed"] == 20261022
    assert encoding.DATA_CONFIG["tasks"] == ["gate"]
    for key in ("repository", "revision", "pooling", "dtype"):
        assert encoding.ENCODER_CONFIG[key] == reference["encoder"][key]
    assert encoding.ENCODER_CONFIG["local_files_only"] is True


def test_new_normalization_is_rejected_and_cpu_cannot_claim_a100(tmp_path, monkeypatch):
    path = tmp_path / "normalization.npz"
    np.savez(path, mean=np.zeros(5120, np.float32), std=np.ones(5120, np.float32))
    with pytest.raises(ValueError, match="OLD train-only"):
        encoding.load_old_normalization(path)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="Slurm"):
        encoding.verify_a100()


def fake_fresh_records():
    records = []
    for condition_number, condition in enumerate(encoding.CONDITIONS):
        for index in range(32):
            pair = encoding.digest([condition, index])[:24]
            ids = [encoding.digest([pair, j])[:24] for j in (0, 1)]
            q = 40 if condition == "long_gap" else 15
            for j in (0, 1):
                records.append({"id": ids[j], "pair": pair, "sibling": ids[1-j],
                    "split": "fresh_development", "task": "gate", "condition": condition,
                    "generation_condition": encoding.CONDITIONS[condition],
                    "seed": condition_number * 32 + index, "scene_index": index,
                    "query": q, "frames": q + 7, "rgb_indices": [0] * (q + 7),
                    "feature_path": f"features/{ids[j]}.npz", "eval_path": f"evaluation_only/{ids[j]}.npz"})
    return records


def test_fresh_record_contract_rejects_subsets_split_leak_and_scene_overlap():
    records = fake_fresh_records()
    assert len(fresh.validate_fresh_records(records)) == 192
    with pytest.raises(ValueError, match="192 fresh"):
        fresh.validate_fresh_records(records[:-2])
    bad = deepcopy(records)
    bad[0]["split"] = "test_id"
    with pytest.raises(ValueError, match="identity/split"):
        fresh.validate_fresh_records(bad)
    bad = deepcopy(records)
    bad[2]["seed"] = bad[0]["seed"]
    with pytest.raises(ValueError, match="share a scene seed"):
        fresh.validate_fresh_records(bad)
    bad = deepcopy(records)
    bad[0]["rgb_indices"][bad[0]["query"]] = 3
    with pytest.raises(ValueError, match="recent RGB"):
        fresh.validate_fresh_records(bad)


def test_fresh_normalization_is_recomputed_with_old_statistics(tmp_path):
    raw = np.arange(10240, dtype=np.float32).reshape(2, 5120)
    old_norm = {"mean": np.ones(5120, np.float32), "std": np.full(5120, 2., np.float32)}
    normalized = (raw-old_norm["mean"])/old_norm["std"]
    path = tmp_path / "unique_features.npz"
    np.savez_compressed(path, raw=raw, normalized=normalized)
    dataset = fresh.FreshGateDataset.__new__(fresh.FreshGateDataset)
    dataset.root, dataset.loaded, dataset.array_files_opened = tmp_path, {}, []
    dataset.manifest = {"unique_rgb_frames": 2, "artifact_sha256": {path.name: encoding.sha256(path)}}
    dataset.verify_old_normalization(old_norm)
    np.testing.assert_array_equal(dataset._normalized_bank, normalized)
    with pytest.raises(ValueError, match="normalization arithmetic"):
        dataset.verify_old_normalization({"mean": old_norm["mean"], "std": np.ones(5120, np.float32)})


def frozen_student_fixture(tmp_path):
    definition = "experiments/phase2/literature_baselines.py"
    protocol = {"created_utc": datetime(2026, 10, 21, tzinfo=timezone.utc).isoformat(),
                "scope": "existing_gate_development_only", "train_pairs": ["train-pair"],
                "student": {"seeds": [17, 23, 41], "epochs": 100, "hidden": 32, "lr": .001,
                            "selection": "fixed final epoch; no validation selection", "targets": "attention only"},
                "source_sha256": {definition: encoding.sha256(ROOT / definition)}}
    encoding.write_json(tmp_path / "protocol.json", protocol)
    protocol_sha = encoding.sha256(tmp_path / "protocol.json")
    encoding.write_json(tmp_path / "summary.json", {"status": "completed_development_diagnostic",
        "protocol_sha256": protocol_sha, "elapsed_seconds": 40., "test_arrays_read": 0, "utility_scorer_trained": False})
    for seed in (17, 23, 41):
        model = SalienceStudent(11, hidden=32)
        torch.save({"model": model.state_dict(), "input_dim": 11, "hidden": 32, "seed": seed,
                    "protocol_sha256": protocol_sha, "target_kind": "future_attention", "trained_utility_scorer": False},
                   tmp_path / f"pafu_seed_{seed}.pt")
    old = SimpleNamespace(records=[{"pair": "train-pair", "task": "gate", "split": "train"}],
                          config={"model": {"rank": 2}})
    return old


def test_fresh_student_load_only_accepts_frozen_old_training_groups_and_attention_targets(tmp_path):
    old = frozen_student_fixture(tmp_path)
    start = "2026-10-22T00:00:00+00:00"
    models, proof = fresh.load_frozen_students(tmp_path, old, start)
    assert set(models) == {17, 23, 41} and proof["fresh_training_calls"] == 0
    assert all(not model.training and all(not p.requires_grad for p in model.parameters()) for model in models.values())
    wrong = SimpleNamespace(records=[{"pair": "different", "task": "gate", "split": "train"}], config=old.config)
    with pytest.raises(ValueError, match="train-only"):
        fresh.load_frozen_students(tmp_path, wrong, start)
    with pytest.raises(ValueError, match="before fresh encoding"):
        fresh.load_frozen_students(tmp_path, old, "2026-10-21T00:00:01+00:00")
    path = tmp_path / "pafu_seed_17.pt"
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["target_kind"] = "deletion_loss"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="binding mismatch"):
        fresh.load_frozen_students(tmp_path, old, start)


def test_wrong_sibling_control_changes_only_historical_cue_not_targets(monkeypatch):
    captured = []

    class Oracle:
        def __init__(self, _episode_id, z, _actions, _q, _scores):
            self.z = z
            captured.append(z.copy())

        def evaluate(self, times):
            value = float(self.z[times[0], 0]) if times else 2.
            return np.array([value, value]), np.zeros((2, 2), np.float32)

    monkeypatch.setattr(fresh, "QueryOracle", Oracle)
    z = np.arange(24, dtype=np.float32).reshape(12, 2)
    sibling = z.copy()
    sibling[0] = 100.
    actions = np.zeros((11, 2), np.float32)
    oracle = Oracle("e", z, actions, 5, None)
    record = {"id": "e", "pair": "pair", "condition": "id", "query": 5}
    result, _ = fresh.history_control(record, z, actions, sibling, np.array([0]), oracle, None)
    np.testing.assert_array_equal(captured[-1][1:], z[1:])
    np.testing.assert_array_equal(captured[-1][0], sibling[0])
    assert z[0, 0] == 0. and result["wrong_minus_privileged"] == 100.
