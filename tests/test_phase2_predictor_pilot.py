"""Fresh CPU optimizer-path checks on small synthetic vectors, not JEPA results."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.phase2 import train_predictor_pilot as pilot
from phase1.config import DEFAULT
from phase1.utils import write_npz


def toy_store(root):
    config = copy.deepcopy(DEFAULT)
    config.update(output=str(root), threads=1)
    config["model"].update(hidden=16, rank=4, dropout=0.)
    config["training"].update(epochs=2, seeds=[17], batch_size=4)
    (root / "features").mkdir(parents=True)
    write_npz(root / "features/normalization.npz", fixture_only=np.array(True))
    records, features = [], {}
    for split_number, split in enumerate(("train", "validation", "test_id")):
        for pair in range(2):
            group = f"{split}-{pair}"
            for sibling in range(2):
                episode = f"{group}-{sibling}"
                record = {"id": episode, "pair": group, "sibling": f"{group}-{1-sibling}",
                          "split": split, "task": "gate", "seed": split_number * 10 + pair,
                          "query": 12, "obs": f"observations/{episode}.npz"}
                records.append(record)
                rng = np.random.default_rng(split_number * 100 + pair * 2 + sibling)
                z = rng.normal(size=(20, 8)).astype(np.float32)
                z[13:] = z[12] + np.float32(sibling * .4)
                features[episode] = z
                write_npz(root / "dataset" / record["obs"], actions=np.zeros((19, 2), np.float32))
    loaded = []
    def read_features(record):
        if record["split"].startswith("test"):
            raise AssertionError("Test feature leakage")
        loaded.append(record["id"])
        return features[record["id"]]
    store = SimpleNamespace(manifest={"records": records}, ix={"signature": "synthetic_vectors_NOT_JEPA"},
                            norm={"mean": np.zeros(8, np.float32)}, features=read_features)
    return config, store, records, loaded


def test_cpu_optimizer_updates_frozen_selection_and_no_test_access(tmp_path):
    c, store, records, loaded = toy_store(tmp_path)
    result = pilot.train_task(c, store, "gate", records,
            {"train": "training", "validation": "validation", "test_id": "development_smoke"},
            tmp_path, epochs=2, seed=17, dropout=1., batch_size=4)
    assert result["parameters_changed"] and result["optimizer_steps"] == 12
    assert result["frozen_validation_replay_exact"] and result["test_examples_or_metrics"] == 0
    assert result["input_rank"] == result["output_rank"] == 4
    assert not any(name.startswith("test") for name in loaded)
    assert all(cell["dropout_events"] == cell["examples"] for cell in result["sampling_counts"].values())
    checkpoint = torch.load(result["frozen_checkpoint"], map_location="cpu", weights_only=True)
    assert checkpoint["frozen_verified"] and checkpoint["checkpoint_kind"] == "new_frozen_predictor_for_future_phase2_use"
    logs = json.loads((Path(result["frozen_checkpoint"]).parent / "training.json").read_text())
    expected = min(logs["logs"], key=lambda row: row["validation_mse"])
    assert result["best_epoch"] == expected["epoch"]
    assert len(checkpoint["signature"]["training_episode_ids"]) == 4


def test_train_only_basis_is_insensitive_to_validation_features(tmp_path):
    from phase1.model import fit_basis
    c1, store1, _, _ = toy_store(tmp_path / "a")
    basis1 = fit_basis(c1, store1, "gate")
    c2, store2, _, _ = toy_store(tmp_path / "b")
    original = store2.features
    def guarded(record):
        assert record["split"] == "train", "basis attempted non-training input"
        return original(record)
    store2.features = guarded
    basis2 = fit_basis(c2, store2, "gate")
    for key in basis1:
        assert np.array_equal(basis1[key], basis2[key])


def test_incomplete_real_encoder_report_refused_before_training(tmp_path):
    report = tmp_path / "report"
    report.mkdir()
    (report / "manifest.json").write_text(json.dumps({"status": "partial_cache_saved_resume_required"}))
    with pytest.raises(ValueError, match="completed real I-JEPA"):
        pilot.verify_inputs(tmp_path / "missing-data", report)


@pytest.mark.parametrize(("setting", "value"), [("recent", 5), ("budget", 3), ("horizons", [3, 9])])
def test_incompatible_sampling_contract_rejected_before_feature_access(tmp_path, setting, value):
    config, store, records, loaded = toy_store(tmp_path)
    config["data"][setting] = value
    with pytest.raises(ValueError, match="requires recent=4, budget=2"):
        pilot.train_task(config, store, "gate", records,
                {"train": "training", "validation": "validation", "test_id": "development_smoke"},
                tmp_path, epochs=1, seed=17, dropout=0., batch_size=4)
    assert not loaded
    assert not (tmp_path / "predictor").exists()


@pytest.mark.parametrize("changed", ["model.py", "inputs.py", "training.py"])
def test_retained_predictor_source_mismatch_refused(tmp_path, monkeypatch, changed):
    control = tmp_path / "experiments/predictor/control"
    source = control / "src/phase1"
    source.mkdir(parents=True)
    files = {}
    for name in ("model.py", "inputs.py", "training.py"):
        path = source / name
        path.write_text("# fixture for byte-hash validation only\n")
        files["src/phase1/" + name] = pilot.sha256(path)
    (control / "source_lock.json").write_text(json.dumps({"files": files}))
    monkeypatch.setattr(pilot, "ROOT", tmp_path)
    assert pilot.verify_predictor_contract(DEFAULT)["files"] == files
    (source / changed).write_text("# changed bytes\n")
    with pytest.raises(ValueError, match="predictor source changed: " + changed):
        pilot.verify_predictor_contract(DEFAULT)


def test_worker_separate_basis_and_checkpoint_output_preserves_source_cache(tmp_path, monkeypatch):
    # This explicit pixel fixture substitutes only the official-encoder gate;
    # it exercises the real data/cache/store/basis/trainer file integration.
    from experiments.phase2 import prepare_controlled_data as data
    from phase1.cache import encode, fit_normalization
    root = tmp_path / "synthetic-pixel-data-NOT-JEPA"
    data.run(root, seed=20261024, train_pairs=2, eval_pairs=2)
    config = json.loads((root / "resolved_config.json").read_text())
    config.update(mode="smoke", threads=1)
    config["encoder"].update(backend="smoke_pixels", smoke_size=2)
    config["model"].update(rank=4, hidden=16)
    index = encode(config)
    fit_normalization(config)
    clean = json.loads((root / "observation_index.json").read_text())
    roles = json.loads((root / "split_roles.json").read_text())["roles"]
    monkeypatch.setattr(pilot, "verify_inputs", lambda *args: (config, clean, roles, index))
    output = tmp_path / "new-training-output"
    output.mkdir()
    pilot.worker(SimpleNamespace(data_root=root, encoded_report=tmp_path / "fixture-report",
                 output=output, epochs=1, seed=17, dropout=.25, batch_size=4, threads=1))
    result = json.loads((output / "training_result.json").read_text())
    assert result["status"] == "new_cpu_predictor_pilot_completed" and result["originals_preserved"]
    assert len(result["tasks"]) == 2 and all(row["parameters_changed"] for row in result["tasks"])
    assert (output / "features").is_symlink() and (output / "dataset").is_symlink()
    assert not (root / "predictor").exists()
    assert all((output / "predictor" / task / "basis.npz").is_file() for task in ("gate", "motion"))
