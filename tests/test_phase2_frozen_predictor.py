"""Synthetic artifact/forward fixtures only; no official checkpoint download."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.phase2 import frozen_predictor as frozen
from experiments.phase2.deletion import CausalContext, CausalParent, OfflineFuture, generate_deletion_supervision
from experiments.phase2.memory import MemoryEntry
from experiments.phase2.training_samples import PredictorContext, PredictorSample, phase1_arrays
from phase1.config import DEFAULT
from phase1.inputs import collate
from phase1.model import DynamicsPredictor, project
from phase1.utils import write_json, write_npz


@pytest.fixture
def setup(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    data, report, run = (tmp_path / name for name in ("synthetic-data-NOT-JEPA", "fixture-encoded-report", "fixture-training"))
    data.mkdir()
    report.mkdir()
    run.mkdir()
    config = copy.deepcopy(DEFAULT)
    config["model"].update(hidden=16, rank=4, dropout=0.)
    config["output"] = str(data)
    config["data"].update(recent=4, budget=2, horizons=[3, 6])
    config["training"].update(epochs=1, seeds=[17])
    records = [{"id": f"{split}-{i}", "task": "gate", "split": split} for split in ("train", "validation") for i in range(2)]
    write_json(data / "resolved_config.json", config)
    write_json(data / "dataset/manifest.json", {"records": records})
    write_json(data / "features/index.json", {"signature": "synthetic_NOT_JEPA"})
    write_npz(data / "features/normalization.npz", mean=np.zeros(8, np.float32), std=np.ones(8, np.float32))
    for name in ("manifest.json", "encoder_config.json", "sample_summary.json"):
        write_json(report / name, {"fixture": "synthetic_NOT_JEPA"})
    monkeypatch.setattr(frozen, "verify_inputs", lambda *args: (config, {"records": records}, {}, {"signature": "synthetic_NOT_JEPA"}))
    basis = {"input_mean": np.zeros(8, np.float32), "projection": np.eye(8, dtype=np.float32)[:, :4],
             "target_mean": np.zeros(8, np.float32), "output_basis": np.eye(8, dtype=np.float32)[:, :4],
             "train_ids": np.array(["train-0", "train-1"])}
    basis_path = run / "predictor/gate/basis.npz"
    write_npz(basis_path, **basis)
    torch.manual_seed(17)
    model = DynamicsPredictor(8, config["model"], (3, 6), basis).eval().requires_grad_(False)
    signature = {"architecture": "phase1.model.DynamicsPredictor", "task": "gate", "seed": 17,
                 "cache_signature": "synthetic_NOT_JEPA",
                 "normalization_sha256": frozen.sha256(data / "features/normalization.npz"),
                 "basis_sha256": frozen.sha256(basis_path), "horizons": [3, 6],
                 "model_config": config["model"], "training": config["training"],
                 "training_episode_ids": ["train-0", "train-1"], "validation_episode_ids": ["validation-0", "validation-1"]}
    path = run / "predictor/gate/seed_17/frozen.pt"
    path.parent.mkdir()
    torch.save({"checkpoint_kind": "new_frozen_predictor_for_future_phase2_use", "signature": signature,
                "model": model.state_dict(), "model_config": config["model"], "feature_dimension": 8,
                "horizons": (3, 6), "selected_epoch": 1, "validation_mse": .2, "frozen_verified": True}, path)
    write_json(run / "training_config.json", config)
    result = {"status": "new_cpu_predictor_pilot_completed", "predictor_trained": True, "originals_preserved": True,
              "tasks": [{"task": "gate", "seed": 17, "basis_train_only": True,
                         "retained_predictor_contract": frozen.verify_predictor_contract(config),
                         "frozen_checkpoint_sha256": frozen.sha256(path), "best_epoch": 1, "best_validation_mse": .2}],
              "original_artifact_sha256": {str(data / name): frozen.sha256(data / name) for name in
                   ("resolved_config.json", "dataset/manifest.json", "features/index.json", "features/normalization.npz")}}
    write_json(run / "training_result.json", result)
    write_json(run / "run_manifest.json", {"status": "completed", "worker_exit_code": 0,
               "training_result_sha256": frozen.sha256(run / "training_result.json"),
               "source_sha256": {"experiments/predictor/control/src/phase1/" + name:
                   frozen.sha256(frozen.ROOT / "experiments/predictor/control/src/phase1" / name)
                   for name in ("model.py", "inputs.py", "training.py")}})
    evaluator = frozen.FrozenPredictorEvaluator(run, report, data, "gate")
    rng = np.random.default_rng(42)
    observations = rng.normal(size=(19, 8)).astype(np.float32)
    actions = rng.uniform(-.2, .2, size=(18, 2)).astype(np.float32)
    entries = {t: MemoryEntry("fixture-episode", t, observations[t]) for t in (1, 4, 7, 9, 10, 11, 12)}
    parent = CausalParent(2, tuple(entries[t] for t in (1, 4, 7)),
                         CausalContext(12, tuple(entries[t] for t in (9, 10, 11, 12))))
    prefix = np.concatenate([np.zeros((1, 2), np.float32), np.cumsum(actions[:12], axis=0)])
    sums = {entry.entry_id: prefix[12] - prefix[entry.timestamp] for entry in entries.values()}
    return evaluator, parent, sums, actions, observations, basis, run, report, data


def test_callback_matches_training_bridge_and_ignores_unprovided_future_suffix(setup):
    evaluator, parent, sums, actions, observations, basis, *_ = setup
    kept = parent.candidates[:2]
    callback = evaluator.bind(parent, sums)
    future = OfflineFuture((3, 6), observations[[15, 18]], actions[12:18], np.ones(2, bool))
    sample = PredictorSample(PredictorContext(12, parent.context.recent, kept, actions[:12], "additive_controls"), future, {})
    arrays, _ = phase1_arrays(sample)
    arrays["projected_z"] = project(arrays["z"], basis)
    arrays["projected_current"] = project(arrays["current"], basis)
    with torch.inference_mode():
        expected = evaluator._model(collate([arrays], "cpu"))[0].numpy()
    assert np.array_equal(callback(kept, parent.context, 3, actions[12:15]), expected[0])
    assert np.array_equal(callback(kept, parent.context, 6, actions[12:18]), expected[1])
    changed = copy.deepcopy(arrays)
    changed["actions"] = changed["actions"].copy()
    changed["actions"][3:] = 10000
    with torch.inference_mode():
        altered = evaluator._model(collate([changed], "cpu"))[0, 0].numpy()
    assert np.array_equal(altered, expected[0])
    assert evaluator.audit()["model_calls"] == 2
    assert "self" not in callback.__code__.co_freevars
    assert not any("store" in name or "future" in name for name in callback.__code__.co_freevars)


def test_binding_copies_summaries_and_generates_all_equal_k_labels(setup):
    evaluator, parent, sums, actions, observations, *_ = setup
    callback = evaluator.bind(parent, sums)
    kept = parent.candidates[:2]
    before = callback(kept, parent.context, 3, actions[12:15])
    for value in sums.values():
        value[:] = 999
    assert np.array_equal(before, callback(kept, parent.context, 3, actions[12:15]))
    future = OfflineFuture((3, 6), observations[[15, 18]], actions[12:18], np.ones(2, bool))
    labels = generate_deletion_supervision(parent, future, callback, provenance=evaluator.provenance)
    assert labels.labels.feature_mse.shape == (3, 2) and np.isfinite(labels.labels.feature_mse).all()
    assert all(len(row) == 2 for row in labels.labels.kept_ids)
    assert evaluator.audit()["passed"]


def test_adapter_rejects_extra_archive_and_wrong_horizon_prefix(setup):
    evaluator, parent, sums, actions, *_ = setup
    with pytest.raises(ValueError, match="exactly"):
        evaluator.bind(parent, {**sums, ("fixture-episode", 0): np.zeros(2, np.float32)})
    callback = evaluator.bind(parent, sums)
    with pytest.raises(ValueError, match="horizon-length"):
        callback(parent.candidates[:2], parent.context, 3, actions[12:18])
    with pytest.raises(ValueError, match="two distinct"):
        callback(parent.candidates[:1], parent.context, 3, actions[12:15])


def test_mutation_and_checkpoint_tampering_are_detected(setup):
    evaluator, parent, sums, actions, _, _, run, report, data = setup
    callback = evaluator.bind(parent, sums)
    evaluator._model.train()
    with pytest.raises(ValueError, match="changed after binding"):
        callback(parent.candidates[:2], parent.context, 3, actions[12:15])
    evaluator._model.eval()
    with torch.no_grad():
        evaluator._model.target_mean.add_(1)
    with pytest.raises(ValueError, match="changed"):
        evaluator.audit()
    path = run / "predictor/gate/seed_17/frozen.pt"
    with path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="checkpoint checksum"):
        frozen.FrozenPredictorEvaluator(run, report, data, "gate")
