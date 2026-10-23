"""Synthetic archive checks; never modify the retained real HPC export."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.phase2 import portable_snapshot as portable
from phase1.model import DynamicsPredictor


@pytest.fixture
def archive(tmp_path, monkeypatch):
    root = tmp_path / "synthetic-not-JEPA"
    root.mkdir()
    run = "runs/cpu-predictor-pilot"

    def json_file(name, value):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def npz_file(name, **values):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **values)

    c = {"model": {"hidden": 4, "rank": 2, "dropout": 0.}, "data": {"recent": 4, "budget": 2, "horizons": [3, 6]}}
    json_file(run + "/training_config.json", c)
    records, episodes = [], {}
    for split in ("train", "validation", "test_id"):
        for index in range(2):
            eid = f"{split}-{index}"
            rec = {"id": eid, "pair": split, "sibling": f"{split}-{1-index}", "task": "gate", "split": split,
                   "query": 12, "seed": 100 if split == "train" else 200 if split == "validation" else 300,
                   "obs": f"observations/{split}/{eid}.npz", "eval": f"evaluation_only/{split}/{eid}.npz"}
            records.append(rec)
            z = np.ones((19, 5), np.float32) * index
            fpath = f"{split}/{eid}.npz"
            npz_file("data/features/" + fpath, z=z, signature=np.array("fixture"), encoder_hash=np.array("fixture-encoder"))
            episodes[eid] = {"path": fpath, "sha256": portable.sha256(root / "data/features" / fpath), "shape": list(z.shape)}
            npz_file("data/dataset/" + rec["obs"], actions=np.zeros((18, 2), np.float32))
            npz_file("data/dataset/" + rec["eval"], evidence=np.array([2], np.int64), hidden_state=np.array([999]))
    json_file("data/dataset/manifest.json", {"records": records})
    json_file("data/features/index.json", {"signature": "fixture", "encoder_hash": "fixture-encoder", "episodes": episodes})
    npz_file("data/features/normalization.npz", mean=np.zeros(5, np.float32), std=np.ones(5, np.float32),
             train_ids=np.array(["train-0", "train-1"]), signature=np.array("fixture"))
    norm_sha = portable.sha256(root / "data/features/normalization.npz")
    json_file("data/features/normalization.json", {"fit_split": "train", "sha256": norm_sha})
    basis = {"input_mean": np.zeros(5, np.float32), "projection": np.eye(5, dtype=np.float32)[:, :2],
             "target_mean": np.zeros(5, np.float32), "output_basis": np.eye(5, dtype=np.float32)[:, :2],
             "train_ids": np.array(["train-0", "train-1"])}
    basis_relative = run + "/predictor/gate/basis.npz"
    npz_file(basis_relative, **basis)
    model = DynamicsPredictor(5, c["model"], (3, 6), basis)
    signature = {"task": "gate", "seed": 17, "cache_signature": "fixture", "normalization_sha256": norm_sha,
                 "basis_sha256": portable.sha256(root / basis_relative)}
    path = root / run / "predictor/gate/seed_17/frozen.pt"
    path.parent.mkdir()
    torch.save({"signature": signature, "model": model.state_dict(), "selected_epoch": 1}, path)
    json_file(run + "/training_result.json", {"fixture": True})
    source_hashes = {"experiments/predictor/control/src/phase1/" + name:
                     portable.sha256(portable.ROOT / "experiments/predictor/control/src/phase1" / name)
                     for name in ("model.py", "inputs.py", "training.py")}
    json_file(run + "/run_manifest.json", {"status": "completed", "source_sha256": source_hashes,
                "training_result_sha256": portable.sha256(root / run / "training_result.json")})
    index = {str(p.relative_to(root)): {"sha256": portable.sha256(p), "bytes": p.stat().st_size}
             for p in root.rglob("*") if p.is_file()}
    json_file("backup-index.json", {"files": index})
    monkeypatch.setattr(portable, "BACKUP_INDEX_SHA256", portable.sha256(root / "backup-index.json"))
    return root


def test_snapshot_loads_only_train_validation_and_freezes_verified_model(archive):
    snapshot = portable.PortableSnapshot(archive)
    assert len(snapshot.records) == 4 and len(snapshot.all_records) == 6
    rec = snapshot.records[0]
    features = snapshot.features(rec)
    assert features.shape == (19, 5) and features.dtype == np.float32
    features[:] = 999
    assert np.all(snapshot.features(rec) == 0)
    assert snapshot.evidence(rec).tolist() == [2]
    model, meta = snapshot.model("gate", "frozen")
    assert not any(m.training for m in model.modules()) and not any(p.requires_grad for p in model.parameters())
    assert meta["state_sha256"] == portable.state_hash(model)
    assert snapshot.audit()["test_arrays_opened"] == 0


@pytest.mark.parametrize("accessor", ["features", "actions", "evidence"])
def test_test_record_rejected_before_array_access(archive, accessor):
    snapshot = portable.PortableSnapshot(archive)
    before = list(snapshot.array_access)
    test = next(r for r in snapshot.all_records if r["split"] == "test_id")
    with pytest.raises(ValueError, match="train/validation"):
        getattr(snapshot, accessor)(test)
    assert snapshot.array_access == before
    with pytest.raises(ValueError, match="does not open test"):
        snapshot.read_npz("data/features/test_id/test_id-0.npz")
    assert snapshot.array_access == before


def test_loaded_file_tamper_and_archive_index_tamper_rejected(archive):
    snapshot = portable.PortableSnapshot(archive)
    path = archive / "data/features/normalization.npz"
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="mutated"):
        snapshot.audit()
    with pytest.raises(ValueError, match="checksum/size"):
        portable.PortableSnapshot(archive)


def test_backup_index_mutation_is_part_of_end_audit(archive):
    snapshot = portable.PortableSnapshot(archive)
    path = archive / "backup-index.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="[Ii]ndex"):
        snapshot.audit()


def test_record_mutation_cannot_change_canonical_metadata(archive):
    snapshot = portable.PortableSnapshot(archive)
    # Even a caller mutating a returned record must not mutate the reference
    # against which 'unchanged train/validation observation record' is checked.
    rec = snapshot.records[0]
    rec["query"] += 1
    with pytest.raises(ValueError, match="unchanged"):
        snapshot.features(rec)


def test_forged_record_and_outside_path_rejected(archive):
    snapshot = portable.PortableSnapshot(archive)
    rec = copy.deepcopy(snapshot.records[0])
    rec["obs"] = "observations/test_id/test_id-0.npz"
    with pytest.raises(ValueError, match="unchanged"):
        snapshot.actions(rec)
    with pytest.raises(ValueError, match="absent"):
        snapshot.verified_path("../outside.npz")


def test_symlink_escape_rejected_before_deserializing(archive, tmp_path):
    snapshot = portable.PortableSnapshot(archive)
    path = archive / "runs/cpu-predictor-pilot/predictor/gate/seed_17/frozen.pt"
    outside = tmp_path / "outside.pt"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="absent"):
        snapshot.checkpoint("gate")
