import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_controlled_data", ROOT/"experiments/phase2/prepare_controlled_data.py")
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def test_new_config_preserves_dynamics_and_reserves_old_seeds(tmp_path):
    reference = json.loads(p.REFERENCE.read_text())
    old = json.dumps(reference, sort_keys=True)
    c = p.make_config(reference, tmp_path)
    assert json.dumps(reference, sort_keys=True) == old
    for key in reference["data"]:
        if key not in {"seed", "pairs"}:
            assert c["data"][key] == reference["data"][key]
    assert c["data"]["motion_dt"] == [3, 6]
    assert sum(c["data"]["pairs"].values()) * 2 * 2 == 128
    for seed in (6301, 9137):
        with pytest.raises(ValueError, match="excluded"):
            p.make_config(reference, tmp_path, seed=seed)


def fixture(tmp_path, *, extra=None):
    root = tmp_path/"dataset/observations/train"
    root.mkdir(parents=True)
    records = []
    for eid, sibling in (("a", "b"), ("b", "a")):
        path = root/f"{eid}.npz"
        data = dict(rgb=np.zeros((5, 96, 96, 3), np.uint8), actions=np.zeros((4, 2), np.float32), query=np.int64(2))
        data.update(extra or {})
        np.savez_compressed(path, **data)
        records.append(dict(id=eid, sibling=sibling, pair="pair", split="train", task="gate", seed=10,
                            query=2, frames=5, obs=f"observations/train/{eid}.npz", obs_sha256=p.sha256(path)))
    return {"output": str(tmp_path), "data": {"image_size": 96}}, records


def test_observation_schema_cannot_smuggle_simulator_state(tmp_path):
    c, records = fixture(tmp_path, extra={"state": np.zeros((5, 2))})
    with pytest.raises(ValueError, match="forbidden"):
        p.audit_observation_boundary(c, records)


def test_identity_index_excludes_diagnostic_metadata(tmp_path):
    c, records = fixture(tmp_path)
    for rec in records:
        rec.update(eval="evaluation_only/train/evidence.npz", label=1, evidence=[0], gap=7)
    audit = p.audit_observation_boundary(c, records)
    assert audit["passed"] and audit["episodes"] == 2 and audit["pairs"] == 1
    assert all(not {"eval", "label", "evidence", "gap"}.intersection(r) for r in audit["observation_index"])


def test_sibling_group_cannot_cross_split(tmp_path):
    c, records = fixture(tmp_path)
    records[1]["split"] = "validation"
    with pytest.raises(ValueError, match="crosses splits"):
        p.audit_observation_boundary(c, records)


def test_diagnostics_path_not_allowed_as_observation(tmp_path):
    c, records = fixture(tmp_path)
    records[0]["obs"] = "evaluation_only/train/a.npz"
    with pytest.raises(ValueError, match="diagnostics"):
        p.audit_observation_boundary(c, records)
