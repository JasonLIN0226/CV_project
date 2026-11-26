"""Checks for environments/motion_v2/frozen.py: input verification and replay comparison.

The real-asset test needs the frozen study folders and runs only where BM_MOTION_ROOT is set
(on Burst: BM_MOTION_ROOT=/scratch/xy3257).
"""
import hashlib
import json
import os

import numpy as np
import pytest

from bounded_memory import runs
from bounded_memory.environments.motion_v2 import frozen

REPO = runs.REPO
EVIDENCE = REPO / "results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu"
CONFIG = REPO / "configs/motion_v2/frozen.yaml"


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def test_check_tree_reports_missing_mismatched_and_skipped(tmp_path):
    expected = {"a.txt": _write(tmp_path / "a.txt", b"a"), "b.txt": _write(tmp_path / "b.txt", b"b"),
                "gone.txt": hashlib.sha256(b"x").hexdigest(), "hf_cache/w.bin": "0" * 64}
    (tmp_path / "b.txt").write_bytes(b"changed")
    report = frozen.check_tree(tmp_path, expected, skip=("hf_cache/",))
    assert report["checked"] == 1 and report["skipped"] == 1
    assert report["missing"] == ["gone.txt"] and report["mismatched"] == ["b.txt"] and not report["ok"]


def test_frozen_step2_inventory_is_the_16614_assets_without_the_encoder_cache():
    checksums = frozen.frozen_step2_checksums()
    assert len(checksums) == 1232 and not any(k.startswith("hf_cache/") for k in checksums)
    assert {"resolved_config.json", "features/index.json", "decoder/train_position_decoder.npz"} <= set(checksums)
    assert sum(k.startswith("predictor/motion/seed_") for k in checksums) == 6   # three readers: checkpoint + selection


def test_tracked_students_and_raw_losses_match_the_step34_inventory():
    inventory = json.loads((EVIDENCE / "artifact_sha256.json").read_text())
    tracked = {k: v for k, v in inventory.items() if k.startswith(("students/", "online/ordinary/raw_losses/", "online/long_gap/raw_losses/"))}
    report = frozen.check_tree(EVIDENCE, tracked)
    assert report["ok"], {k: report[k][:5] for k in ("missing", "mismatched")}
    assert sum(k.endswith(".pt") for k in tracked) == 54
    assert sum(k.startswith("online/ordinary/raw_losses/") for k in tracked) == 28


def test_verify_lists_every_missing_input(tmp_path):
    paths = frozen.Paths(tmp_path / "step2", tmp_path / "dataset", EVIDENCE, tmp_path / "receipt", tmp_path / "lg")
    with pytest.raises(frozen.FrozenInputError) as error:
        frozen.verify(paths, ("ordinary", "long_gap"))
    message = str(error.value)
    for name in ("step2", "dataset", "longgap_receipt", "longgap_features"):
        assert name + ":" in message
    assert "evidence:" not in message


def test_long_gap_needs_both_long_gap_inputs(tmp_path):
    paths = frozen.Paths(tmp_path, tmp_path, EVIDENCE)
    with pytest.raises(frozen.FrozenInputError, match="longgap_receipt"):
        frozen.verify(paths, ("long_gap",))


def _losses(path, scale=1.0, success=None):
    rng = np.random.default_rng(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, position=rng.random((3, 3, 96, 2, 2)) * scale, feature=rng.random((3, 3, 96, 2, 2)),
             success=success if success is not None else np.ones((3, 96, 2), bool))


def test_compare_separates_identical_close_differs_and_missing(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    for name in ("same", "close", "far", "only_old"):
        _losses(old / f"{name}.npz")
    _losses(new / "same.npz")
    _losses(new / "close.npz", scale=1 + 1e-9)
    _losses(new / "far.npz", scale=1.5)
    _losses(new / "only_new.npz")
    rows = frozen.compare(new, old)["methods"]
    assert rows["same"]["verdict"] == "identical" and rows["same"]["position_max_abs_diff"] == 0
    assert rows["close"]["verdict"] == "close"
    assert rows["far"]["verdict"] == "differs"
    assert rows["only_old"]["verdict"] == "missing in replay"
    assert rows["only_new"]["verdict"] == "not in the tracked evidence"
    assert set(rows["same"]["position_mse_+3"]) == {"tracked", "replay"}


def test_compare_flags_a_changed_success_vector(tmp_path):
    success = np.ones((3, 96, 2), bool)
    _losses(tmp_path / "old/m.npz", success=success)
    success = success.copy()
    success[0, 0, 0] = False
    _losses(tmp_path / "new/m.npz", success=success)
    assert frozen.compare(tmp_path / "new", tmp_path / "old")["methods"]["m"]["verdict"] == "differs"


def test_config_points_at_the_tracked_evidence(monkeypatch, tmp_path):
    for name in ("BM_MOTION_ROOT", "BM_SCRATCH"):        # the config also names the fresh scenes and scorers
        monkeypatch.setenv(name, str(tmp_path))
    monkeypatch.setenv("BM_REPO", str(REPO))
    paths = frozen.Paths.from_config(runs.load_config(CONFIG)["assets"])
    assert paths.evidence == EVIDENCE.resolve()
    assert paths.step2.is_relative_to(tmp_path) and paths.longgap_features.is_relative_to(tmp_path)


def test_replay_refuses_an_existing_output(tmp_path):
    with pytest.raises(FileExistsError):
        frozen.replay(None, None, "ordinary", tmp_path)


@pytest.mark.skipif(not os.environ.get("BM_MOTION_ROOT"), reason="needs the frozen Motion v2 folders: set BM_MOTION_ROOT")
def test_real_frozen_inputs_match(monkeypatch):
    monkeypatch.setenv("BM_REPO", str(REPO))
    paths = frozen.Paths.from_config(runs.load_config(CONFIG)["assets"])
    report = frozen.verify(paths, frozen.INTERVALS)
    assert report["step2"]["checked"] == 1232 and report["dataset"]["ok"]
    assert report["longgap_receipt"]["checked"] == 391 and report["longgap_features"]["checked"] == 194
