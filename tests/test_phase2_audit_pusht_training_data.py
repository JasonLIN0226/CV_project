"""Ensure diagnostic summaries cannot silently consume reserved outcomes."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

spec = importlib.util.spec_from_file_location(
    "audit_pusht_training_data", Path(__file__).resolve().parents[1]
    / "experiments/phase2/audit_pusht_training_data.py")
a = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = a
spec.loader.exec_module(a)


@pytest.fixture
def corpus(tmp_path):
    diagnostics = tmp_path/"evaluation_only"
    diagnostics.mkdir()
    episodes, artifacts = [], {}
    for index in range(32):
        name = f"pusht_train_{index:03d}"
        episodes.append({"id": name, "split": "train"})
        trace = {"episode_id": name, "steps": [
            {"info": {"block_pose": [0, 0, 0], "n_contacts": 0}},
            {"info": {"block_pose": [1, 2, .3], "n_contacts": 1, "coverage": .1, "is_success": False}}],
            "paired_diagnostic_max_abs_difference": 0., "later_visible_targets": [{"query": 11, "horizon": 6}]}
        path = diagnostics/f"{name}.json"
        path.write_text(json.dumps(trace))
        artifacts[f"evaluation_only/{name}.json"] = {"bytes": path.stat().st_size, "sha256": a.sha256(path)}
    for split in ("validation", "evaluation"):
        name = f"pusht_{split}_000"
        episodes.append({"id": name, "split": split})
        (diagnostics/f"{name}.json").write_text("RESERVED: DO NOT READ")
    (tmp_path/"protocol.json").write_text(json.dumps({"episodes": episodes}))
    manifest = {"status": "CPU_RGB_action_preparation_completed",
                "protocol_sha256": a.sha256(tmp_path/"protocol.json"), "artifacts": artifacts}
    (tmp_path/"preparation_manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_only_training_diagnostics_are_opened(corpus, monkeypatch):
    original_text, original_bytes = Path.read_text, Path.read_bytes
    def guard(path):
        assert "pusht_evaluation" not in path.name and "pusht_validation" not in path.name
    def read_text(path, *args, **kwargs):
        guard(path)
        return original_text(path, *args, **kwargs)
    def read_bytes(path, *args, **kwargs):
        guard(path)
        return original_bytes(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    result = a.audit_training_data(corpus)
    assert result["episodes"] == 32 and result["episodes_with_contact"] == 32
    assert result["read_splits"] == ["train"]
    assert not result["validation_diagnostics_read"] and not result["evaluation_diagnostics_read"]
    assert len(result["inputs_sha256"]) == 32


def test_tampered_train_diagnostics_rejected_before_parsing(corpus):
    (corpus/"evaluation_only/pusht_train_000.json").write_text("NOT JSON")
    with pytest.raises(ValueError, match="checksum mismatch"):
        a.audit_training_data(corpus)
