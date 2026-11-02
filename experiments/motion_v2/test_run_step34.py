"""Artificial JSON-only admission tests; no real data, Git, Slurm, or models."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location("motion_run_step34_under_test", Path(__file__).with_name("run_step34.py"))
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)
REQUIRED = ("status.json", "audit.json", "data/dataset/manifest.json")
BUDGET = {"prior_a100_seconds": 3705, "total_a100_seconds": 14400,
          "remaining_a100_seconds": 10695, "estimated_seconds": 8500, "wall_seconds": 10500}


def put_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def receipt(tmp_path):
    root = tmp_path / "cpu_receipt"
    manifest = {"mode": "validation_longgap_only",
                "records": [{"id": f"synthetic_{index:03}", "pair": f"synthetic_pair_{index // 2:03}",
                             "split": "validation", "task": "motion"} for index in range(192)]}
    put_json(root / "data/dataset/manifest.json", manifest)
    put_json(root / "audit.json", {"passed": True, "fixture": "artificial JSON, no episodes"})
    put_json(root / "status.json", {"status": "passed",
                                   "dataset_manifest_sha256": checksum(root / "data/dataset/manifest.json")})
    execution = {"cpu_receipt": {"artifact_sha256": {name: checksum(root / name) for name in REQUIRED}}}
    return root, execution, manifest


def refresh(execution, root, name):
    execution["cpu_receipt"]["artifact_sha256"][name] = checksum(root / name)


def test_cpu_receipt_accepts_hash_bound_validation_json(tmp_path):
    root, execution, manifest = receipt(tmp_path)
    before = copy.deepcopy(execution)
    assert RUNNER.validate_cpu_receipt(execution, root) == manifest
    assert execution == before


@pytest.mark.parametrize("missing", REQUIRED)
def test_every_required_receipt_hash_is_mandatory(tmp_path, missing):
    root, execution, _ = receipt(tmp_path)
    del execution["cpu_receipt"]["artifact_sha256"][missing]
    with pytest.raises(ValueError, match="Incomplete"):
        RUNNER.validate_cpu_receipt(execution, root)


@pytest.mark.parametrize("name", REQUIRED)
def test_changed_receipt_bytes_rejected(tmp_path, name):
    root, execution, _ = receipt(tmp_path)
    with (root / name).open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="changed"):
        RUNNER.validate_cpu_receipt(execution, root)


@pytest.mark.parametrize("name", REQUIRED)
def test_missing_required_file_rejected(tmp_path, name):
    root, execution, _ = receipt(tmp_path)
    # These are disposable fixture files, never study records.
    (root / name).unlink()
    with pytest.raises(ValueError, match="changed"):
        RUNNER.validate_cpu_receipt(execution, root)


@pytest.mark.parametrize("bad_hash", [None, "", "not-a-sha", "0" * 64, 123])
def test_invalid_required_hash_rejected(tmp_path, bad_hash):
    root, execution, _ = receipt(tmp_path)
    execution["cpu_receipt"]["artifact_sha256"]["audit.json"] = bad_hash
    with pytest.raises(ValueError, match="changed"):
        RUNNER.validate_cpu_receipt(execution, root)


@pytest.mark.parametrize("state", ["stopped", "failed", "running", "PASS", None])
def test_unpassed_cpu_status_rejected_even_with_valid_hash(tmp_path, state):
    root, execution, _ = receipt(tmp_path)
    status = json.loads((root / "status.json").read_text())
    status["status"] = state
    put_json(root / "status.json", status)
    refresh(execution, root, "status.json")
    with pytest.raises(ValueError, match="did not pass"):
        RUNNER.validate_cpu_receipt(execution, root)


@pytest.mark.parametrize("passed", [False, None, 1, "true"])
def test_audit_requires_literal_true(tmp_path, passed):
    root, execution, _ = receipt(tmp_path)
    put_json(root / "audit.json", {"passed": passed})
    refresh(execution, root, "audit.json")
    with pytest.raises(ValueError, match="did not pass"):
        RUNNER.validate_cpu_receipt(execution, root)


def test_status_manifest_digest_binding_is_independent_of_inventory(tmp_path):
    root, execution, _ = receipt(tmp_path)
    status = json.loads((root / "status.json").read_text())
    status["dataset_manifest_sha256"] = "0" * 64
    put_json(root / "status.json", status)
    refresh(execution, root, "status.json")
    with pytest.raises(ValueError, match="hash disagreement"):
        RUNNER.validate_cpu_receipt(execution, root)


@pytest.mark.parametrize("split", ["train", "test", "test_long_gap", "conf", "confirmation", "development"])
def test_nonvalidation_record_rejected_with_all_hashes_current(tmp_path, split):
    root, execution, manifest = receipt(tmp_path)
    manifest["records"][95]["split"] = split
    put_json(root / "data/dataset/manifest.json", manifest)
    refresh(execution, root, "data/dataset/manifest.json")
    status = json.loads((root / "status.json").read_text())
    status["dataset_manifest_sha256"] = checksum(root / "data/dataset/manifest.json")
    put_json(root / "status.json", status)
    refresh(execution, root, "status.json")
    with pytest.raises(ValueError, match="Only new validation"):
        RUNNER.validate_cpu_receipt(execution, root)


@pytest.mark.parametrize("count", [0, 191, 193])
def test_exactly_192_validation_records_required(tmp_path, count):
    root, execution, manifest = receipt(tmp_path)
    manifest["records"] = [{"id": f"synthetic_{i}", "split": "validation"} for i in range(count)]
    put_json(root / "data/dataset/manifest.json", manifest)
    refresh(execution, root, "data/dataset/manifest.json")
    status = json.loads((root / "status.json").read_text())
    status["dataset_manifest_sha256"] = checksum(root / "data/dataset/manifest.json")
    put_json(root / "status.json", status)
    refresh(execution, root, "status.json")
    with pytest.raises(ValueError, match="Only new validation"):
        RUNNER.validate_cpu_receipt(execution, root)


def test_additional_inventory_files_are_checked(tmp_path):
    root, execution, _ = receipt(tmp_path)
    extra = root / "extra_receipt.json"
    put_json(extra, {"receipt": "synthetic"})
    refresh(execution, root, "extra_receipt.json")
    RUNNER.validate_cpu_receipt(execution, root)
    extra.write_text("changed\n")
    with pytest.raises(ValueError, match="changed"):
        RUNNER.validate_cpu_receipt(execution, root)


@pytest.mark.parametrize("kind", ["relative_escape", "absolute_escape", "symlink_escape"])
def test_receipt_inventory_cannot_escape_root(tmp_path, kind):
    root, execution, _ = receipt(tmp_path)
    outside = tmp_path / "outside.json"
    put_json(outside, {"synthetic": True})
    if kind == "relative_escape":
        name = "../outside.json"
    elif kind == "absolute_escape":
        name = str(outside)
    else:
        name = "linked.json"
        (root / name).symlink_to(outside)
    execution["cpu_receipt"]["artifact_sha256"][name] = checksum(outside)
    with pytest.raises(ValueError):
        RUNNER.validate_cpu_receipt(execution, root)


def artificial_registration(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    paths = {"config": repo / "config.json", "asset_inventory": repo / "assets.json", "document": repo / "protocol.md"}
    put_json(paths["config"], {"budget": dict(BUDGET)})
    put_json(paths["asset_inventory"], {"artifact_sha256": {}})
    paths["document"].write_text("Artificial registration fixture\n")
    source = repo / "source.py"
    source.write_text("# Synthetic source; never executed\n")
    protocol = {"status": "FROZEN_STEP3_4_PROTOCOL", "registered_before_run": True,
                "source_sha256": {source.name: checksum(source)},
                **{name: {"path": path.name, "sha256": checksum(path)} for name, path in paths.items()}}
    protocol_path = repo / "registration.json"
    put_json(protocol_path, protocol)
    monkeypatch.setattr(RUNNER, "REPO", repo)
    monkeypatch.setattr(RUNNER, "committed", lambda path: Path(path))
    return protocol_path, protocol, paths, source


def test_registration_uses_only_monkeypatched_artificial_sources(tmp_path, monkeypatch):
    path, protocol, _, _ = artificial_registration(tmp_path, monkeypatch)
    actual, config = RUNNER.registration(path)
    assert actual == protocol and config["budget"] == BUDGET


def test_registration_source_hash_mismatch_rejected(tmp_path, monkeypatch):
    path, _, _, source = artificial_registration(tmp_path, monkeypatch)
    source.write_text("# Changed synthetic source\n")
    with pytest.raises(ValueError, match="Source hash"):
        RUNNER.registration(path)


@pytest.mark.parametrize("field", ["prior_a100_seconds", "total_a100_seconds", "remaining_a100_seconds", "estimated_seconds", "wall_seconds"])
def test_registered_budget_cannot_drift(tmp_path, monkeypatch, field):
    path, protocol, paths, _ = artificial_registration(tmp_path, monkeypatch)
    config = json.loads(paths["config"].read_text())
    config["budget"][field] += 1
    put_json(paths["config"], config)
    protocol["config"]["sha256"] = checksum(paths["config"])
    put_json(path, protocol)
    with pytest.raises(ValueError, match="Budget differs"):
        RUNNER.registration(path)
