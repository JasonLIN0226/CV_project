"""Byte-only intake verification, including no unsafe checkpoint deserialization."""
import hashlib
import json
from pathlib import Path

import pytest

from experiments.phase2 import verify_assets as v


PREFIX = "/scratch/xl4358"


def manifest(tmp_path, *, optional=False, paths=None):
    paths = paths or [PREFIX + "/models/best.pt", PREFIX + "/basis.npz"]
    records = [{"path": path, "sha256": hashlib.sha256(content).hexdigest(), "kind": "fixture"}
               for path, content in zip(paths, [b"checkpoint bytes", b"basis bytes"])]
    data = {"schema_version": 1, "audited_commit": "fixture", "required_hashed_assets": records}
    if optional:
        data["optional_original_diagnostic_assets"] = [{"path": PREFIX + "/probe.npz", "sha256": "1" * 64}]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    return path, data


def store(tmp_path):
    root = tmp_path / "intake"
    (root / "models").mkdir(parents=True)
    (root / "models/best.pt").write_bytes(b"checkpoint bytes")
    (root / "basis.npz").write_bytes(b"basis bytes")
    return root


def test_root_mapping_verifies_exact_bytes_without_changes(tmp_path):
    path, _ = manifest(tmp_path)
    root = store(tmp_path)
    before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}
    report = v.verify_assets(path, root=root)
    assert report["status"] == "MATCHED" and report["required_hashed_assets_match"]
    assert report["counts"]["required"] == {"matched": 2, "missing": 0, "incompatible": 0, "unreadable": 0, "total": 2}
    assert not report["latest_repair_verified"] and not report["models_deserialized"]
    after = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}
    assert before == after


def test_missing_and_hash_mismatch_are_distinct(tmp_path):
    path, _ = manifest(tmp_path)
    root = store(tmp_path)
    (root / "models/best.pt").unlink()
    (root / "basis.npz").write_bytes(b"different basis")
    report = v.verify_assets(path, root=root)
    assert report["status"] == "INCOMPATIBLE"
    assert [r["status"] for r in report["assets"]] == ["missing", "incompatible"]
    assert report["assets"][0]["actual_sha256"] is None
    assert report["assets"][1]["reason"] == "sha256_mismatch"


def test_absolute_mapping_and_unmapped_never_fall_back_to_remote_paths(tmp_path):
    path, data = manifest(tmp_path)
    exact = tmp_path / "renamed checkpoint.pt"
    exact.write_bytes(b"checkpoint bytes")
    report = v.verify_assets(path, path_mapping={data["required_hashed_assets"][0]["path"]: str(exact)})
    assert report["assets"][0]["status"] == "matched"
    assert report["assets"][1]["status"] == "missing"
    assert report["assets"][1]["reason"] == "unmapped" and report["assets"][1]["local_path"] is None
    with pytest.raises(ValueError, match="original remote paths are never implicit"):
        v.verify_assets(path)


def test_relative_exact_mapping_overrides_root(tmp_path):
    path, data = manifest(tmp_path)
    root = store(tmp_path)
    (root / "renamed.pt").write_bytes(b"checkpoint bytes")
    (root / "models/best.pt").write_bytes(b"wrong")
    report = v.verify_assets(path, root=root, path_mapping={data["required_hashed_assets"][0]["path"]: "renamed.pt"})
    assert report["status"] == "MATCHED"
    assert report["assets"][0]["local_path"] == str(root / "renamed.pt")


def test_relative_mapping_requires_root_and_cannot_escape_it(tmp_path):
    path, data = manifest(tmp_path)
    source = data["required_hashed_assets"][0]["path"]
    report = v.verify_assets(path, path_mapping={source: "relative.pt"})
    assert report["assets"][0]["reason"] == "relative_mapping_requires_root"
    root = store(tmp_path)
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"checkpoint bytes")
    report = v.verify_assets(path, root=root, path_mapping={source: "../outside.pt"})
    assert report["assets"][0]["reason"] == "mapped_path_escapes_root"
    assert report["assets"][0]["actual_sha256"] is None


def test_root_mapping_does_not_follow_escaping_symlink(tmp_path):
    path, _ = manifest(tmp_path)
    root = store(tmp_path)
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"checkpoint bytes")
    (root / "models/best.pt").unlink()
    (root / "models/best.pt").symlink_to(outside)
    report = v.verify_assets(path, root=root)
    assert report["assets"][0]["reason"] == "mapped_path_escapes_root"
    assert report["assets"][0]["actual_sha256"] is None


def test_component_prefix_not_string_prefix(tmp_path):
    path, _ = manifest(tmp_path, paths=["/scratch/xl43580/models/best.pt", PREFIX + "/basis.npz"])
    report = v.verify_assets(path, root=store(tmp_path))
    assert report["assets"][0]["reason"] == "outside_source_prefix"


def test_optional_assets_reported_but_do_not_block_required_match(tmp_path):
    path, _ = manifest(tmp_path, optional=True)
    root = store(tmp_path)
    short = v.verify_assets(path, root=root)
    full = v.verify_assets(path, root=root, include_optional=True)
    assert len(short["assets"]) == 2 and len(full["assets"]) == 3
    assert full["status"] == "MATCHED" and full["counts"]["optional"]["missing"] == 1


@pytest.mark.parametrize("corruption", ["duplicate", "digest", "relative", "traversal", "schema", "empty"])
def test_malformed_manifest_is_rejected_before_hashing(tmp_path, monkeypatch, corruption):
    path, data = manifest(tmp_path)
    if corruption == "duplicate":
        data["required_hashed_assets"].append(data["required_hashed_assets"][0])
    elif corruption == "digest":
        data["required_hashed_assets"][0]["sha256"] = "invalid"
    elif corruption == "relative":
        data["required_hashed_assets"][0]["path"] = "models/best.pt"
    elif corruption == "traversal":
        data["required_hashed_assets"][0]["path"] = PREFIX + "/../other/model.pt"
    elif corruption == "schema":
        data["schema_version"] = 99
    else:
        data["required_hashed_assets"] = []
    path.write_text(json.dumps(data))
    monkeypatch.setattr(v, "_hash_regular_file", lambda *args: pytest.fail("must not hash an invalid manifest"))
    with pytest.raises(ValueError):
        v.verify_assets(path, root=tmp_path)


def test_mapping_typo_is_not_silently_ignored(tmp_path):
    path, _ = manifest(tmp_path)
    with pytest.raises(ValueError, match="unknown manifest path"):
        v.verify_assets(path, root=tmp_path, path_mapping={PREFIX + "/typo.pt": "anything"})


def test_nonregular_file_and_permission_error_reported(tmp_path, monkeypatch):
    path, _ = manifest(tmp_path)
    root = store(tmp_path)
    (root / "models/best.pt").unlink()
    (root / "models/best.pt").mkdir()
    report = v.verify_assets(path, root=root)
    assert report["assets"][0]["reason"] == "not_a_regular_file"

    def unreadable(path):
        raise PermissionError("fixture")

    monkeypatch.setattr(v, "_hash_regular_file", unreadable)
    report = v.verify_assets(path, root=root)
    assert report["status"] == "UNREADABLE" and report["counts"]["required"]["unreadable"] == 2


def test_pt_is_hashed_as_bytes_without_executing_pickle(tmp_path):
    # A pickle GLOBAL/REDUCE instruction would create a marker if deserialized.
    # The fixture is intentionally consumed only as ordinary bytes by the tool.
    path, data = manifest(tmp_path)
    root = store(tmp_path)
    marker = tmp_path / "should-not-exist"
    payload = ("cos\nsystem\n(S'touch " + str(marker) + "'\ntR.").encode()
    (root / "models/best.pt").write_bytes(payload)
    data["required_hashed_assets"][0]["sha256"] = hashlib.sha256(payload).hexdigest()
    path.write_text(json.dumps(data))
    assert v.verify_assets(path, root=root)["status"] == "MATCHED"
    assert not marker.exists()


def test_cli_exit_codes_and_json_output(tmp_path, capsys):
    path, data = manifest(tmp_path)
    root = store(tmp_path)
    assert v.main(["--manifest", str(path), "--root", str(root)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "MATCHED"
    (root / "basis.npz").unlink()
    assert v.main(["--manifest", str(path), "--root", str(root)]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "INCOMPLETE"
    with pytest.raises(SystemExit) as error:
        v.main(["--manifest", str(path)])
    assert error.value.code == 2


def test_published_manifest_can_report_missing_without_remote_access(tmp_path):
    report = v.verify_assets(root=tmp_path / "not-uploaded")
    assert report["counts"]["required"]["total"] == 21
    assert report["counts"]["required"]["missing"] == 21
    assert report["status"] == "INCOMPLETE"
    assert all(str(tmp_path) in r["local_path"] for r in report["assets"])
