"""Historical receipt verification must not certify a newer source revision."""
import hashlib
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "verify_phase2_history_source_snapshots",
    Path(__file__).resolve().parents[1] / "tools/verify_phase2_history.py")
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


@pytest.fixture
def source_tree(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    snapshots = root / "results/phase2_history/source_snapshots"
    snapshots.mkdir(parents=True)
    monkeypatch.setattr(v, "ROOT", root)
    monkeypatch.setattr(v, "SOURCE_SNAPSHOTS", snapshots)
    relative = "tests/example.py"
    current = root / relative
    current.parent.mkdir()
    historical = b"raise AssertionError('archive must never execute')\n"
    expected = hashlib.sha256(historical).hexdigest()
    return root, snapshots, relative, current, historical, expected


def test_matching_current_source_needs_no_archive(source_tree):
    _, snapshots, relative, current, historical, expected = source_tree
    current.write_bytes(historical)
    # A stale or unrelated archive cannot override a matching working source.
    (snapshots / f"{expected}.py").write_bytes(b"wrong archive\n")
    archived = {}
    v.verify_source(relative, expected, archived)
    assert archived == {}


def test_fallback_reports_historical_and_current_identities(source_tree):
    root, snapshots, relative, current, historical, expected = source_tree
    current.write_bytes(b"# later working revision\n")
    snapshot = snapshots / f"{expected}.py"
    snapshot.write_bytes(historical)
    archived = {}
    v.verify_source(relative, expected, archived)
    assert archived == {relative: {
        "sha256": expected,
        "snapshot": str(snapshot.relative_to(root)),
        "current_working_sha256": hashlib.sha256(current.read_bytes()).hexdigest(),
        "current_working_source_matches_historical_receipt": False,
    }}
    # Deleted working files can still have verifiable historical provenance.
    current.unlink()
    v.verify_source(relative, expected, archived)
    assert archived[relative]["current_working_sha256"] is None


def test_missing_tampered_or_escaping_source_snapshots_fail_closed(source_tree):
    root, snapshots, relative, current, historical, expected = source_tree
    current.write_bytes(b"# later working revision\n")
    archived = {}
    snapshot = snapshots / f"{expected}.py"
    with pytest.raises(ValueError, match="Historical source missing or changed"):
        v.verify_source(relative, expected, archived)
    snapshot.write_bytes(b"# modified historical source\n")
    with pytest.raises(ValueError, match="Historical source missing or changed"):
        v.verify_source(relative, expected, archived)
    snapshot.unlink()
    outside_archive = root / "unregistered.py"
    outside_archive.write_bytes(historical)
    snapshot.symlink_to(outside_archive)
    with pytest.raises(ValueError, match="[Ee]scapes"):
        v.verify_source(relative, expected, archived)
    with pytest.raises(ValueError, match="Source path escapes"):
        v.verify_source("../outside.py", expected, archived)
    with pytest.raises(ValueError, match="Invalid source digest"):
        v.verify_source(relative, "../invalid", archived)
    assert archived == {}
