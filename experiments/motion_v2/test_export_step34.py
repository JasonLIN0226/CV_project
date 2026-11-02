"""Synthetic byte/receipt tests only; never run main or access a real job/data."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest


_SPEC = importlib.util.spec_from_file_location("motion_step34_export_tests", Path(__file__).with_name("export_step34.py"))
exporter = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(exporter)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def terminal_fixture(state="COMPLETED"):
    job_id = "90001"
    context = exporter.native_context()
    row = {"JobIDRaw": job_id, "JobName": "motion-v2-step34", "User": "xy3257", "Account": context["account"],
           "Partition": context["partition"], "State": state, "ExitCode": "0:0", "Start": "2026-11-01T01:00:00",
           "End": "2026-11-01T01:01:00", "ElapsedRaw": "60", "AllocTRES": "cpu=4,mem=16G,node=1,gres/gpu=1",
           "ReqTRES": "cpu=4,mem=16G,node=1,gres/gpu=1"}
    control = (f"JobId={job_id} JobState={state} Restarts=0 Account={context['account']} Partition={context['partition']} "
               f"NumNodes=1 TimeLimit=02:55:00 UserId=xy3257(1234) Command={context['wrapper_path']} WorkDir={context['workdir']} "
               "ReqTRES=cpu=4,mem=16G,node=1,gres/gpu=1 AllocTRES=cpu=4,mem=16G,node=1,gres/gpu=1 "
               "RunTime=00:01:00 NodeList=synthetic-node StartTime=2026-11-01T01:00:00 EndTime=2026-11-01T01:01:00")
    raw = "|".join(row[key] for key in exporter.ACCOUNTING_FIELDS) + "\n"
    parsed = exporter._submit().parse_complete_accounting(raw, job_id)
    terminal = exporter._submit().validate_terminal_accounting(row, control, context)
    receipt = {"mode": "settle", "attempt": 1, "job_id": job_id, "complete_accounting": parsed,
               "terminal_accounting": terminal, "terminal_scontrol": control, "terminal_scontrol_sha256": sha(control.encode()),
               "cumulative_a100_seconds": 3765,
               "commands": [{"name": "final_sacct_duplicates", "argv": ["sacct", "--duplicates", "-j", job_id,
                              "--noheader", "--parsable2", "--format=" + ",".join(exporter.ACCOUNTING_FIELDS)],
                             "returncode": 0, "stdout": raw},
                            {"name": "final_scontrol", "argv": ["scontrol", "show", "job", "--oneliner", job_id],
                             "returncode": 0, "stdout": control}]}
    status = {"status": "passed", "native_state": state, "restarts": 0, "actual_a100_gpu_seconds": 60,
              "cumulative_a100_seconds": 3765}
    return receipt, status, raw, control, job_id


@pytest.mark.parametrize("value", ["../outside", "/absolute", "a/../b", "a//b", "./a", ".", "a\\b", "C:/x", "", "a\x00b"])
def test_archive_path_traversal_and_aliases_rejected(value):
    with pytest.raises(ValueError):
        exporter.relative_name(value)


def test_known_hash_and_checksum_tampering(tmp_path):
    write(tmp_path / "result.json", b'{"value": 1}\n')
    checksum = sha(b'{"value": 1}\n')
    write(tmp_path / "SHA256SUMS", (checksum + "  result.json\n").encode())
    record = exporter.file_record(tmp_path / "result.json", tmp_path)
    assert record["sha256"] == checksum and record["bytes"] == 13
    assert exporter.verify_checksums(tmp_path / "SHA256SUMS", tmp_path) == {"result.json": checksum}
    (tmp_path / "result.json").write_bytes(b'{"value": 2}\n')
    with pytest.raises(ValueError, match="Checksum mismatch"):
        exporter.verify_checksums(tmp_path / "SHA256SUMS", tmp_path)


def test_file_and_directory_symlinks_rejected(tmp_path):
    write(tmp_path / "actual/file.txt", b"same bytes")
    (tmp_path / "linked-file").symlink_to(tmp_path / "actual/file.txt")
    (tmp_path / "linked-directory").symlink_to(tmp_path / "actual", target_is_directory=True)
    for path in (tmp_path / "linked-file", tmp_path / "linked-directory/file.txt"):
        with pytest.raises(ValueError, match="Symlink"):
            exporter.file_record(path, tmp_path)


def test_only_new_episode_encoding_is_omitted():
    assert exporter.omit_encoded_features("longgap_features/episodes/" + "a" * 24 + ".npz")
    for name in ("longgap_features/index.json", "students/independent_local_feature_seed17.pt",
                 "students/supervision_audit.npz", "students/minibatch_indices.npz", "labels/train/inputs.npz",
                 "online/ordinary/raw_losses/baseline_fifo.npz", "diagnostics/long_gap/raw_loss_arrays.npz",
                 "online/ordinary/bootstrap_indices.npz", "other/episodes/" + "a" * 24 + ".npz"):
        assert not exporter.omit_encoded_features(name)


def test_exclusive_zip_members_have_manifest_hash_and_original_path(tmp_path):
    source = tmp_path / "source"
    write(source / "students/example.pt", b"opaque checkpoint; do not deserialize")
    record = exporter.file_record(source / "students/example.pt", source)
    archive = tmp_path / "evidence.zip"
    exporter.build_archive(archive, source, [record], {"export_manifest.json": {"files": [record]}})
    with zipfile.ZipFile(archive) as zipped:
        assert set(zipped.namelist()) == {"students/example.pt", "export_manifest.json"}
        assert sha(zipped.read("students/example.pt")) == record["sha256"]
        inner = json.loads(zipped.read("export_manifest.json"))
        assert inner["files"][0]["original_absolute_path"] == str(source / "students/example.pt")
    original = archive.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        exporter.build_archive(archive, source, [record], {})
    assert archive.read_bytes() == original


def test_zip_rejects_changed_or_unsafe_source_inventory(tmp_path):
    write(tmp_path / "source/result.txt", b"original")
    record = exporter.file_record(tmp_path / "source/result.txt", tmp_path / "source")
    (tmp_path / "source/result.txt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed before"):
        exporter.build_archive(tmp_path / "changed.zip", tmp_path / "source", [record], {})
    with pytest.raises(ValueError):
        exporter.build_archive(tmp_path / "unsafe.zip", tmp_path / "source", [dict(record, path="../escape")], {})


def test_terminal_native_receipt_is_accepted_without_commands():
    receipt, status, raw, control, job_id = terminal_fixture()
    result = exporter.validate_terminal_evidence(receipt, status, raw, control, job_id, 1)
    assert result["native_terminal_verified"] is True
    assert result["native_commands_executed_by_export"] == 0
    assert result["actual_a100_gpu_seconds"] == 60
    assert result["scientific_success_inferred"] is False


@pytest.mark.parametrize("state", ["FAILED", "TIMEOUT", "CANCELLED"])
def test_failed_native_state_is_preserved_after_valid_accounting(state):
    receipt, status, raw, control, job_id = terminal_fixture(state)
    result = exporter.validate_terminal_evidence(receipt, status, raw, control, job_id, 1)
    assert result["native_state"] == state and result["native_terminal_verified"] is True
    assert result["scientific_success_inferred"] is False


def test_later_submission_prevents_export_of_an_older_attempt(tmp_path):
    (tmp_path / "runtime/attempt2/submission").mkdir(parents=True)
    with pytest.raises(ValueError, match="later submission"):
        exporter.terminal_history(tmp_path, 1)


@pytest.mark.parametrize("change", ["pending", "failed_settlement", "stdout_mismatch", "wrong_owner", "restarts", "wrong_job"])
def test_nonterminal_or_unbound_native_evidence_is_rejected(change):
    receipt, status, raw, control, job_id = terminal_fixture()
    receipt, status = copy.deepcopy(receipt), dict(status)
    if change == "pending":
        raw = raw.replace("COMPLETED", "PENDING")
        receipt["commands"][0]["stdout"] = raw
    elif change == "failed_settlement":
        status["status"] = "stopped_accounting_error"
    elif change == "stdout_mismatch":
        raw += "\n"
    elif change == "wrong_owner":
        raw = raw.replace("xy3257", "other_account")
        receipt["commands"][0]["stdout"] = raw
    elif change == "restarts":
        control = control.replace("Restarts=0", "Restarts=1")
        receipt["commands"][1]["stdout"] = control
    else:
        job_id = "90002"
    with pytest.raises((ValueError, RuntimeError)):
        exporter.validate_terminal_evidence(receipt, status, raw, control, job_id, 1)


def make_scientific_fixture(root):
    science = root / exporter.SCIENCE_RELATIVE
    write(science / "status.json", b'{"status":"stopped"}\n')
    write(science / "students/example.pt", b"opaque checkpoint")
    write(science / ("longgap_features/episodes/" + "a" * 24 + ".npz"), b"opaque compressed encoding")
    paths = sorted(p for p in science.rglob("*") if p.is_file())
    mapping = {p.relative_to(science).as_posix(): sha(p.read_bytes()) for p in paths}
    write(science / "artifact_sha256.json", json.dumps(mapping).encode())
    mapping["artifact_sha256.json"] = sha((science / "artifact_sha256.json").read_bytes())
    write(science / "SHA256SUMS", "".join(checksum + "  " + name + "\n" for name, checksum in mapping.items()).encode())
    return science


def test_failed_science_can_export_existing_files_but_is_never_complete(tmp_path):
    science = make_scientific_fixture(tmp_path)
    result = exporter.scientific_inventory(science, tmp_path)
    assert result["artifact_map_verified"] is True and result["artifact_members_verified"] == 3
    assert result["complete"] is False and result["source_status"] == {"status": "stopped"}
    assert result["missing_expected_paths"]
    assert len(result["omitted"]) == 1
    assert any(row["path"].endswith("students/example.pt") for row in result["included"])


def test_even_omitted_feature_hashes_are_verified(tmp_path):
    science = make_scientific_fixture(tmp_path)
    encoded = science / ("longgap_features/episodes/" + "a" * 24 + ".npz")
    encoded.write_bytes(b"tampered encoded data")
    with pytest.raises(ValueError, match="Scientific artifact SHA mismatch"):
        exporter.scientific_inventory(science, tmp_path)


def test_missing_registered_file_cannot_be_relabelled_as_partial_output(tmp_path):
    science = make_scientific_fixture(tmp_path)
    mapping = json.loads((science / "artifact_sha256.json").read_text())
    mapping["missing.json"] = "0" * 64
    (science / "artifact_sha256.json").write_text(json.dumps(mapping))
    with pytest.raises(ValueError, match="missing or unregistered"):
        exporter.scientific_inventory(science, tmp_path)
