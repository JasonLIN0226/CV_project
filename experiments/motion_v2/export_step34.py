"""Terminal-only immutable Motion step3/4 evidence export, without computation.

Only byte hashing, JSON/text validation and ZIP packaging occur. There are no
Slurm calls, model imports, array decoding, deletions, or original-file writes.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import zipfile


ROOT = Path("/scratch/xy3257/motion-v2-20261101-step34-deployment1")
SCIENCE_RELATIVE = "repo/results/motion_v2/step3_4/gpu"
MAX_ATTEMPTS = 3
ACCOUNTING_FIELDS = ("JobIDRaw", "JobName", "User", "Account", "Partition", "State",
                     "ExitCode", "Start", "End", "ElapsedRaw", "AllocTRES", "ReqTRES")
STUDENT_LABELS = ("local_feature", "local_position", "use_time_feature", "use_time_position", "future_attention",
                  "shuffle_local_feature", "shuffle_local_position", "shuffle_use_time_feature", "shuffle_use_time_position")
_SUBMIT = None


def require(value, message):
    if not value:
        raise ValueError(message)


def relative_name(value):
    """Canonical relative POSIX names are both manifest keys and ZIP names."""
    require(isinstance(value, str) and value and "\\" not in value and "\x00" not in value,
            "Invalid relative artifact name")
    path = PurePosixPath(value)
    require(bool(path.parts) and not path.is_absolute() and all(part not in (".", "..") for part in path.parts)
            and path.as_posix() == value and ":" not in value, "Unsafe/noncanonical relative artifact name")
    return value


def safe_path(path, root, *, exists=True):
    path, root = Path(path).absolute(), Path(root).absolute()
    require(".." not in path.parts and path.is_relative_to(root), "Path escapes the export source root")
    for component in (path, *path.parents):
        require(not component.is_symlink(), "Symlink files/directories are forbidden: " + str(component))
    if exists:
        require(path.exists(), "Required path is missing: " + str(path))
    return path


def file_record(path, root):
    path = safe_path(path, root)
    before = path.stat()
    require(stat.S_ISREG(before.st_mode), "Only regular files may be exported")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    require(identity(before) == identity(after), "Source changed while hashing: " + str(path))
    return {"path": relative_name(path.relative_to(Path(root).absolute()).as_posix()),
            "original_absolute_path": str(path), "bytes": after.st_size, "sha256": digest.hexdigest()}


def read_json(path, root):
    return json.loads(safe_path(path, root).read_text(encoding="utf-8"))


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, ensure_ascii=False, allow_nan=False)
        output.write("\n")


def walk_regular_files(directory, root):
    directory = safe_path(directory, root)
    require(directory.is_dir(), "Expected evidence directory")
    result = []
    for current, directories, files in os.walk(directory, followlinks=False):
        for name in directories:
            safe_path(Path(current) / name, root)
        for name in files:
            path = safe_path(Path(current) / name, root)
            require(stat.S_ISREG(path.stat().st_mode), "Special files cannot be evidence")
            result.append(path)
    return sorted(result)


def verify_checksums(path, root):
    """Verify every listed member; return exact path-to-byte-SHA membership."""
    path = safe_path(path, root)
    hashes = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        require("  " in line, "Malformed SHA256SUMS line")
        checksum, name = line.split("  ", 1)
        name = relative_name(name)
        require(re.fullmatch(r"[0-9a-f]{64}", checksum) and name not in hashes, "Invalid or duplicate checksum member")
        member = safe_path(path.parent / name, root)
        require(file_record(member, root)["sha256"] == checksum, "Checksum mismatch: " + str(member))
        hashes[name] = checksum
    require(hashes, "Empty checksum inventory")
    return hashes


def verify_receipt(directory, name, root):
    directory = safe_path(directory, root)
    hashes = verify_checksums(directory / "SHA256SUMS", root)
    require({name, "status.json"} <= hashes.keys(), "Receipt/status missing from immutable checksum inventory")
    receipt = read_json(directory / name, root)
    status_value = read_json(directory / "status.json", root)
    require(status_value.get("receipt_sha256") == hashes[name], "Status does not bind this receipt")
    return receipt, status_value, hashes


def _submit():
    global _SUBMIT
    if _SUBMIT is None:
        path = Path(__file__).with_name("submit_step34.py")
        spec = importlib.util.spec_from_file_location("_motion_step34_export_native_parsers", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _SUBMIT = module
    return _SUBMIT


def native_context():
    return {"owner": "xy3257", "account": "csci_ga_2271-2026fa", "partition": "c12m85-a100-1",
            "time_limit_seconds": 10500, "hard_cap_seconds": 14400,
            "wrapper_path": str(ROOT / "repo/experiments/motion_v2/run_step34.sbatch"),
            "workdir": str(ROOT / "repo")}


def validate_terminal_evidence(receipt, status_value, sacct_text, scontrol_text, job_id, attempt):
    """Validate saved native records only; never query an active or terminal job."""
    require(type(attempt) is int and attempt in (1, 2, 3), "Invalid attempt")
    require(isinstance(job_id, str) and re.fullmatch(r"[0-9]+", job_id), "Invalid submitted job identity")
    require(status_value.get("status") == "passed" and receipt.get("mode") == "settle"
            and receipt.get("attempt") == attempt and receipt.get("job_id") == job_id,
            "A passed matching terminal settlement is required")
    commands = receipt.get("commands", [])
    for name, raw, argv in (
        ("final_sacct_duplicates", sacct_text,
         ["sacct", "--duplicates", "-j", job_id, "--noheader", "--parsable2", "--format=" + ",".join(ACCOUNTING_FIELDS)]),
        ("final_scontrol", scontrol_text, ["scontrol", "show", "job", "--oneliner", job_id]),
    ):
        matches = [value for value in commands if value.get("name") == name]
        require(len(matches) == 1 and matches[0].get("returncode") == 0
                and matches[0].get("argv") == argv and matches[0].get("stdout") == raw,
                "Saved native stdout/command binding differs: " + name)
    parsed = _submit().parse_complete_accounting(sacct_text, job_id)
    terminal = _submit().validate_terminal_accounting(parsed["allocation_row"], scontrol_text, native_context())
    require(receipt.get("complete_accounting") == parsed and receipt.get("terminal_accounting") == terminal,
            "Structured settlement differs from its saved native records")
    require(receipt.get("terminal_scontrol") == scontrol_text
            and receipt.get("terminal_scontrol_sha256") == hashlib.sha256(scontrol_text.encode()).hexdigest(),
            "Terminal native control hash differs")
    require(status_value.get("native_state") == terminal["native_state"] and status_value.get("restarts") == 0
            and status_value.get("actual_a100_gpu_seconds") == terminal["actual_a100_gpu_seconds"],
            "Settlement status differs from native accounting")
    require(receipt.get("cumulative_a100_seconds") == status_value.get("cumulative_a100_seconds")
            and type(receipt.get("cumulative_a100_seconds")) is int
            and 0 <= receipt["cumulative_a100_seconds"] <= 14400, "Cumulative accounting is missing or outside budget")
    return {"attempt": attempt, "job_id": job_id, "native_state": terminal["native_state"],
            "restarts": 0, "native_terminal_verified": True,
            "actual_a100_gpu_seconds": terminal["actual_a100_gpu_seconds"],
            "cumulative_a100_seconds": receipt["cumulative_a100_seconds"],
            "native_commands_executed_by_export": 0, "scientific_success_inferred": False}


def terminal_history(root, attempt):
    root = Path(root).absolute()
    require(type(attempt) is int and attempt in (1, 2, 3), "Invalid selected attempt")
    for future in range(attempt + 1, MAX_ATTEMPTS + 1):
        path = safe_path(root / "runtime" / f"attempt{future}" / "submission", root, exists=False)
        require(not path.exists(), "A later submission exists; export only the latest settled attempt")
    history = []
    for number in range(1, attempt + 1):
        directory = root / "runtime" / f"attempt{number}"
        submitted, _, sub_hashes = verify_receipt(directory / "submission", "submission.json", root)
        require(submitted.get("sbatch_attempted") is True and submitted.get("attempt") == number,
                "Incomplete or ambiguous submission outcome")
        job_id = submitted.get("job_id")
        settled, status_value, _ = verify_receipt(directory / "settlement", "settlement.json", root)
        require(settled.get("submission_sha256") == sub_hashes["submission.json"]
                and settled.get("registration") == submitted.get("registration"), "Submission/settlement identity chain differs")
        sacct = safe_path(directory / "settlement/final_sacct_duplicates.stdout.txt", root).read_text(encoding="utf-8")
        control = safe_path(directory / "settlement/final_scontrol.stdout.txt", root).read_text(encoding="utf-8")
        value = validate_terminal_evidence(settled, status_value, sacct, control, job_id, number)
        value["submission_sha256"] = sub_hashes["submission.json"]
        value["settlement_sha256"] = file_record(directory / "settlement/settlement.json", root)["sha256"]
        history.append(value)
    return history


def omit_encoded_features(relative):
    """The sole omitted scientific class is this exact newly encoded subtree."""
    path = PurePosixPath(relative_name(relative))
    return (len(path.parts) == 3 and path.parts[:2] == ("longgap_features", "episodes")
            and re.fullmatch(r"[0-9a-f]{24}\.npz", path.name) is not None)


def scientific_inventory(science_root, root):
    science_root = safe_path(science_root, root, exists=False)
    if not science_root.exists():
        return {"present": False, "files": [], "included": [], "omitted": [], "artifact_map_verified": False,
                "source_status": None, "missing_expected_paths": [SCIENCE_RELATIVE + "/status.json"], "complete": False}
    files = walk_regular_files(science_root, root)
    artifact_path = science_root / "artifact_sha256.json"
    require(artifact_path.is_file(), "Existing scientific output lacks final artifact_sha256.json; cannot certify its files")
    expected = read_json(artifact_path, root)
    require(isinstance(expected, dict) and expected, "Invalid scientific artifact inventory")
    actual_names = {path.relative_to(science_root).as_posix() for path in files}
    require(set(expected) == actual_names - {"artifact_sha256.json", "SHA256SUMS"},
            "Scientific inventory has missing or unregistered files")
    records, included, omitted = [], [], []
    for path in files:
        relative = relative_name(path.relative_to(science_root).as_posix())
        record = file_record(path, root)
        if relative in expected:
            require(re.fullmatch(r"[0-9a-f]{64}", expected[relative])
                    and record["sha256"] == expected[relative], "Scientific artifact SHA mismatch: " + relative)
        if omit_encoded_features(relative):
            omitted.append({**record, "reason": "New long-gap episode encoding retained remotely; byte SHA verified, no arrays decoded"})
        else:
            included.append(record)
        records.append(record)
    checksums = [path for path in files if path.name == "SHA256SUMS"]
    require(science_root / "SHA256SUMS" in checksums, "Final scientific SHA256SUMS is missing")
    for path in checksums:
        verify_checksums(path, root)
    status_value = read_json(science_root / "status.json", root)
    required = required_scientific_paths(status_value)
    missing = [str(PurePosixPath(SCIENCE_RELATIVE) / name) for name in required if name not in actual_names]
    return {"present": True, "files": records, "included": included, "omitted": omitted,
            "artifact_map_verified": True, "artifact_map_sha256": file_record(artifact_path, root)["sha256"],
            "artifact_members_verified": len(expected), "nested_checksum_lists_verified": len(checksums),
            "source_status": status_value, "missing_expected_paths": missing,
            "complete": status_value.get("status") == "completed_step4_stop" and not missing}


def required_scientific_paths(status_value):
    required = ["status.json", "asset_reuse.json", "environment.json", "runtime/allocation_start.json",
                "runtime/native_allocation_checks.jsonl", "longgap_features/index.json", "longgap_features/encoder_metadata.json",
                "students/summary.json", "students/normalization.npz", "students/provenance.json",
                "students/supervision_audit.npz", "students/minibatch_indices.npz", "students/label_scales.json"]
    for split in ("ordinary", "long_gap"):
        required += [f"diagnostics/{split}/{name}" for name in
                     ("summary.json", "status.json", "bootstrap_indices.npz", "frame_choices.json", "raw_loss_arrays.npz")]
    for split in ("train", "validation"):
        required += [f"labels/{split}/{name}" for name in
                     ("states.json", "inputs.npz", "labels.npz", "raw_teacher_losses.npz", "continuation_endpoints.npz", "label_provenance.json")]
    for architecture in ("independent", "set"):
        for label in STUDENT_LABELS:
            for seed in (17, 29, 43):
                stem = f"students/{architecture}_{label}_seed{seed}"
                required.extend([stem + ".pt", stem + "_history.json"])
    online_splits = ["ordinary"]
    if status_value.get("long_gap_online_evaluated") is True:
        online_splits.append("long_gap")
    elif status_value.get("long_gap_online_evaluated") is False:
        required.append("online/long_gap_not_evaluated.json")
    for split in online_splits:
        required += [f"online/{split}/{name}" for name in ("summary.json", "status.json", "bootstrap_indices.npz", "trace_inventory.json")]
        methods = ["baseline_" + p for p in ("fifo", "reservoir", "temporal_buckets", "diversity", "surprise", "attention", "keep_first")]
        methods += [f"student_{arch}_{label}" for arch in ("independent", "set") for label in STUDENT_LABELS]
        methods += ["reference_last_visible", "reference_b", "reference_posthoc_best"]
        required += [f"online/{split}/raw_losses/{method}.npz" for method in methods]
    return sorted(required)


def collect_evidence(root, attempt, history, science):
    records, missing, checksum_lists = {}, [], []

    def include(path):
        record = file_record(path, root)
        records[record["path"]] = record

    for number in range(1, attempt + 1):
        runtime = safe_path(Path(root) / "runtime" / f"attempt{number}", root)
        files = walk_regular_files(runtime, root)
        for path in files:
            include(path)
            if path.name == "SHA256SUMS":
                verify_checksums(path, root)
                checksum_lists.append(path.relative_to(root).as_posix())
        for stage in ("preflight", "submission", "settlement"):
            require(f"runtime/attempt{number}/{stage}/SHA256SUMS" in checksum_lists, "Runtime stage checksum evidence missing")
    for entry in history:
        for stream in ("stdout", "stderr"):
            path = safe_path(Path(root) / f"scientific-job-{entry['job_id']}.{stream}.log", root, exists=False)
            if path.exists():
                include(path)
            else:
                missing.append(path.relative_to(root).as_posix())
    marker = safe_path(Path(root) / "scientific_job_started", root, exists=False)
    for name in ("scontrol_start.txt", "started_at_utc.txt", "git_head.txt", "runner_exit_code.txt", "finished_at_utc.txt"):
        path = safe_path(marker / name, root, exists=False)
        if path.exists():
            include(path)
        else:
            missing.append(path.relative_to(root).as_posix())
    # Root operational text logs can document deployment/preparation, never old assets.
    for path in sorted(Path(root).iterdir()):
        if path.is_file() and path.suffix.lower() in (".json", ".jsonl", ".txt", ".log", ".md", ".sh", ".sbatch"):
            include(path)
    for record in science["included"]:
        records[record["path"]] = record
    return {"files": [records[name] for name in sorted(records)], "missing_operational_paths": sorted(missing),
            "runtime_checksum_lists_verified": checksum_lists}


def build_archive(archive_path, root, records, generated):
    """Exclusive ZIP creation; re-hash every archived source while streaming it."""
    require(not Path(archive_path).exists() and not Path(archive_path).is_symlink(), "Archive already exists")
    names = [relative_name(record["path"]) for record in records]
    require(len(names) == len(set(names)) and not (set(names) & set(generated)), "Duplicate ZIP member")
    for name in generated:
        relative_name(name)
    with Path(archive_path).open("xb") as destination:
        with zipfile.ZipFile(destination, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
            for record in records:
                source_path = safe_path(Path(root) / record["path"], root)
                require(str(source_path) == record["original_absolute_path"], "Source absolute-path identity changed")
                before = file_record(source_path, root)
                require(before == record, "Source changed before ZIP copy")
                checksum, count = hashlib.sha256(), 0
                info = zipfile.ZipInfo(record["path"])
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (stat.S_IFREG | 0o444) << 16
                with source_path.open("rb") as source, archive.open(info, "w", force_zip64=True) as output:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        checksum.update(block)
                        count += len(block)
                        output.write(block)
                require(count == record["bytes"] and checksum.hexdigest() == record["sha256"], "Source changed during ZIP copy")
                require(file_record(source_path, root) == record, "Source changed after ZIP copy")
            for name, value in generated.items():
                archive.writestr(name, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def execute(attempt):
    require(sys.flags.isolated == 1 and sys.flags.dont_write_bytecode == 1, "Invoke with Python -I -B")
    root = safe_path(ROOT, ROOT)
    output = safe_path(root / f"export_step34_attempt{attempt}", root, exists=False)
    require(not output.exists(), "Export directory already exists; never overwrite or retry in place")
    history = terminal_history(root, attempt)
    output.mkdir(exist_ok=False)
    status_value = {"status": "stopped_export_error", "attempt": attempt, "job_id": history[-1]["job_id"],
                    "scientific_success_inferred": False, "scientific_computation_performed": False,
                    "original_files_modified": False, "file_deletion": False,
                    "started_utc": datetime.now(timezone.utc).isoformat()}
    code = 2
    try:
        science = scientific_inventory(root / SCIENCE_RELATIVE, root)
        evidence = collect_evidence(root, attempt, history, science)
        inventory = {key: value for key, value in science.items() if key not in ("included",)}
        manifest = {"schema_version": 1, "source_root": str(root), "attempt": attempt, "job_id": history[-1]["job_id"],
                    "terminal_history": history, "files": evidence["files"], "omitted": science["omitted"],
                    "missing_expected_scientific_paths": science["missing_expected_paths"],
                    "missing_operational_paths": evidence["missing_operational_paths"],
                    "runtime_checksum_lists_verified": evidence["runtime_checksum_lists_verified"],
                    "scientific_complete": science["complete"], "scientific_source_status": science["source_status"],
                    "array_decoding": False, "old_assets_included": False, "original_statuses_changed": False,
                    "omission_policy": "Only new longgap_features/episodes/*.npz; all are byte-hash verified"}
        write_new(output / "scientific_file_inventory.json", inventory)
        write_new(output / "export_manifest.json", manifest)
        archive = output / f"motion-step34-attempt{attempt}-job{history[-1]['job_id']}-evidence.zip"
        build_archive(archive, root, evidence["files"], {"export_manifest.json": manifest, "scientific_file_inventory.json": inventory})
        require(terminal_history(root, attempt) == history, "Submission/settlement history changed during export")
        after_science = scientific_inventory(root / SCIENCE_RELATIVE, root)
        require(after_science == science, "Scientific source changed during export")
        after_evidence = collect_evidence(root, attempt, history, after_science)
        require(after_evidence == evidence, "Runtime/marker/log source changed during export")
        complete = science["complete"] and not evidence["missing_operational_paths"]
        status_value.update(status="passed" if complete else "exported_incomplete_science_or_evidence",
                            evidence_exported=True, scientific_complete=science["complete"],
                            archive=file_record(archive, output), exported_files=len(evidence["files"]),
                            scientific_files_hashed=len(science["files"]), encoded_feature_files_omitted=len(science["omitted"]),
                            missing_expected_scientific_paths=science["missing_expected_paths"],
                            missing_operational_paths=evidence["missing_operational_paths"])
        code = 0
    except Exception as error:
        status_value.update(error_type=type(error).__name__, error=str(error), evidence_exported=False)
    status_value["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_new(output / "status.json", status_value)
    with (output / "SHA256SUMS").open("x", encoding="utf-8") as checksums:
        for path in sorted(output.iterdir()):
            if path.name != "SHA256SUMS" and path.is_file():
                checksums.write(file_record(path, output)["sha256"] + "  " + path.name + "\n")
    print(json.dumps(status_value, ensure_ascii=False, allow_nan=False))
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=int, choices=(1, 2, 3), required=True)
    return execute(parser.parse_args(argv).attempt)


if __name__ == "__main__":
    raise SystemExit(main())
