"""Export existing job-16550 evidence after native terminal accounting.

Standalone stdlib-only packaging: no model imports, NPZ decoding, scientific
calculation, subprocesses, deletion, overwrite, or edits to original receipts.
The sole write destination is a previously nonexistent deployment3/export1.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import traceback
import zipfile


sys.dont_write_bytecode = True
ROOT = Path("/scratch/xy3257/motion-v2-20261031-deployment3")
JOB_ID = "16550"
EXPORT = ROOT / "export1"
SCIENCE = ROOT / "repo/results/motion_v2/v2_3_1/step2_gpu_attempt1"
BOUND = ROOT / "repo/results/motion_v2/v2_3/direction_bound"
ARCHIVE_NAME = "motion-v2_3_1-job16550-evidence.zip"
TERMINAL = {"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL", "PREEMPTED",
            "BOOT_FAIL", "DEADLINE", "OUT_OF_MEMORY", "REVOKED", "SPECIAL_EXIT"}
ACCOUNTING_COLUMNS = ("JobIDRaw", "JobName", "User", "Account", "Partition", "State",
                      "ExitCode", "Start", "End", "ElapsedRaw", "AllocTRES", "ReqTRES")
MAX_SMALL_FILE_BYTES = 32 * 1024 * 1024
_HASH_CACHE = {}


def require(value, message):
    if not value:
        raise ValueError(message)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def root_relative(path):
    return Path(path).absolute().relative_to(ROOT).as_posix()


def safe(path):
    path = Path(path).absolute()
    require(path.is_relative_to(ROOT), "Path lies outside deployment3: " + str(path))
    require(path.resolve(strict=False).is_relative_to(ROOT), "Symlink escapes deployment3: " + str(path))
    return path


def regular(path):
    path = safe(path)
    require(path.is_file(), "Required regular file is absent: " + str(path))
    resolved = path.resolve(strict=True)
    require(stat.S_ISREG(resolved.stat().st_mode), "Expected regular file: " + str(path))
    return resolved


def fingerprint(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def file_hash(path):
    resolved = regular(path)
    initial = resolved.stat()
    key = (str(resolved), fingerprint(initial))
    if key in _HASH_CACHE:
        return _HASH_CACHE[key]
    checksum = hashlib.sha256()
    with resolved.open("rb") as source:
        before = os.fstat(source.fileno())
        require(fingerprint(before) == fingerprint(initial), "File changed before hashing: " + str(path))
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(chunk)
        after = os.fstat(source.fileno())
    require(fingerprint(before) == fingerprint(after) == fingerprint(resolved.stat()),
            "File changed while hashing: " + str(path))
    _HASH_CACHE[key] = checksum.hexdigest()
    return checksum.hexdigest()


def file_record(path):
    path = safe(path)
    return {"path": root_relative(path), "bytes": regular(path).stat().st_size,
            "sha256": file_hash(path)}


def read_json(path):
    with regular(path).open(encoding="utf-8") as source:
        return json.load(source)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as destination:
        json.dump(value, destination, indent=2, ensure_ascii=False, allow_nan=False)
        destination.write("\n")


def verify_checksum_list(directory):
    inventory = safe(directory / "SHA256SUMS")
    if not inventory.exists():
        return {"present": False, "files_verified": 0}
    seen = set()
    for line in regular(inventory).read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        require(re.fullmatch(r"[0-9a-f]{64}", expected) is not None
                and Path(name).name == name and name not in seen,
                "Malformed or duplicate receipt checksum entry")
        seen.add(name)
        require(file_hash(directory / name) == expected, "Original receipt checksum mismatch: " + name)
    return {"present": True, "files_verified": len(seen), "sha256": file_hash(inventory)}


def terminal_accounting():
    """Validate stored native terminal evidence; do not query or alter Slurm."""
    folder = ROOT / "settlement"
    accounting_path = folder / "accounting.json"
    accounting = read_json(accounting_path)
    require(accounting.get("job_id") == JOB_ID, "Settlement belongs to another job")
    native = accounting["sacct"]
    require(native.get("returncode") == 0, "No successful native sacct record in settlement")
    raw = regular(folder / "final_sacct.stdout.txt").read_text(encoding="utf-8")
    require(raw == native["stdout"], "Stored sacct text differs from the accounting receipt")
    selected = []
    for line in raw.splitlines():
        cells = line.split("|")
        if len(cells) == len(ACCOUNTING_COLUMNS) + 1 and cells[-1] == "":
            cells.pop()
        if cells and cells[0].strip() == JOB_ID:
            require(len(cells) == len(ACCOUNTING_COLUMNS), "Malformed native top-level sacct row")
            selected.append(dict(zip(ACCOUNTING_COLUMNS, [cell.strip() for cell in cells])))
    require(len(selected) == 1, "Expected exactly one native top-level accounting row")
    row = selected[0]
    require(row == accounting["allocation_row"], "Native row differs from saved parsed accounting")
    native_state = row["State"].split()[0].rstrip("+") if row["State"].strip() else ""
    require(native_state in TERMINAL, "Scientific job is not confirmed terminal; export refused")
    require(row["End"] not in {"", "Unknown", "None"}, "Terminal native end time is missing")
    require(row["User"] == "xy3257" and row["Account"] == "csci_ga_2271-2026fa"
            and row["Partition"] == "c12m85-a100-1", "Unexpected native job identity")
    status_path = folder / "status.json"
    status = read_json(status_path) if status_path.is_file() else None
    if status is not None:
        require(status.get("job_id") == JOB_ID and status.get("receipt_sha256") == file_hash(accounting_path),
                "Settlement status does not bind the original accounting receipt")
    checks = verify_checksum_list(folder)
    return {"native_terminal_verified": True, "job_id": JOB_ID, "native_state": native_state,
            "native_allocation_row": row, "settlement_original_status": None if status is None else status.get("status"),
            "settlement_status_present": status is not None,
            "settlement_receipt_sha256": file_hash(accounting_path), "settlement_checksums": checks,
            "failure_status_rewritten": False, "native_commands_executed_by_export": 0}


def link_record(path):
    path = safe(path)
    require(path.is_symlink(), "Expected a symbolic link")
    target = path.resolve(strict=False)
    require(target.is_relative_to(ROOT), "Symlink target escapes deployment3")
    return {"path": root_relative(path), "target": os.readlink(path),
            "resolved_target": root_relative(target), "target_exists": target.exists()}


def science_inventory():
    files, links = [], []
    safe(SCIENCE)
    require(not SCIENCE.is_symlink(), "Scientific output root must not be a symlink")
    if not SCIENCE.exists():
        return {"scientific_output_present": False, "regular_files": files, "symlinks": links}
    require(not SCIENCE.is_symlink() and SCIENCE.is_dir(), "Scientific output root is not a real directory")
    for parent, directories, names in os.walk(SCIENCE, followlinks=False):
        parent = safe(parent)
        descend = []
        for name in sorted(directories):
            path = safe(parent / name)
            if path.is_symlink():
                links.append(link_record(path))
            else:
                require(stat.S_ISDIR(path.lstat().st_mode), "Unexpected non-directory in scientific tree")
                descend.append(name)
        directories[:] = descend
        for name in sorted(names):
            path = safe(parent / name)
            if path.is_symlink():
                links.append(link_record(path))
            else:
                require(stat.S_ISREG(path.lstat().st_mode), "Refusing special file in scientific output: " + str(path))
                files.append(file_record(path))
    return {"scientific_output_present": True, "regular_files": sorted(files, key=lambda item: item["path"]),
            "symlinks": sorted(links, key=lambda item: item["path"]), "symlink_directories_followed": False}


def verify_original_runner_inventory():
    path = SCIENCE / "artifact_sha256.json"
    if not path.exists():
        return {"present": False, "files_verified": 0,
                "note": "Original runner inventory absent at the recorded stopping stage; not recreated or overwritten"}
    mapping = read_json(path)
    require(isinstance(mapping, dict), "Original runner SHA256 inventory is not a mapping")
    for name, expected in mapping.items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts
                and re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
                "Unsafe original runner SHA256 inventory entry")
        require(file_hash(SCIENCE / relative) == expected,
                "Original runner artifact SHA256 mismatch: " + name)
    return {"present": True, "files_verified": len(mapping), "sha256": file_hash(path),
            "original_inventory_modified": False}


def select_small_evidence(inventory):
    selected, missing, excluded, selected_links = {}, [], [], []

    def add(path, expected=True):
        path = safe(path)
        if not path.exists() and not path.is_symlink():
            if expected:
                missing.append(root_relative(path))
            return
        if path.is_symlink():
            selected_links.append(link_record(path))
            excluded.append({"path": root_relative(path), "reason": "symlink recorded separately, not dereferenced into archive"})
            return
        require(path.is_file(), "Expected evidence file, found another object: " + str(path))
        record = file_record(path)
        if record["bytes"] > MAX_SMALL_FILE_BYTES:
            excluded.append({**record, "reason": "larger than the 32 MiB small-file packaging ceiling; preserved remotely"})
            return
        selected[record["path"]] = record

    def add_tree(folder):
        folder = safe(folder)
        if not folder.exists():
            missing.append(root_relative(folder) + "/")
            return
        require(folder.is_dir() and not folder.is_symlink(), "Evidence directory must not be a symlink")
        for parent, directories, names in os.walk(folder, followlinks=False):
            retain = []
            for name in sorted(directories):
                path = safe(Path(parent) / name)
                if path.is_symlink():
                    selected_links.append(link_record(path))
                else:
                    retain.append(name)
            directories[:] = retain
            for name in sorted(names):
                add(Path(parent) / name)

    for name in ("receipt.json", "status.json", "SHA256SUMS"):
        add(ROOT / "preflight" / name)
    for folder in (ROOT / "submission", ROOT / "settlement"):
        add_tree(folder)
    for name in ("scontrol_start.txt", "started_at_utc.txt", "git_head.txt",
                 "runner_exit_code.txt", "finished_at_utc.txt"):
        add(ROOT / "scientific_job_started" / name)
    for name in ("preflight-command.log", "unpack-command.log", "submit-command.log", "settlement-command.log",
                 "unpack_receipt.json", "auxiliary_job_initial.txt", "a100_partition.txt", "reused_archive_sha256.txt",
                 f"scientific-job-{JOB_ID}.stdout.log", f"scientific-job-{JOB_ID}.stderr.log"):
        add(ROOT / name)
    for pattern in ("*.log", "*.stdout.txt", "*.stderr.txt", "*-command*.txt"):
        for path in sorted(ROOT.glob(pattern)):
            add(path, expected=False)
    for name in ("direction_bound.npz", "status.json", "artifact_sha256.json"):
        add(BOUND / name)
    expected_root = ("resolved_config.json", "registration_verification.json", "allocation_start.json",
                     "native_allocation_checks.jsonl", "runtime_environment.json", "encoder_metadata.json",
                     "direction_bound_verification.json", "input_cache_preparation.json",
                     "control_input_identity_audit.json", "status.json", "artifact_sha256.json")
    for name in expected_root:
        add(SCIENCE / name)
    if SCIENCE.exists():
        for pattern in ("*.json", "*.jsonl"):
            for path in sorted(SCIENCE.glob(pattern)):
                add(path, expected=False)
    for folder in (SCIENCE / "metrics", SCIENCE / "decoder"):
        add_tree(folder)
    add(SCIENCE / "validation_losses.npz")
    for name in ("index.json", "normalization.npz", "normalization_provenance.json"):
        add(SCIENCE / "features" / name)
    add(SCIENCE / "predictor/motion/basis.json")
    for seed in (17, 29, 43):
        for name in ("training.jsonl", "selection.json"):
            add(SCIENCE / "predictor/motion" / f"seed_{seed}" / name)
    for record in inventory["regular_files"]:
        if record["path"] not in selected and not any(item["path"] == record["path"] for item in excluded):
            excluded.append({**record, "reason": "scientific file retained remotely; outside the small-evidence allowlist"})
    return {"files": sorted(selected.values(), key=lambda item: item["path"]),
            "missing_expected_paths": sorted(set(missing)), "excluded_from_archive": excluded,
            "selected_symlinks_recorded_only": selected_links}


def copy_zip_entry(archive, path, archive_name, expected_sha):
    path = regular(path)
    checksum = hashlib.sha256()
    before = path.stat()
    with path.open("rb") as source, archive.open(archive_name, "w", force_zip64=True) as destination:
        require(fingerprint(os.fstat(source.fileno())) == fingerprint(before), "File changed before packaging")
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(chunk)
            destination.write(chunk)
        require(fingerprint(os.fstat(source.fileno())) == fingerprint(before), "File changed while packaging")
    require(fingerprint(path.stat()) == fingerprint(before) and checksum.hexdigest() == expected_sha,
            "Packaged file differs from its recorded byte hash: " + str(path))


def main():
    require(ROOT.is_dir() and not ROOT.is_symlink() and ROOT.resolve() == ROOT,
            "The registered deployment3 root is unavailable or symlinked")
    require(ROOT.stat().st_uid == os.getuid(), "Deployment3 is not owned by this account")
    gate = terminal_accounting()
    require(not EXPORT.exists() and not EXPORT.is_symlink(), "export1 already exists; no overwrite or retry")
    EXPORT.mkdir(exist_ok=False)
    status = {"status": "stopped_export_error", "job_id": JOB_ID, "started_utc": utc_now(),
              "scientific_arrays_decoded": False, "model_imports": 0, "native_commands_executed": 0,
              "source_files_modified": False, "file_deletion": False, "settlement_original_status": gate["settlement_original_status"]}
    try:
        inventory = science_inventory()
        inventory.update(source_root=str(ROOT), scientific_root=root_relative(SCIENCE), job_id=JOB_ID,
                         created_utc=utc_now(), original_runner_inventory_verification=verify_original_runner_inventory())
        write_json(EXPORT / "scientific_file_inventory.json", inventory)
        selection = select_small_evidence(inventory)
        manifest = {"created_utc": utc_now(), "source_root": str(ROOT), "job_id": JOB_ID,
                    "terminal_accounting_gate": gate, **selection,
                    "small_file_ceiling_bytes": MAX_SMALL_FILE_BYTES,
                    "original_runner_inventory_verified": inventory["original_runner_inventory_verification"],
                    "generated_inventory_sha256": file_hash(EXPORT / "scientific_file_inventory.json"),
                    "scientific_arrays_decoded": False, "missing_files_synthesized": False,
                    "original_status_records_changed": False, "symlinks_followed_into_archive": False,
                    "omitted_categories": ["feature episode NPZs", "Hugging Face weights and caches",
                                           "all predictor checkpoints", "predictor basis.npz"]}
        write_json(EXPORT / "export_manifest.json", manifest)
        archive_path = EXPORT / ARCHIVE_NAME
        with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6,
                             allowZip64=True) as archive:
            for entry in selection["files"]:
                copy_zip_entry(archive, ROOT / entry["path"], entry["path"], entry["sha256"])
            for name in ("scientific_file_inventory.json", "export_manifest.json"):
                copy_zip_entry(archive, EXPORT / name, name, file_hash(EXPORT / name))
        status.update(status="passed", archive_path=str(archive_path), archive_bytes=archive_path.stat().st_size,
                      archive_sha256=file_hash(archive_path), copied_source_files=len(selection["files"]),
                      scientific_regular_files_hashed=len(inventory["regular_files"]),
                      scientific_symlinks_recorded=len(inventory["symlinks"]),
                      missing_expected_paths=len(selection["missing_expected_paths"]),
                      scientific_success_inferred=False)
    except Exception as error:
        status.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    status["finished_utc"] = utc_now()
    write_json(EXPORT / "status.json", status)
    with (EXPORT / "SHA256SUMS").open("x", encoding="utf-8") as destination:
        for path in sorted(EXPORT.iterdir()):
            if path.is_file() and path.name != "SHA256SUMS":
                destination.write(file_hash(path) + "  " + path.name + "\n")
    print(json.dumps(status, ensure_ascii=False, allow_nan=False))
    return 0 if status["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
