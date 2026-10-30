"""Read-only Slurm/accounting export for the completed scaled CPU recovery.

Does not submit jobs, run inference, edit prior evidence, or change the existing
result packer. Creates one NEW evidence directory, ZIP, SHA256 file and receipt.
Run only after the independent worker from scaled_recovery_submission.json has
completed; a live control session is recorded as a snapshot, not as completed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from datetime import datetime, timezone
import zipfile


SACCT = "/opt/slurm/bin/sacct"
INTERRUPTED_JOB = "11023"
CONTROL_JOB = "11056"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def new_text(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(value)


def json_text(value):
    return json.dumps(value, indent=2, allow_nan=False) + "\n"


def query_accounting(job_id):
    if not re.fullmatch(r"[1-9][0-9]*", job_id):
        raise ValueError("Expected a plain numeric Slurm job ID")
    command = [SACCT, "-j", job_id, "--parsable2",
               "--format=JobID,State,ExitCode,ElapsedRaw,Timelimit,Reason,AllocTRES,NodeList,Start,End"]
    result = subprocess.run(command, text=True, capture_output=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError("sacct failed for job " + job_id + ": " + result.stderr.strip())
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Empty accounting output for job " + job_id)
    header = [name.strip().lower() for name in lines[0].split("|")]
    required = {"jobid", "state", "exitcode", "elapsedraw", "reason", "alloctres"}
    if not required.issubset(header):
        raise ValueError("Unexpected sacct header for job " + job_id)
    rows = []
    for line in lines[1:]:
        fields = line.split("|")
        if len(fields) != len(header):
            raise ValueError("Malformed sacct row for job " + job_id)
        rows.append(dict(zip(header, fields)))
    exact = [row for row in rows if row["jobid"] == job_id]
    if len(exact) != 1:
        raise ValueError("Need exactly one parent-job accounting row for " + job_id)
    return {"command": command, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr,
            "parent_row": exact[0], "rows": rows}


def export(work):
    work = Path(work).resolve()
    destination = work / "scaled-execution-evidence"
    archive_path = work / "scaled-execution-evidence.zip"
    checksum_path = work / "scaled-execution-evidence.zip.sha256"
    receipt_path = work / "scaled-execution-evidence.receipt.json"
    if any(path.exists() for path in (destination, archive_path, checksum_path, receipt_path)):
        raise ValueError("Existing execution export found; refuse any overwrite")

    sources = {}

    def include(relative, archive_name, expected_sha=None):
        path = (work / relative).resolve()
        if not path.is_relative_to(work) or not path.is_file():
            raise ValueError("Missing/unsafe work artifact: " + str(relative))
        actual = sha(path)
        if expected_sha is not None and actual != expected_sha:
            raise ValueError("Frozen artifact hash mismatch: " + str(relative))
        if archive_name in sources:
            raise ValueError("Duplicate output member: " + archive_name)
        sources[archive_name] = (path, actual)
        return path

    submission_path = include("scaled_recovery_submission.json", "execution/scaled_recovery_submission.json")
    submission = read(submission_path)
    if submission.get("status") != "submitted" or submission.get("returncode") != 0:
        raise ValueError("Recovery submission is not confirmed successful")
    match = re.fullmatch(r"([1-9][0-9]*)(?:;([^\s;]+))?\s*", submission.get("stdout", ""))
    if not match:
        raise ValueError("Recovery stdout does not contain exactly one parsable job number")
    worker_job = match.group(1)
    if worker_job in (INTERRUPTED_JOB, CONTROL_JOB):
        raise ValueError("Recovery worker must be independent of old/control jobs")
    if str(submission.get("parent_control_job")) != CONTROL_JOB:
        raise ValueError("Recovery submission control-job identity differs")
    if submission.get("new_gpu_count") != 0:
        raise ValueError("Recovery submission is not the declared CPU-only attempt")

    sbatch_path = include("scaled-recovery.sbatch", "execution/scaled-recovery.sbatch", submission["script_sha256"])
    request = submission.get("command", [])
    if (not isinstance(request, list) or not request or request[0] != "/opt/slurm/bin/sbatch"
            or Path(request[-1]).resolve() != sbatch_path
            or "--output=" + str(work / "scaled-probe-actual.log") not in request):
        raise ValueError("Saved submission command does not bind the expected sbatch/log")
    include("submit_scaled_probe_recovery.py", "source/submit_scaled_probe_recovery.py", submission["submitter_sha256"])
    reviewed = submission.get("reviewed_source_sha256", {})
    required_sources = {"bundle/experiments/phase2/run_scaled_use_time_probe.py": "source/run_scaled_use_time_probe.py",
                        "pack_scaled_probe.py": "source/pack_scaled_probe.py"}
    if set(reviewed) != {str((work / name).resolve()) for name in required_sources}:
        raise ValueError("Unexpected reviewed recovery-source set")
    for relative, name in required_sources.items():
        include(relative, name, reviewed[str((work / relative).resolve())])

    # Query the exact worker, not the interactive control job that submitted it.
    accounting = {job: query_accounting(job) for job in (worker_job, INTERRUPTED_JOB, CONTROL_JOB)}
    current = accounting[worker_job]["parent_row"]
    if current["state"] != "COMPLETED" or current["exitcode"] != "0:0":
        raise ValueError("Independent CPU worker has not completed successfully: " + json_text(current).strip())
    if not current["elapsedraw"].isdigit() or int(current["elapsedraw"]) <= 0:
        raise ValueError("Invalid worker elapsed accounting")
    for item in current["alloctres"].split(","):
        key, separator, value = item.partition("=")
        if separator and key.startswith("gres/gpu") and value != "0":
            raise ValueError("Worker accounting unexpectedly includes a GPU")
    interrupted = accounting[INTERRUPTED_JOB]["parent_row"]
    if not interrupted["state"].startswith("CANCELLED"):
        raise ValueError("Interrupted job no longer matches recorded cancelled attempt")

    include("scaled-probe-actual.log", "execution/scaled-probe-actual.log")
    include("scaled-probe-session11023.log", "execution/scaled-probe-session11023.log")
    include("sacct-interrupted-11023.txt", "execution/sacct-interrupted-11023-at-recovery.txt")
    summary_path = include("scaled-use-time-probe/summary.json", "results/summary.json")
    if read(summary_path).get("status") != "completed_scaled_use_time_pilot":
        raise ValueError("The independent worker has no completed scaled-pilot summary")
    include("scaled-use-time-probe/protocol.json", "results/protocol.json")
    package_receipt_path = include("scaled-use-time-probe-full.receipt.json", "results/scaled-use-time-probe-full.receipt.json")
    package_receipt = read(package_receipt_path)
    package_path = work / "scaled-use-time-probe-full.zip"
    if (package_receipt.get("status") != "verified_package_created"
            or Path(package_receipt.get("path", "")).resolve() != package_path
            or not package_path.is_file()
            or package_path.stat().st_size != package_receipt.get("bytes")
            or sha(package_path) != package_receipt.get("sha256")
            or package_receipt.get("zip_crc_passed") is not True):
        raise ValueError("Completed result ZIP is not bound to its packer receipt")

    # No output is created until completion and all existing inputs are verified.
    destination.mkdir()
    for name, (path, expected) in sources.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(path.read_bytes())
        if sha(target) != expected or sha(path) != expected:
            raise ValueError("Evidence changed while snapshotting: " + str(path))
    self_target = destination / "source/export_scaled_execution.py"
    with self_target.open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    for job, report in accounting.items():
        new_text(destination / f"execution/sacct-{job}.txt", report["stdout"])
        new_text(destination / f"execution/sacct-{job}.stderr.txt", report["stderr"])

    worker_receipt = {"status": "verified_completed_cpu_recovery", "created_utc": datetime.now(timezone.utc).isoformat(),
                      "worker_job_id": worker_job, "submission_stdout": submission["stdout"],
                      "submission_cluster_suffix": match.group(2), "worker_accounting": current,
                      "interrupted_job_id": INTERRUPTED_JOB, "interrupted_accounting": interrupted,
                      "control_job_id": CONTROL_JOB, "control_accounting_snapshot": accounting[CONTROL_JOB]["parent_row"],
                      "result_zip_sha256": package_receipt["sha256"], "result_zip_bytes": package_receipt["bytes"],
                      "exporter_sha256": sha(__file__), "new_model_calls": 0, "new_job_submissions": 0,
                      "gpu_used_by_export": False,
                      "accounting_commands": {job: report["command"] for job, report in accounting.items()},
                      "limits": ["Control-job state is a point-in-time snapshot, not a completion claim",
                                 "CANCELLED and Reason are retained verbatim; no cause of cancellation is inferred",
                                 "This is execution provenance; scientific results are in the separately verified result ZIP"]}
    new_text(destination / "worker_receipt.json", json_text(worker_receipt))
    files = {str(path.relative_to(destination)): {"sha256": sha(path), "bytes": path.stat().st_size}
             for path in sorted(destination.rglob("*")) if path.is_file()}
    manifest = {"status": "verified_execution_snapshot", "worker_job_id": worker_job,
                "created_utc": datetime.now(timezone.utc).isoformat(), "files": files,
                "existing_artifacts_unchanged": True, "new_model_calls": 0, "new_job_submissions": 0}
    new_text(destination / "evidence_manifest.json", json_text(manifest))
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                archive.write(path, str(path.relative_to(destination)))
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Execution evidence ZIP failed CRC check")
        for name, expected in files.items():
            payload = archive.read(name)
            if len(payload) != expected["bytes"] or hashlib.sha256(payload).hexdigest() != expected["sha256"]:
                raise ValueError("Packaged execution evidence mismatch: " + name)
    result = {"status": "created_verified_scaled_execution_evidence", "worker_job_id": worker_job,
              "path": str(archive_path), "sha256": sha(archive_path), "bytes": archive_path.stat().st_size,
              "zip_members": len(files) + 1, "zip_crc_passed": True,
              "manifest_sha256": sha(destination / "evidence_manifest.json"),
              "new_model_calls": 0, "new_job_submissions": 0, "original_files_overwritten": 0}
    new_text(checksum_path, result["sha256"] + "  " + archive_path.name + "\n")
    new_text(receipt_path, json_text(result))
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    export(args.work)


if __name__ == "__main__":
    main()
