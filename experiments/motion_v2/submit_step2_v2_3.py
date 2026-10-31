"""Submit the registered Motion v2.3 A100 job once, or settle its native bill.

No model imports, array decoding, installation, cancellation or retry. A fresh
registered submission directory is the durable one-shot guard. Even an unknown
sbatch outcome consumes that guard and must be investigated without resubmitting.
Settlement reads the recorded job ID and never submits or cancels a job.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import shlex
import subprocess
import sys
import traceback


sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
ROOT = Path("/scratch/xy3257/motion-v2-20261031-deployment2")
MANIFEST_SHA = "9ac43b81eb75456e17f3542e64e8a0ab58282207752ae3712836f6d2ea20032a"
ARCHIVE_SHA = "58f14130ddda659f9c3a11bbbc50d2729cf61b0613da7bbf7522466afa28831d"
EXPECTED_SUBMISSION = {
    "wrapper_path": "experiments/motion_v2/run_step2_v2_3.sbatch",
    "output_dir": str(ROOT / "submission"),
    "preflight_dir": str(ROOT / "preflight"),
    "unpack_receipt": str(ROOT / "unpack_receipt.json"),
    "bound_dir": str(REPO / "results/motion_v2/v2_3/direction_bound"),
    "science_output": str(REPO / "results/motion_v2/v2_3/step2_gpu_attempt1"),
    "account": "csci_ga_2271-2026fa", "partition": "c12m85-a100-1",
    "gpu_request": "gpu:a100:1", "time_limit_seconds": 7200,
}
TERMINAL = {"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL",
            "PREEMPTED", "BOOT_FAIL", "DEADLINE", "OUT_OF_MEMORY", "REVOKED", "SPECIAL_EXIT"}
ACCOUNTING_FIELDS = ("JobIDRaw", "JobName", "User", "Account", "Partition", "State",
                     "ExitCode", "Start", "End", "ElapsedRaw", "AllocTRES", "ReqTRES")


def require(value, message):
    if not value:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as source:
        return json.load(source)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as target:
        json.dump(value, target, indent=2, ensure_ascii=False, allow_nan=False)
        target.write("\n")


def safe_path(value, parent=ROOT, exists=True):
    path = Path(value).absolute()
    require(path.is_relative_to(parent), "Path is outside the registered directory: " + str(path))
    for component in (path, *path.parents):
        require(not component.is_symlink(), "Symlink path is not allowed: " + str(component))
    if exists:
        require(path.exists(), "Required file or directory is missing: " + str(path))
    return path


def repo_file(value):
    path = Path(value)
    return safe_path(path if path.is_absolute() else REPO / path, REPO)


def command(argv, timeout=30, env=None):
    try:
        result = subprocess.run(argv, cwd=REPO, capture_output=True, text=True,
                                timeout=timeout, env=env)
        return {"argv": list(argv), "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr, "captured_utc": now()}
    except subprocess.TimeoutExpired as error:
        def as_text(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else (value or "")
        return {"argv": list(argv), "returncode": None, "timed_out": True,
                "stdout": as_text(error.stdout), "stderr": as_text(error.stderr), "captured_utc": now()}


def checked_command(argv):
    result = command(argv)
    require(result["returncode"] == 0, "Read-only command failed: " + repr(argv) + " " + result["stderr"])
    return result["stdout"].strip()


def committed(path):
    path = repo_file(path)
    relative = path.relative_to(REPO).as_posix()
    original = subprocess.run(["git", "show", "HEAD:" + relative], cwd=REPO,
                              capture_output=True, check=True, timeout=30).stdout
    require(original == path.read_bytes(), "File differs from committed HEAD: " + relative)
    return path


def check_hash_inventory(mapping):
    require(isinstance(mapping, dict) and mapping, "Missing registered source hashes")
    for relative, expected in mapping.items():
        require(re.fullmatch(r"[0-9a-f]{64}", expected) is not None, "Invalid source SHA256")
        require(sha256(committed(relative)) == expected, "Registered source SHA256 mismatch: " + relative)


def verify_sha256sums(directory):
    directory = safe_path(directory)
    inventory = directory / "SHA256SUMS"
    seen = {}
    for line in inventory.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        require(name not in seen and Path(name).name == name, "Unsafe or duplicate receipt hash entry")
        require(sha256(safe_path(directory / name)) == expected, "Receipt SHA256 mismatch: " + name)
        seen[name] = expected
    require(set(seen) == {"receipt.json", "status.json"}, "Incomplete preflight hash inventory")
    return {**seen, "SHA256SUMS": sha256(inventory)}


def seconds(value):
    days, clock = value.split("-", 1) if "-" in value else ("0", value)
    items = [int(item) for item in clock.split(":")]
    require(len(items) == 3 and min(items) >= 0, "Unexpected native time syntax")
    return int(days) * 86400 + items[0] * 3600 + items[1] * 60 + items[2]


def one_gpu(tres):
    """Generic and typed GPU TRES describe the same allocation, never add them."""
    entries = {}
    for item in tres.split(","):
        if "=" in item:
            key, value = item.split("=", 1)
            if key == "gres/gpu" or key.startswith("gres/gpu:"):
                require(key not in entries and value.isdigit(), "Invalid native GPU TRES")
                entries[key] = int(value)
    require(entries and all(value == 1 for value in entries.values()), "Native TRES does not certify exactly one GPU")
    typed = [key for key in entries if key.startswith("gres/gpu:")]
    require(len(typed) <= 1 and all(key == "gres/gpu:a100" for key in typed), "Unexpected native GPU type")
    return 1


def control_fields(result, job_id):
    require(result["returncode"] == 0, "Cannot read immediate native job state")
    fields = dict(item.split("=", 1) for item in result["stdout"].split() if "=" in item)
    require(fields.get("JobId") == job_id, "Native job identity differs from submitted job")
    require(fields.get("Restarts") == "0", "Native Restarts is missing or nonzero")
    require(fields.get("Account") == EXPECTED_SUBMISSION["account"] and
            fields.get("Partition") == EXPECTED_SUBMISSION["partition"], "Native account/partition mismatch")
    require(fields.get("NumNodes") == "1", "Native job must request one node")
    require(seconds(fields["TimeLimit"]) == 7200, "Native time limit differs from the frozen 7200 seconds")
    require(fields.get("UserId", "").split("(", 1)[0] == "xy3257", "Unexpected native job owner")
    require(fields.get("Command") == str(REPO / EXPECTED_SUBMISSION["wrapper_path"])
            and fields.get("WorkDir") == str(REPO), "Native wrapper command or working directory mismatch")
    one_gpu(fields.get("ReqTRES", ""))
    # Requeue is preserved as evidence. Restarts, not Requeue, is the native gate.
    return fields


def validate_wrapper(path):
    directives = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("#SBATCH "):
            continue
        items = shlex.split(line[len("#SBATCH "):])
        require(items, "Empty SBATCH directive")
        if "=" in items[0]:
            require(len(items) == 1, "Unexpected extra SBATCH values")
            key, value = items[0].split("=", 1)
        else:
            require(len(items) <= 2, "Unexpected SBATCH directive")
            key, value = items[0], items[1] if len(items) == 2 else True
        require(key not in directives, "Duplicate SBATCH directive: " + key)
        directives[key] = value
    expected = {"--account": EXPECTED_SUBMISSION["account"],
                "--partition": EXPECTED_SUBMISSION["partition"], "--gres": "gpu:a100:1",
                "--nodes": "1", "--ntasks": "1", "--cpus-per-task": "4", "--mem": "16G",
                "--time": "02:00:00", "--no-requeue": True, "--open-mode": "append",
                "--output": str(ROOT / "scientific-job-%j.stdout.log"),
                "--error": str(ROOT / "scientific-job-%j.stderr.log")}
    require(all(directives.get(key) == value for key, value in expected.items()),
            "Wrapper resource/output directives differ from the frozen deployment")
    require(not any(key in directives for key in ("--gpus", "--gpus-per-node", "--gpus-per-task", "--requeue")),
            "Wrapper contains a competing GPU or requeue request")
    return directives


def registration(args):
    require(REPO == ROOT / "repo", "Run this helper only from the new deployment checkout")
    require(sys.flags.isolated == 1 and sys.flags.dont_write_bytecode == 1, "Use Python -I -B")
    require(pwd.getpwuid(os.getuid()).pw_name == "xy3257", "Unexpected authenticated account")
    require(re.fullmatch(r"[0-9a-f]{40}", args.expected_head) is not None, "Require exact 40-character pushed HEAD")
    head = checked_command(["git", "rev-parse", "HEAD"])
    require(head == args.expected_head, "Checkout differs from the registered pushed HEAD")
    protocol_path = committed(args.protocol)
    protocol = read_json(protocol_path)
    specification = protocol["submission"]
    for key, value in EXPECTED_SUBMISSION.items():
        require(specification.get(key) == value, "Unregistered submission parameter: " + key)
    check_hash_inventory(protocol["source_sha256"])
    own = Path(__file__).relative_to(REPO).as_posix()
    require(own in protocol["source_sha256"], "Deployment protocol does not lock this submit helper")
    wrapper = committed(specification["wrapper_path"])
    require(sha256(wrapper) == specification["wrapper_sha256"] ==
            protocol["source_sha256"][specification["wrapper_path"]], "Wrapper SHA256 mismatch")
    directives = validate_wrapper(wrapper)
    science_ref = protocol["frozen_scientific_protocol"]
    science_path = committed(science_ref["path"])
    require(sha256(science_path) == science_ref["sha256"], "Scientific protocol SHA256 mismatch")
    science = read_json(science_path)
    require(science.get("status") == "FROZEN_STEP2_PROTOCOL" and science.get("registered_before_run") is True,
            "Scientific protocol is not frozen before execution")
    check_hash_inventory(science["source_sha256"])
    require(science["dataset_manifest_sha256"] == MANIFEST_SHA, "Scientific manifest registration changed")
    manifest = committed(science["dataset_manifest_path"])
    require(sha256(manifest) == MANIFEST_SHA, "Actual manifest SHA256 mismatch")
    return protocol, science, wrapper, {
        "head_commit": head, "deployment_protocol_sha256": sha256(protocol_path),
        "scientific_protocol_sha256": science_ref["sha256"], "wrapper_sha256": sha256(wrapper),
        "wrapper_directives": directives, "manifest_sha256": MANIFEST_SHA,
    }


def preparation_receipts(protocol, science, registered):
    specification = protocol["submission"]
    preflight = safe_path(specification["preflight_dir"])
    preflight_hashes = verify_sha256sums(preflight)
    preflight_status = read_json(preflight / "status.json")
    receipt = read_json(preflight / "receipt.json")
    require(preflight_status.get("status") == "passed" and
            preflight_status.get("receipt_sha256") == preflight_hashes["receipt.json"], "CPU preflight did not pass")
    for key in ("head_commit", "deployment_protocol_sha256", "scientific_protocol_sha256", "manifest_sha256"):
        require(receipt["registration"].get(key) == registered[key], "Preflight registration mismatch: " + key)
    unpack_path = safe_path(specification["unpack_receipt"])
    unpack = read_json(unpack_path)
    expected_unpack = {"status": "passed", "operation": "unpack", "manifest_sha256": MANIFEST_SHA,
                       "artifact_files": 2432, "archive_sha256": ARCHIVE_SHA, "archive_bytes": 12609763,
                       "splits": ["train", "validation"], "generation_or_decoding": False,
                       "test_access": False, "file_deletion": False}
    require(all(unpack.get(key) == value for key, value in expected_unpack.items()), "Unpack receipt is not the frozen passed transfer")
    bound = safe_path(specification["bound_dir"])
    hashes = read_json(bound / "artifact_sha256.json")
    require(set(hashes) == {"direction_bound.npz", "status.json"}, "Unexpected CPU-bound hash inventory")
    require({path.name for path in bound.iterdir()} == {"direction_bound.npz", "status.json", "artifact_sha256.json"},
            "Unexpected CPU-bound output inventory")
    for name, checksum in hashes.items():
        require(sha256(safe_path(bound / name)) == checksum, "CPU-bound artifact byte hash mismatch: " + name)
    bound_status = read_json(bound / "status.json")
    expected_bound = {"status": "passed", "actual_a100_gpu_seconds": 0, "evaluation_split": "validation",
                      "pairs": 96, "episodes": 192, "horizons": [3, 6], "test_opened_or_generated": False,
                      "training_or_model_imported": False, "protocol_sha256": registered["scientific_protocol_sha256"],
                      "manifest_sha256": MANIFEST_SHA}
    require(all(bound_status.get(key) == value for key, value in expected_bound.items()), "CPU direction bound did not pass the registered preparation")
    bound_helper = "experiments/motion_v2/prepare_direction_bound_v2_3.py"
    require(bound_status["source_sha256"] == {bound_helper: science["source_sha256"][bound_helper]}, "CPU-bound source hash mismatch")
    science_output = safe_path(specification["science_output"], exists=False)
    require(not science_output.exists(), "Scientific output already exists; no repeat is permitted")
    invocation_marker = safe_path(ROOT / "scientific_job_started", exists=False)
    require(not invocation_marker.exists(), "Scientific invocation marker already exists; do not resubmit")
    return {"preflight_hashes": preflight_hashes, "unpack_receipt_sha256": sha256(unpack_path),
            "direction_bound_hashes": {**hashes, "artifact_sha256.json": sha256(bound / "artifact_sha256.json")},
            "scientific_output_absent_before_submission": str(science_output), "dataset_arrays_decoded": False}


def save_native(output, name, result):
    for stream in ("stdout", "stderr"):
        with (output / (name + "." + stream + ".txt")).open("x", encoding="utf-8") as target:
            target.write(result[stream])


def finish(output, receipt_name, receipt, status):
    write_json(output / receipt_name, receipt)
    status["receipt_sha256"] = sha256(output / receipt_name)
    write_json(output / "status.json", status)
    names = sorted(path.name for path in output.iterdir() if path.is_file())
    with (output / "SHA256SUMS").open("x", encoding="utf-8") as target:
        for name in names:
            target.write(sha256(output / name) + "  " + name + "\n")
    print(json.dumps(status, ensure_ascii=False, allow_nan=False))


def submit(args):
    output = safe_path(args.output, exists=False)
    require(output == ROOT / "submission", "Use only the registered one-shot submission directory")
    require(not output.exists(), "Submission directory already exists; do not retry even after an unknown outcome")
    output.mkdir(exist_ok=False)
    receipt = {"started_utc": now(), "sbatch_attempted": False, "job_id": None,
               "retry_permitted": False, "control_session_job_id": os.environ.get("SLURM_JOB_ID")}
    status = {"status": "stopped_before_submission", "sbatch_attempts": 0,
              "scientific_success_inferred": False, "retry_permitted": False}
    exit_code = 1
    try:
        protocol, science, wrapper, registered = registration(args)
        receipt["registration"] = registered
        receipt["preparation"] = preparation_receipts(protocol, science, registered)
        argv = ["sbatch", "--parsable", "--no-requeue", str(wrapper)]
        # Inherited SBATCH_* options can override directives. Remove them from
        # this subprocess only, preserving the original control environment.
        removed = sorted(key for key in os.environ if key.startswith("SBATCH_"))
        environment = {key: value for key, value in os.environ.items() if not key.startswith("SBATCH_")}
        marker = {"created_utc": now(), "command": argv, "registration": registered,
                  "removed_sbatch_environment_keys": removed, "sbatch_attempt_limit": 1,
                  "unknown_outcome_requires_manual_read_only_investigation": True}
        write_json(output / "submission_started.json", marker)
        receipt["sbatch_attempted"] = True
        status.update(status="submission_outcome_unknown", sbatch_attempts=1)
        result = command(argv, timeout=60, env=environment)  # The only sbatch call.
        receipt["sbatch"] = result
        save_native(output, "sbatch", result)
        require(result["returncode"] == 0, "sbatch failed or timed out; outcome is not retried")
        match = re.fullmatch(r"([0-9]+)(?:;([A-Za-z0-9_.-]+))?", result["stdout"].strip())
        require(match is not None, "Unrecognized sbatch response; do not submit again")
        job_id = match.group(1)
        receipt["job_id"] = job_id
        receipt["cluster"] = match.group(2)
        status.update(status="stopped_after_submission", job_id=job_id)
        native = command(["scontrol", "show", "job", "--oneliner", job_id])
        receipt["immediate_scontrol"] = native
        save_native(output, "immediate_scontrol", native)
        fields = control_fields(native, job_id)
        receipt["immediate_fields"] = fields
        receipt["requeue_observed_not_gate"] = fields.get("Requeue")
        receipt["requires_final_native_accounting"] = True
        status.update(status="submitted_verified", native_state=fields.get("JobState"),
                      restarts=0, final_accounting_pending=True)
        exit_code = 0
    except Exception as error:
        status.update(error_type=type(error).__name__, error=str(error))
        receipt["traceback"] = traceback.format_exc()
    receipt["finished_utc"] = now()
    finish(output, "submission.json", receipt, status)
    return exit_code


def accounting_row(result, job_id):
    require(result["returncode"] == 0, "Native sacct query failed")
    rows = [line.rstrip("|").split("|") for line in result["stdout"].splitlines()
            if line.split("|", 1)[0] == job_id]
    require(len(rows) == 1 and len(rows[0]) == len(ACCOUNTING_FIELDS), "Native top-level accounting row is unavailable or malformed")
    return dict(zip(ACCOUNTING_FIELDS, rows[0]))


def settle(args):
    require(args.submission is not None, "--settle requires the original --submission receipt")
    output = safe_path(args.output, exists=False)
    require(output == ROOT / "settlement" and not output.exists(), "Settlement output must be the new registered settlement directory")
    protocol, science, wrapper, registered = registration(args)
    submission_path = safe_path(args.submission)
    require(submission_path == ROOT / "submission/submission.json", "Use the original registered submission receipt")
    submitted = read_json(submission_path)
    status_path = submission_path.parent / "status.json"
    require(read_json(status_path).get("receipt_sha256") == sha256(submission_path), "Submission receipt hash mismatch")
    require(submitted.get("sbatch_attempted") is True and submitted.get("registration") == registered,
            "Submission does not match the frozen deployment")
    job_id = submitted.get("job_id", "")
    require(isinstance(job_id, str) and re.fullmatch(r"[0-9]+", job_id) is not None, "No confirmed job ID in original submission receipt")
    formatting = ",".join(name + "%40" if name in {"State", "JobName"} else name for name in ACCOUNTING_FIELDS)
    sacct = command(["sacct", "-j", job_id, "--noheader", "--parsable2", "--format=" + formatting])
    row = accounting_row(sacct, job_id)
    state = row["State"].split()[0].rstrip("+")
    if state not in TERMINAL:
        print(json.dumps({"status": "not_terminal", "job_id": job_id, "native_state": row["State"],
                          "persistent_results_written": False, "scientific_success_inferred": False}))
        return 75
    native = command(["scontrol", "show", "job", "--oneliner", job_id])
    output.mkdir(exist_ok=False)
    save_native(output, "final_sacct", sacct)
    save_native(output, "final_scontrol", native)
    receipt = {"created_utc": now(), "job_id": job_id, "registration": registered,
               "submission_receipt_sha256": sha256(submission_path), "sacct": sacct,
               "scontrol": native, "allocation_row": row, "scientific_success_inferred": False}
    status = {"status": "stopped_accounting_error", "job_id": job_id, "native_state": state,
              "scientific_success_inferred": False, "sbatch_attempts_in_settlement": 0}
    exit_code = 1
    try:
        require(row["User"] == "xy3257" and row["Account"] == EXPECTED_SUBMISSION["account"]
                and row["Partition"] == EXPECTED_SUBMISSION["partition"], "Native final accounting identity mismatch")
        gpu_count = one_gpu(row["AllocTRES"])
        one_gpu(row["ReqTRES"])
        require(row["ElapsedRaw"].isdigit() and row["End"] not in {"", "Unknown", "None"}, "Final elapsed time/end is missing")
        gpu_seconds = int(row["ElapsedRaw"]) * gpu_count
        receipt["allocated_gpu_count"] = gpu_count
        receipt["actual_a100_gpu_seconds"] = gpu_seconds
        receipt["actual_a100_gpu_hours"] = gpu_seconds / 3600
        status["actual_a100_gpu_seconds"] = gpu_seconds
        if native["returncode"] == 0:
            receipt["final_fields"] = control_fields(native, job_id)
        else:
            receipt["final_scontrol_unavailable"] = "Raw failure retained; terminal sacct allocation row is the accounting authority"
        require(gpu_seconds <= 7200, "Actual allocation exceeded the registered 7200 A100-second budget")
        status.update(status="passed", accounting_complete=True, registered_budget_seconds=7200)
        exit_code = 0
    except Exception as error:
        status.update(error_type=type(error).__name__, error=str(error))
        receipt["traceback"] = traceback.format_exc()
    finish(output, "accounting.json", receipt, status)
    return exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--settle", action="store_true")
    parser.add_argument("--submission", type=Path)
    args = parser.parse_args()
    require(args.settle or args.submission is None, "--submission is only accepted in read-only settlement mode")
    return settle(args) if args.settle else submit(args)


if __name__ == "__main__":
    raise SystemExit(main())
