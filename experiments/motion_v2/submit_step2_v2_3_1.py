"""Submit the registered Motion v2.3.1 A100 job once, or settle its native bill.

No model imports, array decoding, installation, cancellation or retry. A fresh
registered submission directory is the durable one-shot guard. Even an unknown
sbatch outcome consumes that guard and must be investigated without resubmitting.
Settlement reads the recorded job ID and never submits or cancels a job.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
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
ROOT = Path("/scratch/xy3257/motion-v2-20261031-deployment3")
MANIFEST_SHA = "9ac43b81eb75456e17f3542e64e8a0ab58282207752ae3712836f6d2ea20032a"
ARCHIVE_SHA = "58f14130ddda659f9c3a11bbbc50d2729cf61b0613da7bbf7522466afa28831d"
EXPECTED_SUBMISSION = {
    "wrapper_path": "experiments/motion_v2/run_step2_v2_3_1.sbatch",
    "output_dir": str(ROOT / "submission"),
    "preflight_dir": str(ROOT / "preflight"),
    "unpack_receipt": str(ROOT / "unpack_receipt.json"),
    "bound_dir": str(REPO / "results/motion_v2/v2_3/direction_bound"),
    "science_output": str(REPO / "results/motion_v2/v2_3_1/step2_gpu_attempt1"),
    "account": "csci_ga_2271-2026fa", "partition": "c12m85-a100-1",
    "gpu_request": "gpu:a100:1", "time_limit_seconds": 10800,
}
TERMINAL = {"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL",
            "PREEMPTED", "BOOT_FAIL", "DEADLINE", "OUT_OF_MEMORY", "REVOKED", "SPECIAL_EXIT"}
ACCOUNTING_FIELDS = ("JobIDRaw", "JobName", "User", "Account", "Partition", "State",
                     "ExitCode", "Start", "End", "ElapsedRaw", "AllocTRES", "ReqTRES")
_PARSERS = None


def parsers():
    """Load only the shared stdlib parser, lazily; importing this module does no input IO."""
    global _PARSERS
    if _PARSERS is None:
        path = Path(__file__).with_name("slurm_records.py")
        spec = importlib.util.spec_from_file_location("motion_v231_submission_slurm_records", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _PARSERS = module
    return _PARSERS


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
    return parsers().parse_duration(value)


def one_gpu(tres):
    return parsers().one_gpu(tres)


def deployment_context():
    return {"account": EXPECTED_SUBMISSION["account"], "partition": EXPECTED_SUBMISSION["partition"],
            "owner": "xy3257", "time_limit_seconds": 10800, "hard_cap_seconds": 14400,
            "wrapper_path": str(REPO / EXPECTED_SUBMISSION["wrapper_path"]), "workdir": str(REPO)}


def validate_submission_control(text, job_id, expected):
    """Pure native-field requirements, with explicit historical or current context."""
    fields = parsers().parse_control(text)
    required = {"JobId", "JobState", "Restarts", "Account", "Partition", "NumNodes",
                "TimeLimit", "UserId", "Command", "WorkDir", "ReqTRES"}
    require(required <= fields.keys(), "Missing native submission fields: " + repr(sorted(required - fields.keys())))
    require(fields.get("JobId") == job_id, "Native job identity differs from submitted job")
    require(bool(fields.get("JobState")), "Native job state is missing")
    require(fields.get("Restarts") == "0", "Native Restarts is missing or nonzero")
    require(fields.get("Account") == expected["account"] and
            fields.get("Partition") == expected["partition"], "Native account/partition mismatch")
    parsers().require_one_node(fields.get("NumNodes", ""))
    require(seconds(fields["TimeLimit"]) == expected["time_limit_seconds"], "Native time limit differs from registration")
    require(fields.get("UserId", "").split("(", 1)[0] == expected["owner"], "Unexpected native job owner")
    require(fields.get("Command") == expected["wrapper_path"]
            and fields.get("WorkDir") == expected["workdir"], "Native wrapper command or working directory mismatch")
    one_gpu(fields.get("ReqTRES", ""))
    # Requeue is preserved as evidence. Restarts, not Requeue, is the native gate.
    return fields


def control_fields(result, job_id):
    require(result["returncode"] == 0, "Cannot read immediate native job state")
    return validate_submission_control(result["stdout"], job_id, deployment_context())


def validate_settlement_record(row, control_text, expected):
    """Pure final-accounting requirements, including a never-allocated cancellation."""
    required = {"JobIDRaw", "User", "Account", "Partition", "State", "Start", "End", "ElapsedRaw", "AllocTRES", "ReqTRES"}
    require(required <= row.keys(), "Missing native accounting fields: " + repr(sorted(required - row.keys())))
    state = parsers().normalize_state(row["State"])
    require(state in TERMINAL, "Final native accounting must be terminal")
    require(row["User"] == expected["owner"] and row["Account"] == expected["account"]
            and row["Partition"] == expected["partition"], "Native final accounting identity mismatch")
    require(row["ElapsedRaw"].isdigit() and row["End"] not in {"", "Unknown", "None"}, "Final elapsed time/end is missing")
    one_gpu(row["ReqTRES"])
    gpu_count = parsers().gpu_count(row["AllocTRES"])
    elapsed = int(row["ElapsedRaw"])
    fields = None
    if control_text is not None:
        fields = validate_submission_control(control_text, row["JobIDRaw"], expected)
    if gpu_count == 0:
        require(state == "CANCELLED" and elapsed == 0 and row["Start"] in {"", "None", "Unknown"}
                and row["AllocTRES"].strip() in {"", "(null)", "None"},
                "Zero allocation is only certified for a cancelled, never-started, zero-elapsed job")
        if fields is not None:
            require({"RunTime", "NodeList", "AllocTRES"} <= fields.keys(),
                    "Final scontrol omits never-allocated cancellation evidence")
            require(parsers().normalize_state(fields["JobState"]) == "CANCELLED"
                    and seconds(fields["RunTime"]) == 0
                    and fields.get("NodeList") in {"", "(null)", "None"}
                    and parsers().gpu_count(fields.get("AllocTRES", "")) == 0,
                    "Final scontrol contradicts never-allocated cancellation")
        allocation_case = "cancelled_before_allocation"
    else:
        one_gpu(row["AllocTRES"])
        allocation_case = "allocated_single_a100"
    gpu_seconds = elapsed * gpu_count
    require(gpu_seconds <= min(expected["time_limit_seconds"], expected["hard_cap_seconds"]),
            "Actual allocation exceeded the registered A100-second budget")
    return {"native_state": state, "allocated_gpu_count": gpu_count,
            "actual_a100_gpu_seconds": gpu_seconds, "actual_a100_gpu_hours": gpu_seconds / 3600,
            "allocation_case": allocation_case, "accounting_complete": True,
            "final_control_available": fields is not None, "final_fields": fields,
            "scientific_success_inferred": False}


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
                "--time": "03:00:00", "--no-requeue": True, "--open-mode": "append",
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
    require("experiments/motion_v2/slurm_records.py" in protocol["source_sha256"],
            "Deployment protocol does not lock the shared native parser")
    require(protocol["budget"]["registered_a100_seconds"] == 10800
            and protocol["budget"]["user_total_cap_seconds"] == 14400,
            "Deployment budget differs from the authorized 10800/14400-second limits")
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
    reuse = science["reused_direction_bound"]
    require(bound == repo_file(reuse["source_path"]), "Bound path differs from the explicit reused source")
    require(reuse["source_deployment"] == "/scratch/xy3257/motion-v2-20261031-deployment2",
            "Unexpected original direction-bound deployment")
    original_protocol = committed(reuse["original_protocol_path"])
    require(sha256(original_protocol) == reuse["original_protocol_sha256"],
            "Original direction-bound scientific protocol SHA256 mismatch")
    original_registration = read_json(original_protocol)
    bound_helper = reuse["original_helper_path"]
    require(bound_helper == "experiments/motion_v2/prepare_direction_bound_v2_3.py"
            and original_registration["source_sha256"][bound_helper] == reuse["original_helper_sha256"],
            "Original direction-bound preparation source registration mismatch")
    require(set(reuse["artifact_sha256"]) == {"direction_bound.npz", "status.json", "artifact_sha256.json"},
            "Explicit reused-bound registration must lock all three original artifacts")
    for name, checksum in reuse["artifact_sha256"].items():
        require(sha256(committed(bound / name)) == checksum, "Original bound byte hash mismatch: " + name)
    hashes = read_json(bound / "artifact_sha256.json")
    require(set(hashes) == {"direction_bound.npz", "status.json"}, "Unexpected CPU-bound hash inventory")
    require({path.name for path in bound.iterdir()} == {"direction_bound.npz", "status.json", "artifact_sha256.json"},
            "Unexpected CPU-bound output inventory")
    for name, checksum in hashes.items():
        require(sha256(safe_path(bound / name)) == checksum, "CPU-bound artifact byte hash mismatch: " + name)
    bound_status = read_json(bound / "status.json")
    expected_bound = {"status": "passed", "actual_a100_gpu_seconds": 0, "evaluation_split": "validation",
                      "pairs": 96, "episodes": 192, "horizons": [3, 6], "test_opened_or_generated": False,
                      "training_or_model_imported": False, "protocol_sha256": reuse["original_protocol_sha256"],
                      "manifest_sha256": MANIFEST_SHA}
    require(all(bound_status.get(key) == value for key, value in expected_bound.items()), "CPU direction bound did not pass the registered preparation")
    require(bound_status["source_sha256"] == {bound_helper: reuse["original_helper_sha256"]}, "Original CPU-bound source hash mismatch")
    science_output = safe_path(specification["science_output"], exists=False)
    require(not science_output.exists(), "Scientific output already exists; no repeat is permitted")
    invocation_marker = safe_path(ROOT / "scientific_job_started", exists=False)
    require(not invocation_marker.exists(), "Scientific invocation marker already exists; do not resubmit")
    return {"preflight_hashes": preflight_hashes, "unpack_receipt_sha256": sha256(unpack_path),
            "direction_bound_hashes": {**hashes, "artifact_sha256.json": sha256(bound / "artifact_sha256.json")},
            "direction_bound_reused_without_new_preparation": True,
            "direction_bound_original_protocol_sha256": reuse["original_protocol_sha256"],
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
    row = parsers().parse_accounting_row(result["stdout"], job_id, ",".join(ACCOUNTING_FIELDS))
    require(row is not None, "Native top-level accounting row is unavailable")
    return row


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
    state = parsers().normalize_state(row["State"])
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
        accounting = validate_settlement_record(row, native["stdout"] if native["returncode"] == 0 else None,
                                               deployment_context())
        receipt.update(accounting)
        status.update({key: accounting[key] for key in (
            "allocated_gpu_count", "actual_a100_gpu_seconds", "allocation_case", "accounting_complete")})
        if native["returncode"] != 0:
            receipt["final_scontrol_unavailable"] = "Raw failure retained; terminal sacct allocation row is the accounting authority"
        status.update(status="passed", accounting_complete=True, registered_budget_seconds=10800)
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
