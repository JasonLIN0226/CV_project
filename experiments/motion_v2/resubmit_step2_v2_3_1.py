"""External runtime orchestration for the explicitly authorized frozen retry.

The scientific checkout remains at 6dc5ed2. This file is delivered outside that
checkout, uses its frozen validators, and never edits its wrapper or results.
Each attempt consumes a fresh submission directory before its sole sbatch call.
An unknown submission outcome never permits another call or automatic retry.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
import traceback


sys.dont_write_bytecode = True
SCIENTIFIC_HEAD = "6dc5ed253dfa305b26be8f19d40edcd6a5878ea7"
DEPLOYMENT = Path("/scratch/xy3257/motion-v2-20261031-deployment3")
SCIENTIFIC_REPO = DEPLOYMENT / "repo"
RUNTIME_ROOT = DEPLOYMENT / "runtime_retries_v2_3_1"
DEPLOYMENT_PROTOCOL = "experiments/motion_v2/protocol_step2_deployment_v2_3_1.json"
FROZEN_SUBMIT = "experiments/motion_v2/submit_step2_v2_3_1.py"
FROZEN_SUBMIT_SHA = "970533810a79155525474749e4ae41f555a60d0c1abd2bd0554ac958b283075e"
FROZEN_CHECKER = "experiments/motion_v2/check_deployment_v2_3_1.py"
FACTS = ("no_batch_step", "stdout_stderr_absent", "scientific_marker_absent", "scientific_output_absent")
TERMINAL = {"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL", "PREEMPTED",
            "BOOT_FAIL", "DEADLINE", "OUT_OF_MEMORY", "REVOKED", "SPECIAL_EXIT"}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as source:
        return json.load(source)


def safe_path(value, parent, exists=True):
    path = Path(value).absolute()
    require(path.is_relative_to(parent), "Path is outside its registered root: " + str(path))
    require(all(not item.is_symlink() for item in (path, *path.parents)), "Symlink path is forbidden: " + str(path))
    if exists:
        require(path.exists(), "Missing registered path: " + str(path))
    return path


def path_absent(path):
    """A dangling symlink or any existing inode also forbids a retry."""
    try:
        Path(path).lstat()
    except FileNotFoundError:
        return True
    return False


def validate_retry_admission(attempt, history, *, scientific_head,
                             frozen_source_verified, frozen_preparation_verified):
    """Pure decision over already verified native facts; no I/O or submission."""
    require(type(attempt) is int and attempt in (2, 3), "Only attempts 2 and 3 are authorized")
    require(scientific_head == SCIENTIFIC_HEAD, "Scientific HEAD must remain exactly frozen")
    require(frozen_source_verified is True and frozen_preparation_verified is True,
            "Frozen source and preparation verification are mandatory")
    require(isinstance(history, list) and len(history) == attempt - 1,
            "History must include every preceding attempt without skips")
    ids = set()
    total = 0
    for ordinal, row in enumerate(history, 1):
        require(isinstance(row, dict), "Malformed prior-attempt record")
        require(type(row.get("attempt")) is int and row["attempt"] == ordinal,
                "Prior attempts must be consecutive from the original attempt")
        job_id = row.get("job_id")
        require(isinstance(job_id, str) and re.fullmatch(r"[0-9]+", job_id) is not None
                and job_id not in ids, "Missing or duplicate prior job ID")
        ids.add(job_id)
        require(row.get("terminal") is True, "Every earlier job must have a confirmed terminal state")
        require(type(row.get("restarts")) is int and row["restarts"] == 0, "Every earlier job must have Restarts=0")
        seconds = row.get("actual_a100_gpu_seconds")
        require(type(seconds) is int and seconds >= 0, "Native GPU seconds must be a nonnegative integer")
        if ordinal == 1:
            require(job_id == "16550" and seconds == 769, "Original attempt/accounting must remain 16550 / 769 seconds")
        require(all(row.get(fact) is True for fact in FACTS),
                "Retry execution evidence is refuted or unknown; root review is required before any submission")
        total += seconds
    require(total + 10800 <= 14400, "A complete 10800-second reservation would exceed the cumulative 14400-second cap")
    return {"eligible": True, "attempt": attempt, "cumulative_a100_gpu_seconds": total,
            "maximum_additional_a100_gpu_seconds": 10800,
            "projected_cumulative_a100_gpu_seconds": total + 10800,
            "user_total_cap_seconds": 14400, "maximum_total_attempts": 3}


def inspect_duplicate_accounting(text, job_id, columns_csv, parser):
    """Pure validation of the complete --duplicates output, including batch rows."""
    require(all(isinstance(value, str) for value in (text, job_id, columns_csv)), "Native accounting inputs must be strings")
    row = parser.parse_accounting_row(text, job_id, columns_csv)
    require(row is not None, "The exact native top-level allocation row is missing")
    columns = [name.strip().split("%", 1)[0] for name in columns_csv.split(",")]
    require("JobIDRaw" in columns, "Raw duplicate accounting must name JobIDRaw")
    index = columns.index("JobIDRaw")
    batch_ids = []
    for line in text.splitlines():
        if not line.strip():
            continue
        cells = [value.strip() for value in line.split("|")]
        if len(cells) == len(columns) + 1 and cells[-1] == "":
            cells.pop()
        require(len(cells) == len(columns), "Malformed row in complete duplicate accounting")
        identity = cells[index]
        require(identity == job_id or identity.startswith(job_id + "."), "Unexpected job identity in duplicate accounting")
        if identity == job_id + ".batch" or identity.startswith(job_id + ".batch."):
            batch_ids.append(identity)
    return {"allocation_row": row, "batch_step_ids": batch_ids, "no_batch_step": not batch_ids}


def retained_control_after_purge(lookup, fresh_row, saved_row, saved_control_text,
                                 expected_control_sha256, parser):
    """Pure, narrowly bounded use of retained official control evidence."""
    require(type(lookup.get("returncode")) is int and lookup["returncode"] != 0
            and lookup.get("timed_out") is not True and not lookup.get("stdout", "").strip(),
            "Retained control is only permitted after an explicit expired-job lookup")
    require(re.fullmatch(r"(?:slurm_load_jobs error: |scontrol: error: )?Invalid job id specified",
                         lookup.get("stderr", "").strip()) is not None,
            "Only an explicit Invalid job id response permits retained evidence")
    require(fresh_row == saved_row, "Fresh allocation accounting differs from preserved terminal accounting")
    require(isinstance(expected_control_sha256, str)
            and re.fullmatch(r"[0-9a-f]{64}", expected_control_sha256) is not None
            and hashlib.sha256(saved_control_text.encode("utf-8")).hexdigest() == expected_control_sha256,
            "Preserved terminal scontrol SHA256 mismatch")
    fields = parser.parse_control(saved_control_text)
    require(fields.get("JobId") == fresh_row.get("JobIDRaw"), "Preserved terminal control job ID differs")
    state = parser.normalize_state(fresh_row.get("State", ""))
    require(state in TERMINAL and parser.normalize_state(fields.get("JobState", "")) == state,
            "Preserved control must agree with fresh terminal accounting")
    require(fields.get("Restarts") == "0", "Preserved terminal Restarts must be zero")
    return {"control_text": saved_control_text, "control_sha256": expected_control_sha256,
            "evidence_kind": "preserved_terminal_scontrol_after_explicit_purge"}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_inventory(directory):
    """Verify an existing receipt tree without modifying it."""
    directory = safe_path(directory, DEPLOYMENT)
    inventory = safe_path(directory / "SHA256SUMS", directory)
    mapping = {}
    for line in inventory.read_text(encoding="utf-8").splitlines():
        checksum, name = line.split("  ", 1)
        require(re.fullmatch(r"[0-9a-f]{64}", checksum) is not None and name not in mapping,
                "Malformed or duplicate receipt checksum")
        path = safe_path(directory / name, directory)
        require(path.is_file() and sha256(path) == checksum, "Receipt checksum mismatch: " + name)
        mapping[name] = checksum
    actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()}
    require(actual == set(mapping) | {"SHA256SUMS"}, "Receipt inventory does not cover every file")
    require({"status.json"} <= mapping.keys(), "Receipt status is not hashed")
    return {**mapping, "SHA256SUMS": sha256(inventory)}


def bootstrap(args):
    require(sys.flags.isolated == 1 and sys.flags.dont_write_bytecode == 1, "Use Python -I -B")
    require(Path(args.scientific_repo).absolute() == SCIENTIFIC_REPO, "Use the exact frozen scientific repository")
    require(re.fullmatch(r"[0-9a-f]{64}", args.protocol_sha256) is not None, "Require the registered runtime protocol SHA256")
    protocol_path = safe_path(args.protocol, RUNTIME_ROOT)
    require(sha256(protocol_path) == args.protocol_sha256, "Runtime protocol SHA256 mismatch")
    protocol = read_json(protocol_path)
    expected = {"status": "FROZEN_RUNTIME_RESUBMISSION_PROTOCOL", "registered_before_run": True,
                "scientific_repo": str(SCIENTIFIC_REPO), "scientific_head": SCIENTIFIC_HEAD,
                "runtime_root": str(RUNTIME_ROOT), "maximum_total_attempts": 3,
                "per_attempt_gpu_seconds": 10800, "cumulative_cap_seconds": 14400}
    require(all(protocol.get(key) == value for key, value in expected.items()), "Runtime authorization differs from the registered scope")
    package = safe_path(RUNTIME_ROOT / "source", RUNTIME_ROOT)
    own = safe_path(Path(__file__).absolute(), package).relative_to(package).as_posix()
    source_hashes = protocol["source_sha256"]
    require(own in source_hashes, "Runtime registration does not lock this external helper")
    for relative, checksum in source_hashes.items():
        require(sha256(safe_path(package / relative, package)) == checksum, "Runtime source SHA256 mismatch: " + relative)
    originals = protocol["original_receipt_sha256"]
    existing = set()
    for directory in (DEPLOYMENT / "submission", DEPLOYMENT / "settlement"):
        for path in safe_path(directory, DEPLOYMENT).rglob("*"):
            require(not path.is_symlink(), "Original receipt contains a symlink")
            if path.is_file():
                existing.add(path.relative_to(DEPLOYMENT).as_posix())
    require(existing == set(originals), "Runtime registration must preserve every original submission/settlement file")
    for relative, checksum in originals.items():
        require(sha256(safe_path(DEPLOYMENT / relative, DEPLOYMENT)) == checksum,
                "Original immutable receipt changed: " + relative)
    helper_path = safe_path(SCIENTIFIC_REPO / FROZEN_SUBMIT, SCIENTIFIC_REPO)
    require(sha256(helper_path) == FROZEN_SUBMIT_SHA, "Frozen submit helper SHA256 mismatch before import")
    committed = subprocess.run(["git", "show", SCIENTIFIC_HEAD + ":" + FROZEN_SUBMIT], cwd=SCIENTIFIC_REPO,
                               check=True, capture_output=True, timeout=30).stdout
    require(committed == helper_path.read_bytes(), "Frozen submit helper differs from exact scientific commit")
    frozen = load_module(helper_path, "motion_v231_retry_frozen_submit")
    old_args = SimpleNamespace(protocol=SCIENTIFIC_REPO / DEPLOYMENT_PROTOCOL, expected_head=SCIENTIFIC_HEAD)
    deployment, science, wrapper, registration = frozen.registration(old_args)
    return frozen, protocol, deployment, science, wrapper, registration


def submitted_attempt(frozen, attempt, runtime_protocol_sha, registration):
    directory = DEPLOYMENT / "submission" if attempt == 1 else RUNTIME_ROOT / f"attempt{attempt}" / "submission"
    inventory = verify_inventory(directory)
    submitted = read_json(directory / "submission.json")
    status = read_json(directory / "status.json")
    require(status.get("receipt_sha256") == inventory.get("submission.json"), "Submission receipt hash chain differs")
    require(status.get("status") == "submitted_verified" and submitted.get("sbatch_attempted") is True,
            "Prior submission is incomplete or unknown; no retry is permitted")
    require(submitted.get("registration") == registration, "Prior submission scientific registration differs")
    job_id = submitted.get("job_id")
    require(isinstance(job_id, str) and re.fullmatch(r"[0-9]+", job_id) is not None, "Prior job ID is missing")
    if attempt == 1:
        require(job_id == "16550", "Original attempt must remain job 16550")
    else:
        require(submitted.get("attempt") == attempt and submitted.get("runtime_protocol_sha256") == runtime_protocol_sha,
                "Prior retry receipt is from a different authorized attempt/protocol")
        settlement = directory.parent / "settlement"
        hashes = verify_inventory(settlement)
        settled_status = read_json(settlement / "status.json")
        settled = read_json(settlement / "accounting.json")
        require(settled_status.get("status") == "passed" and settled_status.get("receipt_sha256") == hashes.get("accounting.json")
                and settled.get("job_id") == job_id and settled.get("submission_receipt_sha256") == inventory["submission.json"],
                "Previous retry must have a complete matching native settlement")
    return {"attempt": attempt, "job_id": job_id, "submission_path": str(directory / "submission.json"),
            "submission_sha256": inventory["submission.json"], "submission_inventory": inventory}


def capture(frozen, output, receipt, name, argv, require_success=True):
    result = frozen.command(argv)
    receipt.setdefault("native_commands", []).append(result)
    frozen.save_native(output, name, result)
    if require_success:
        require(result["returncode"] == 0, "Native command failed; cannot infer retry eligibility: " + repr(argv))
    return result


def retained_terminal_control(frozen, entry, lookup, fresh_row):
    directory = DEPLOYMENT / "settlement" if entry["attempt"] == 1 else RUNTIME_ROOT / f"attempt{entry['attempt']}" / "settlement"
    inventory = verify_inventory(directory)
    status, saved = read_json(directory / "status.json"), read_json(directory / "accounting.json")
    require(status.get("status") == "passed" and status.get("receipt_sha256") == inventory.get("accounting.json")
            and saved.get("job_id") == entry["job_id"]
            and saved.get("submission_receipt_sha256") == entry["submission_sha256"],
            "Preserved terminal settlement does not match its immutable submission")
    path = safe_path(directory / "final_scontrol.stdout.txt", directory)
    text = path.read_bytes().decode("utf-8")
    require(saved.get("scontrol", {}).get("returncode") == 0 and saved["scontrol"].get("stdout") == text,
            "Preserved settlement did not contain successful official scontrol evidence")
    retained = retained_control_after_purge(lookup, fresh_row, saved["allocation_row"], text,
                                           inventory["final_scontrol.stdout.txt"], frozen.parsers())
    return {**retained, "source_path": str(path), "settlement_receipt_sha256": inventory["accounting.json"]}


def fresh_history(frozen, output, receipt, entries):
    history = []
    fields = ",".join(frozen.ACCOUNTING_FIELDS)
    formatting = ",".join(name + "%40" if name in {"State", "JobName"} else name for name in frozen.ACCOUNTING_FIELDS)
    for entry in entries:
        job_id, ordinal = entry["job_id"], entry["attempt"]
        accounting = capture(frozen, output, receipt, f"prior_{ordinal}_sacct_duplicates",
                             ["sacct", "--duplicates", "-j", job_id, "--starttime=2026-10-31T00:00:00",
                              "--noheader", "--parsable2", "--format=" + formatting])
        inspected = inspect_duplicate_accounting(accounting["stdout"], job_id, fields, frozen.parsers())
        control = capture(frozen, output, receipt, f"prior_{ordinal}_scontrol",
                          ["scontrol", "show", "job", "--oneliner", job_id], require_success=False)
        if control["returncode"] == 0:
            control_evidence = {"control_text": control["stdout"], "evidence_kind": "fresh_native_scontrol"}
        else:
            control_evidence = retained_terminal_control(frozen, entry, control, inspected["allocation_row"])
        native = frozen.validate_settlement_record(inspected["allocation_row"], control_evidence["control_text"], frozen.deployment_context())
        controls = native["final_fields"]
        require(controls is not None and controls.get("Restarts") == "0", "Previous native Restarts must be confirmed zero")
        require(frozen.parsers().normalize_state(controls["JobState"]) == native["native_state"],
                "Control/accounting terminal states disagree")
        require(frozen.parsers().gpu_count(controls.get("AllocTRES", "")) == native["allocated_gpu_count"],
                "Control/accounting allocated GPU counts disagree")
        paths = {"stdout": DEPLOYMENT / f"scientific-job-{job_id}.stdout.log",
                 "stderr": DEPLOYMENT / f"scientific-job-{job_id}.stderr.log",
                 "marker": DEPLOYMENT / "scientific_job_started",
                 "science_output": Path(frozen.EXPECTED_SUBMISSION["science_output"])}
        probes = {key: {"path": str(path), "absent": path_absent(path)} for key, path in paths.items()}
        history.append({**entry, "terminal": native["native_state"] in TERMINAL,
                        "native_state": native["native_state"], "restarts": int(controls["Restarts"]),
                        "actual_a100_gpu_seconds": native["actual_a100_gpu_seconds"],
                        "no_batch_step": inspected["no_batch_step"], "batch_step_ids": inspected["batch_step_ids"],
                        "stdout_stderr_absent": probes["stdout"]["absent"] and probes["stderr"]["absent"],
                        "scientific_marker_absent": probes["marker"]["absent"],
                        "scientific_output_absent": probes["science_output"]["absent"],
                        "path_probes": probes, "native_accounting": native,
                        "control_evidence": control_evidence})
    receipt["verified_native_history"] = history
    return history


def verify_diagnosis(args, predecessor):
    path = safe_path(args.diagnosis, RUNTIME_ROOT)
    require(path == RUNTIME_ROOT / ("diagnosis_" + predecessor["job_id"]) / "receipt.json",
            "Use the collector receipt for the immediate predecessor")
    require(re.fullmatch(r"[0-9a-f]{64}", args.diagnosis_sha256) is not None
            and sha256(path) == args.diagnosis_sha256, "Collector receipt SHA256 differs")
    inventory = verify_inventory(path.parent)
    status, diagnosis = read_json(path.parent / "status.json"), read_json(path)
    require(status.get("receipt_sha256") == inventory.get("receipt.json"), "Collector receipt hash chain differs")
    require(status.get("status") in {"collected", "collected_with_unknowns"}, "Collector did not finish its evidence pass")
    require(diagnosis.get("job_id") == predecessor["job_id"] and diagnosis.get("owner") == "xy3257",
            "Collector identity is not the expected previous job")
    facts = diagnosis.get("four_facts", {})
    require(all(facts.get(key, {}).get("state") == "proven" for key in FACTS),
            "Collector execution evidence is not fully established; root review is required")
    # Cause/node/config diagnostic errors do not gate an explicitly authorized retry.
    return {"path": str(path), "sha256": args.diagnosis_sha256, "inventory_sha256": inventory,
            "job_id": predecessor["job_id"], "four_facts": facts,
            "diagnostic_errors": diagnosis.get("diagnostic_errors", []),
            "cause_or_node_diagnostics_used_as_admission_gate": False}


def refresh_environment(frozen, deployment, output):
    path = SCIENTIFIC_REPO / FROZEN_CHECKER
    require(sha256(frozen.committed(path)) == deployment["source_sha256"][FROZEN_CHECKER], "Frozen environment checker SHA256 differs")
    checker = load_module(path, "motion_v231_retry_frozen_environment")
    checker.protect_old_deployments()
    directory = output.parent / "preflight_environment"
    require(path_absent(directory), "Fresh retry environment receipt directory already exists")
    directory.mkdir(exist_ok=False)
    caches = checker.configure_caches(directory)
    environment = checker.check_environment()
    record = {"status": "passed", "environment": environment, "cache_environment": caches,
              "scientific_entrypoint_run": False, "model_instance_created": False,
              "installs_performed": False, "scientific_source_modified": False}
    frozen.write_json(directory / "receipt.json", record)
    return {"path": str(directory / "receipt.json"), "sha256": sha256(directory / "receipt.json")}


def refresh_data_hashes(frozen, deployment, output):
    transfer_relative = "experiments/motion_v2/transfer_dataset.py"
    path = SCIENTIFIC_REPO / transfer_relative
    require(sha256(frozen.committed(path)) == deployment["source_sha256"][transfer_relative],
            "Frozen data transfer helper SHA256 differs")
    transfer = load_module(path, "motion_v231_retry_frozen_transfer")
    specification = deployment["dataset_transfer"]
    dataset = safe_path(specification["new_dataset"], SCIENTIFIC_REPO)
    expected = transfer.entries(dataset)
    require(len(expected) == specification["artifact_files"] == 2432, "The original train/validation artifact count changed")
    for relative, checksum in sorted(expected.items()):
        require(sha256(safe_path(dataset / relative, dataset)) == checksum, "Actual dataset byte SHA256 changed: " + relative)
    archive = safe_path(specification["archive"], Path("/scratch/xy3257"))
    require(archive.stat().st_size == specification["archive_bytes"]
            and sha256(archive) == specification["archive_sha256"], "The immutable original data archive changed")
    record = {"status": "passed", "dataset": str(dataset), "artifact_files": 2432,
              "manifest_sha256": sha256(dataset / "manifest.json"),
              "expected_artifact_hash_map_sha256": hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest(),
              "archive": str(archive), "archive_sha256": specification["archive_sha256"],
              "archive_bytes": specification["archive_bytes"], "generation_or_decoding": False,
              "test_access": False, "source_or_data_modified": False}
    target = output / "data_reverification.json"
    frozen.write_json(target, record)
    return {"path": str(target), "sha256": sha256(target), "artifact_files": 2432}


def submit(args, frozen, protocol, deployment, science, wrapper, registration):
    require(type(args.attempt) is int and args.attempt in (2, 3), "Only attempts 2 and 3 are authorized")
    output = safe_path(args.output, RUNTIME_ROOT, exists=False)
    require(output == RUNTIME_ROOT / f"attempt{args.attempt}" / "submission", "Use the exact fresh attempt submission directory")
    require(path_absent(output), "This attempt was already consumed; an unknown outcome cannot be retried")
    if args.attempt == 2:
        require(path_absent(RUNTIME_ROOT / "attempt3"), "Attempt ordering is inconsistent")
    output.parent.mkdir(exist_ok=True)
    output.mkdir(exist_ok=False)
    receipt = {"started_utc": frozen.now(), "attempt": args.attempt, "owner": "xy3257",
               "runtime_protocol_sha256": args.protocol_sha256, "registration": registration,
               "sbatch_attempted": False, "job_id": None, "automatic_retry_on_unknown_outcome": False}
    status = {"status": "stopped_before_submission", "attempt": args.attempt, "sbatch_attempts": 0,
              "scientific_success_inferred": False, "automatic_retry_on_unknown_outcome": False}
    code = 1
    try:
        entries = [submitted_attempt(frozen, ordinal, args.protocol_sha256, registration)
                   for ordinal in range(1, args.attempt)]
        receipt["diagnosis"] = verify_diagnosis(args, entries[-1])
        receipt["preparation"] = frozen.preparation_receipts(deployment, science, registration)
        receipt["environment_check"] = refresh_environment(frozen, deployment, output)
        receipt["fresh_data_verification"] = refresh_data_hashes(frozen, deployment, output)
        history = fresh_history(frozen, output, receipt, entries)
        receipt["admission"] = validate_retry_admission(
            args.attempt, history, scientific_head=registration["head_commit"],
            frozen_source_verified=True, frozen_preparation_verified=True)
        # Recheck all entry guards immediately before consuming the sbatch marker.
        for row in history:
            require(all(path_absent(probe["path"]) for probe in row["path_probes"].values()),
                    "A scientific marker/output/log appeared before submission")
        argv = ["sbatch", "--parsable", "--no-requeue", str(wrapper)]
        removed = sorted(key for key in os.environ if key.startswith("SBATCH_"))
        environment = {key: value for key, value in os.environ.items() if not key.startswith("SBATCH_")}
        frozen.write_json(output / "submission_started.json", {
            "created_utc": frozen.now(), "attempt": args.attempt, "command": argv,
            "registration": registration, "runtime_protocol_sha256": args.protocol_sha256,
            "admission": receipt["admission"], "removed_sbatch_environment_keys": removed,
            "sbatch_attempt_limit": 1, "unknown_outcome_requires_stop": True})
        receipt["sbatch_attempted"] = True
        status.update(status="submission_outcome_unknown", sbatch_attempts=1)
        result = frozen.command(argv, timeout=60, env=environment)  # Sole sbatch call.
        receipt["sbatch"] = result
        frozen.save_native(output, "sbatch", result)
        require(result["returncode"] == 0, "sbatch failed/timed out; outcome is never retried")
        match = re.fullmatch(r"([0-9]+)(?:;([A-Za-z0-9_.-]+))?", result["stdout"].strip())
        require(match is not None, "Unrecognized sbatch outcome; this attempt remains consumed")
        job_id = match.group(1)
        receipt.update(job_id=job_id, cluster=match.group(2))
        status.update(status="stopped_after_submission", job_id=job_id)
        require(job_id not in {entry["job_id"] for entry in entries}, "New submission returned a previous job ID")
        native = capture(frozen, output, receipt, "immediate_scontrol", ["scontrol", "show", "job", "--oneliner", job_id])
        receipt["immediate_scontrol"] = native
        fields = frozen.control_fields(native, job_id)
        receipt["immediate_fields"] = fields
        receipt["requeue_observed_not_gate"] = fields.get("Requeue")
        status.update(status="submitted_verified", native_state=fields.get("JobState"), restarts=0,
                      final_accounting_pending=True)
        code = 0
    except Exception as error:
        status.update(error_type=type(error).__name__, error=str(error))
        receipt["traceback"] = traceback.format_exc()
    receipt["finished_utc"] = frozen.now()
    frozen.finish(output, "submission.json", receipt, status)
    return code


def settle(args, frozen, protocol, registration):
    require(args.attempt in (2, 3), "Only an authorized retry can be settled here")
    require(args.submission is not None, "Settlement needs its original retry submission receipt")
    submission = safe_path(args.submission, RUNTIME_ROOT)
    require(submission == RUNTIME_ROOT / f"attempt{args.attempt}" / "submission" / "submission.json",
            "Settlement must reference this attempt's original receipt")
    inventory = verify_inventory(submission.parent)
    submitted, submitted_status = read_json(submission), read_json(submission.parent / "status.json")
    require(submitted_status.get("receipt_sha256") == inventory["submission.json"]
            and submitted.get("registration") == registration
            and submitted.get("runtime_protocol_sha256") == args.protocol_sha256
            and submitted.get("attempt") == args.attempt and submitted.get("sbatch_attempted") is True,
            "Settlement submission identity or hash chain differs")
    job_id = submitted.get("job_id", "")
    require(re.fullmatch(r"[0-9]+", job_id) is not None, "Unknown submission outcome has no settleable job ID")
    output = safe_path(args.output, RUNTIME_ROOT, exists=False)
    require(output == submission.parent.parent / "settlement" and path_absent(output), "Use the fresh per-attempt settlement directory")
    output.mkdir(exist_ok=False)
    receipt = {"created_utc": frozen.now(), "job_id": job_id, "attempt": args.attempt,
               "registration": registration, "runtime_protocol_sha256": args.protocol_sha256,
               "submission_receipt_sha256": inventory["submission.json"], "scientific_success_inferred": False}
    status = {"status": "stopped_accounting_error", "job_id": job_id, "attempt": args.attempt,
              "sbatch_attempts_in_settlement": 0, "scientific_success_inferred": False}
    code = 1
    try:
        formatting = ",".join(name + "%40" if name in {"State", "JobName"} else name for name in frozen.ACCOUNTING_FIELDS)
        result = capture(frozen, output, receipt, "final_sacct_duplicates",
                         ["sacct", "--duplicates", "-j", job_id, "--starttime=2026-10-31T00:00:00",
                          "--noheader", "--parsable2", "--format=" + formatting])
        inspected = inspect_duplicate_accounting(result["stdout"], job_id, ",".join(frozen.ACCOUNTING_FIELDS), frozen.parsers())
        control = capture(frozen, output, receipt, "final_scontrol", ["scontrol", "show", "job", "--oneliner", job_id])
        native = frozen.validate_settlement_record(inspected["allocation_row"], control["stdout"], frozen.deployment_context())
        receipt.update(native)
        receipt.update(sacct=result, scontrol=control, allocation_row=inspected["allocation_row"],
                       batch_step_ids=inspected["batch_step_ids"], no_batch_step=inspected["no_batch_step"])
        previous = submitted["admission"]["cumulative_a100_gpu_seconds"]
        cumulative = previous + native["actual_a100_gpu_seconds"]
        require(cumulative <= 14400, "Final cumulative native GPU time exceeded the user cap")
        receipt["cumulative_a100_gpu_seconds"] = cumulative
        status.update(status="passed", accounting_complete=True,
                      native_state=native["native_state"], actual_a100_gpu_seconds=native["actual_a100_gpu_seconds"],
                      cumulative_a100_gpu_seconds=cumulative, registered_budget_seconds=10800)
        code = 0
    except Exception as error:
        status.update(error_type=type(error).__name__, error=str(error))
        receipt["traceback"] = traceback.format_exc()
    frozen.finish(output, "accounting.json", receipt, status)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--scientific-repo", required=True, type=Path)
    parser.add_argument("--attempt", required=True, type=int, choices=(2, 3))
    parser.add_argument("--diagnosis", type=Path)
    parser.add_argument("--diagnosis-sha256")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--settle", action="store_true")
    parser.add_argument("--submission", type=Path)
    args = parser.parse_args(argv)
    require(args.settle or (args.diagnosis is not None and args.diagnosis_sha256 is not None),
            "Submission needs the preceding collector receipt and its SHA256")
    require(args.settle or args.submission is None, "--submission is only for settlement")
    frozen, protocol, deployment, science, wrapper, registration = bootstrap(args)
    if args.settle:
        return settle(args, frozen, protocol, registration)
    return submit(args, frozen, protocol, deployment, science, wrapper, registration)


if __name__ == "__main__":
    raise SystemExit(main())
