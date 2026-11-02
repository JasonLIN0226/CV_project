"""Collect read-only infrastructure evidence for explicitly bound Motion jobs.

Only stdlib imports and immutable new diagnostic files. This does not submit,
cancel, retry, alter old evidence, load models, or decode scientific arrays.
Unknown accounting or filesystem evidence never proves that science did not run.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import stat
import subprocess
import sys
import traceback


sys.dont_write_bytecode = True
ROOT = Path("/scratch/xy3257/motion-v2-20261031-deployment3")
RUNTIME = ROOT / "runtime_retries_v2_3_1"
OWNER = "xy3257"
ACCOUNT = "csci_ga_2271-2026fa"
PARTITION = "c12m85-a100-1"
TERMINAL = {"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL", "PREEMPTED",
            "BOOT_FAIL", "DEADLINE", "OUT_OF_MEMORY", "REVOKED", "SPECIAL_EXIT"}
PREDECESSOR_SUBMISSIONS = {RUNTIME / f"attempt{attempt}/submission/submission.json": attempt
                           for attempt in (2, 3)}
PATH_KINDS = ("stdout", "stderr", "scientific_marker", "scientific_output")


def require(value, message):
    if not value:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as source:
        return json.load(source)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as destination:
        json.dump(value, destination, indent=2, ensure_ascii=False, allow_nan=False)
        destination.write("\n")


def safe_path(path):
    path = Path(path).absolute()
    require(path.is_relative_to(ROOT) and path.resolve(strict=False).is_relative_to(ROOT),
            "Diagnostic path must remain inside deployment3")
    for part in (path, *path.parents):
        if part == ROOT.parent:
            break
        require(not part.is_symlink(), "Diagnostic paths cannot traverse a symlink: " + str(part))
    return path


def discover_sacct_fields(help_text):
    """Use every exact field reported by this cluster's sacct --helpformat."""
    require(isinstance(help_text, str), "Expected textual sacct field help")
    fields = help_text.split()
    require(fields and all(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", field) for field in fields),
            "Cannot safely recognize the full native sacct field list")
    names = [field.casefold() for field in fields]
    require(len(set(names)) == len(names), "Native helpformat has duplicate field names")
    require({"jobidraw", "user", "account", "state", "nodelist"} <= set(names),
            "Native field list lacks identity/state/node evidence")
    return fields


def interpret_sacct_records(text, fields, job_id):
    """Interpret all rows without dropping steps, duplicates, or foreign rows.

    Malformation returns unknown evidence, never a guessed no-.batch result.
    This pure function performs no filesystem, subprocess, clock or model work.
    """
    result = {"parsed": False, "rows": [], "field_names": list(fields), "top_level_rows": [],
              "step_ids": [], "batch_step_ids": [], "duplicate_ids": [],
              "native_owner_verified": False, "terminal": False,
              "batch_evidence": {"state": "unknown", "reason": "accounting_not_parsed"}}
    try:
        require(isinstance(text, str) and isinstance(job_id, str) and re.fullmatch(r"[0-9]+", job_id),
                "Expected native text and one numeric job ID")
        require(fields and len(set(field.casefold() for field in fields)) == len(fields), "Ambiguous accounting columns")
        names = {field.casefold(): field for field in fields}
        require({"jobidraw", "user", "account", "state", "nodelist"} <= names.keys(), "Missing required accounting columns")
        lines = [line for line in text.splitlines() if line.strip()]
        rows = []
        for index, line in enumerate(lines):
            cells = line.split("|")
            if len(cells) == len(fields) + 1 and cells[-1] == "":
                cells.pop()
            cells = [cell.strip() for cell in cells]
            if index == 0 and [cell.casefold() for cell in cells] == [field.casefold() for field in fields]:
                continue
            require(len(cells) == len(fields), "An all-field native row is malformed or truncated")
            rows.append(dict(zip(fields, cells)))
        result["rows"] = rows
        ids = [row[names["jobidraw"]] for row in rows]
        require(all(value == job_id or value.startswith(job_id + ".") for value in ids),
                "All-field query returned an unexpected job identity")
        require(all(re.fullmatch(re.escape(job_id) + r"(?:\.[A-Za-z0-9_.-]+)?", value) for value in ids),
                "Native job/step identity is missing, truncated, or malformed")
        top = [row for row in rows if row[names["jobidraw"]] == job_id]
        counts = Counter(ids)
        duplicates = sorted(value for value, count in counts.items() if count > 1)
        steps = [value for value in ids if value != job_id]
        batches = [value for value in steps if value.startswith(job_id + ".batch")]
        result.update(parsed=True, top_level_rows=top, step_ids=steps,
                      batch_step_ids=batches, duplicate_ids=duplicates)
        require(not duplicates, "Duplicate native job/step rows are ambiguous")
        require(len(top) == 1, "Expected one unique top-level allocation row")
        owner_matches = top[0][names["user"]] == OWNER and top[0][names["account"]] == ACCOUNT
        result["native_owner_verified"] = owner_matches
        require(owner_matches, "Native job owner/account does not match the authorized account")
        state_text = top[0][names["state"]]
        state = state_text.split()[0].rstrip("+") if state_text.strip() else ""
        require(re.fullmatch(r"[A-Z][A-Z_]*", state) is not None, "Missing or malformed native job state")
        result.update(native_state=state, terminal=state in TERMINAL,
                      allocated_node_list=top[0][names["nodelist"]])
        if batches:
            result["batch_evidence"] = {"state": "refuted", "reason": "native_batch_step_present", "step_ids": batches}
        elif state not in TERMINAL:
            result["batch_evidence"] = {"state": "unknown", "reason": "job_not_terminal; a batch step could still start"}
        else:
            result["batch_evidence"] = {"state": "proven", "reason": "complete successful all-field --duplicates query has a unique authorized terminal allocation and no .batch row"}
    except (TypeError, ValueError, KeyError) as error:
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
        result["batch_evidence"] = {"state": "unknown", "reason": str(error)}
    return result


def assess_start_evidence(accounting, path_probes):
    """Combine native and filesystem facts without treating errors as absence."""
    by_kind = {probe["kind"]: probe for probe in path_probes}

    def absent_fact(kinds):
        evidence = [by_kind.get(kind) for kind in kinds]
        if any(probe is not None and probe.get("exists") is True for probe in evidence):
            return {"state": "refuted", "reason": "at least one expected path exists", "evidence": evidence}
        if any(probe is None or probe.get("exists") is None or probe.get("state") == "unknown" for probe in evidence):
            return {"state": "unknown", "reason": "one or more path probes unavailable", "evidence": evidence}
        if all(probe.get("exists") is False and probe.get("state") == "absent" for probe in evidence):
            return {"state": "proven", "reason": "all listed paths confirmed absent", "evidence": evidence}
        return {"state": "unknown", "reason": "inconsistent path probe", "evidence": evidence}

    facts = {"no_batch_step": accounting.get("batch_evidence", {"state": "unknown", "reason": "missing accounting evidence"}),
             "stdout_stderr_absent": absent_fact(("stdout", "stderr")),
             "scientific_marker_absent": absent_fact(("scientific_marker",)),
             "scientific_output_absent": absent_fact(("scientific_output",))}
    proven = all(fact.get("state") == "proven" for fact in facts.values())
    eligible = proven and accounting.get("native_owner_verified") is True and accounting.get("terminal") is True
    return {"four_facts": facts, "retry_eligibility": {
        "eligible": eligible, "all_four_proven": proven,
        "reason": "all four no-scientific-start facts proven for the authorized terminal job" if eligible
                  else "no retry eligibility inferred: missing, contradictory, or unknown scientific-start evidence",
        "node_or_config_diagnostic_errors_do_not_block": True,
        "submission_authorized_by_this_collector": False}}


def path_probe(kind, path):
    result = {"kind": kind, "path": str(path), "checked_utc": now(), "exists": None,
              "state": "unknown", "error": None}
    try:
        path = safe_path(path)
        info = path.lstat()
        result.update(exists=True, state="present", kind_on_disk="directory" if stat.S_ISDIR(info.st_mode) else "file_or_other")
    except FileNotFoundError:
        result.update(exists=False, state="absent")
    except (OSError, ValueError) as error:
        result.update(error_type=type(error).__name__, error=str(error))
    return result


def record_command(output, records, label, argv, skipped_reason=None):
    record = {"name": label, "argv": list(argv), "started_utc": now(), "returncode": None,
              "stdout": "", "stderr": "", "timed_out": False, "error": None}
    if skipped_reason is not None:
        record.update(skipped=True, error=skipped_reason)
    else:
        try:
            completed = subprocess.run(argv, text=True, capture_output=True, check=False, timeout=60)
            record.update(returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)
        except subprocess.TimeoutExpired as error:
            def text(value):
                return value.decode(errors="replace") if isinstance(value, bytes) else (value or "")
            record.update(timed_out=True, stdout=text(error.stdout), stderr=text(error.stderr),
                          error_type=type(error).__name__, error=str(error))
        except OSError as error:
            record.update(error_type=type(error).__name__, error=str(error))
    record["finished_utc"] = now()
    record["raw_files"] = {}
    for stream in ("stdout", "stderr"):
        path = output / (label + "." + stream + ".txt")
        with path.open("x", encoding="utf-8") as destination:
            destination.write(record[stream])
        record["raw_files"][stream] = {"path": path.name, "sha256": sha256(path)}
    records.append(record)
    print(json.dumps({"probe": label, "returncode": record["returncode"], "error": record["error"]}), flush=True)
    return record


def parse_control_fields(text):
    matches = list(re.finditer(r"(?<!\S)([A-Za-z][A-Za-z0-9_/:]*)=", text))
    require(matches and not text[:matches[0].start()].strip(), "Cannot recognize native scontrol fields")
    fields = {}
    for index, match in enumerate(matches):
        key = match.group(1)
        require(key not in fields, "Duplicate native scontrol field")
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        fields[key] = text[match.end():end].strip()
    return fields


def verify_allowlist(args):
    require(pwd.getpwuid(os.getuid()).pw_name == OWNER, "Collector must run as the authorized xy3257 account")
    require(ROOT.is_dir() and ROOT.resolve() == ROOT and ROOT.stat().st_uid == os.getuid(),
            "The registered deployment3 root is missing, symlinked, or owned by another account")
    require(re.fullmatch(r"[0-9]+", args.job_id) is not None, "Expected one numeric job ID")
    require(re.fullmatch(r"b-[0-9]+-[0-9]+", args.node) is not None, "Node must have the observed native node syntax")
    if args.job_id == "16550":
        require(args.node == "b-19-12" and args.predecessor_submission is None, "Initial diagnosis is limited to job16550/nodeb-19-12")
        return {"mode": "explicit_initial_job", "job_id": "16550", "node": "b-19-12"}
    require(args.predecessor_submission is not None, "Future jobs require a bound predecessor submission receipt")
    source = safe_path(args.predecessor_submission)
    require(source in PREDECESSOR_SUBMISSIONS, "Predecessor receipt is outside the explicit attempt2/attempt3 allowlist")
    receipt = read_json(source)
    status = read_json(source.parent / "status.json")
    require(receipt.get("job_id") == args.job_id and receipt.get("owner") == OWNER
            and receipt.get("attempt") == PREDECESSOR_SUBMISSIONS[source]
            and receipt.get("sbatch_attempted") is True
            and status.get("receipt_sha256") == sha256(source), "Future job receipt does not bind the authorized job/account")
    return {"mode": "bound_future_submission", "receipt_path": str(source),
            "receipt_sha256": sha256(source), "attempt": receipt["attempt"], "job_id": args.job_id}


def execute(args):
    binding = verify_allowlist(args)
    output = safe_path(args.output)
    require(output == RUNTIME / ("diagnosis_" + args.job_id), "Use the exact registered fresh diagnosis_JOB directory")
    require(not output.exists() and not output.is_symlink(), "Diagnosis output already exists; never overwrite it")
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"schema_version": 1, "started_utc": now(), "job_id": args.job_id, "node": args.node,
               "owner": OWNER, "owner_uid": os.getuid(), "account": ACCOUNT, "partition": PARTITION,
               "owner_verified_locally": True, "job_binding": binding,
               "source_path": str(Path(__file__).absolute()), "source_sha256": sha256(Path(__file__)),
               "commands": [], "diagnostic_errors": [], "scientific_imports": False,
               "scientific_arrays_decoded": False, "scientific_runner_invocations": 0,
               "job_submission_or_cancellation": False, "file_deletion": False, "original_files_modified": False}
    accounting = {"native_owner_verified": False, "terminal": False,
                  "batch_evidence": {"state": "unknown", "reason": "native query incomplete"}}
    try:
        help_record = record_command(output, receipt["commands"], "01_sacct_helpformat", ["sacct", "--helpformat"])
        fields, discovery_error = [], None
        try:
            require(help_record["returncode"] == 0, "sacct --helpformat failed")
            fields = discover_sacct_fields(help_record["stdout"])
        except (ValueError, TypeError) as error:
            discovery_error = str(error)
        receipt["sacct_fields"] = fields
        all_args = ["sacct", "-j", args.job_id, "--duplicates", "--starttime=2026-10-31T00:00:00",
                    "--noheader", "--parsable2"]
        if fields:
            all_args.append("--format=" + ",".join(fields))
        all_record = record_command(output, receipt["commands"], "02_sacct_all_fields_all_steps", all_args,
                                    skipped_reason=discovery_error)
        if all_record["returncode"] == 0:
            accounting = interpret_sacct_records(all_record["stdout"], fields, args.job_id)
        else:
            accounting["batch_evidence"] = {"state": "unknown", "reason": all_record.get("error") or "sacct all-field --duplicates query failed"}
        receipt["accounting_evidence"] = accounting
        job_record = record_command(output, receipt["commands"], "03_scontrol_job",
                                    ["scontrol", "show", "job", "--oneliner", args.job_id])
        control = {}
        if job_record["returncode"] == 0:
            try:
                control = parse_control_fields(job_record["stdout"])
                require(control.get("JobId") == args.job_id and control.get("UserId", "").split("(", 1)[0] == OWNER
                        and control.get("Account") == ACCOUNT, "Native scontrol job identity is not authorized")
                receipt["native_job_fields"] = control
            except (ValueError, TypeError) as error:
                control = {}
                receipt["diagnostic_errors"].append({"probe": "scontrol_job", "error": str(error)})
        node_matches = binding["mode"] == "explicit_initial_job" or (
            accounting.get("native_owner_verified") is True and accounting.get("allocated_node_list") == args.node) or (
            bool(control) and args.node in {control.get("NodeList"), control.get("BatchHost")})
        receipt["node_binding"] = {"verified": bool(node_matches), "requested_node": args.node,
                                   "source": "explicit user allowlist or current owner-bound allocated NodeList/BatchHost; never SchedNodeList"}
        node_record = record_command(output, receipt["commands"], "04_scontrol_node",
                                     ["scontrol", "show", "node", args.node],
                                     skipped_reason=None if node_matches else "Future node cannot be bound to an authorized allocated job node")
        receipt["node_diagnostics"] = {"states": re.findall(r"(?:^|\s)State=([^\s]+)", node_record["stdout"]),
                                      "reason_lines": [line.strip() for line in node_record["stdout"].splitlines() if "Reason=" in line]}
        config_record = record_command(output, receipt["commands"], "05_scontrol_config", ["scontrol", "show", "config"])
        receipt["resume_configuration"] = {key: re.findall(r"^\s*" + key + r"\s*=\s*(.*?)\s*$", config_record["stdout"], flags=re.MULTILINE)
                                           for key in ("ResumeTimeout", "ResumeProgram")}
        record_command(output, receipt["commands"], "06_sinfo_partition_nodes",
                       ["sinfo", "-p", PARTITION, "-N", "-o", "%N %T %E"])
    except Exception as error:
        receipt["collection_exception"] = {"error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()}
    paths = {"stdout": ROOT / f"scientific-job-{args.job_id}.stdout.log",
             "stderr": ROOT / f"scientific-job-{args.job_id}.stderr.log",
             "scientific_marker": ROOT / "scientific_job_started",
             "scientific_output": ROOT / "repo/results/motion_v2/v2_3_1/step2_gpu_attempt1"}
    receipt["path_probes"] = [path_probe(kind, paths[kind]) for kind in PATH_KINDS]
    receipt.update(assess_start_evidence(accounting, receipt["path_probes"]))
    for record in receipt["commands"]:
        if record["returncode"] != 0:
            receipt["diagnostic_errors"].append({"probe": record["name"], "returncode": record["returncode"],
                                                  "error": record["error"], "stderr": record["stderr"]})
    unknowns = any(fact["state"] == "unknown" for fact in receipt["four_facts"].values())
    receipt["status"] = "collected_with_unknowns" if unknowns or receipt["diagnostic_errors"] or "collection_exception" in receipt else "collected"
    receipt["finished_utc"] = now()
    write_json(output / "receipt.json", receipt)
    status = {"status": receipt["status"], "job_id": args.job_id, "node": args.node, "owner": OWNER,
              "retry_eligibility": receipt["retry_eligibility"], "four_fact_states": {key: value["state"] for key, value in receipt["four_facts"].items()},
              "receipt_sha256": sha256(output / "receipt.json"), "diagnostic_error_count": len(receipt["diagnostic_errors"]),
              "scientific_runner_invocations": 0, "job_submission_or_cancellation": False}
    write_json(output / "status.json", status)
    names = sorted(path.name for path in output.iterdir() if path.is_file())
    with (output / "SHA256SUMS").open("x", encoding="utf-8") as destination:
        for name in names:
            destination.write(sha256(output / name) + "  " + name + "\n")
    print(json.dumps(status, ensure_ascii=False, allow_nan=False), flush=True)
    # A diagnostic failure is retained evidence, not authorization to retry.
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predecessor-submission", type=Path)
    return execute(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
