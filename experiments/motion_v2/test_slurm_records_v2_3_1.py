"""Offline real-record replay and fake-clock tests; never call Slurm or models."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FIXTURES = {
    "16542_pending": (
        "results/motion_v2/v2_3/deployment2/remote_evidence/submission/immediate_scontrol.stdout.txt",
        "850e8e394ebe06d301666c0575e3e9c5fec01289c5a17db6d5e513ca657c532d"),
    "16512_cpu_running": (
        "results/motion_v2/v2_2/deployment1/remote_evidence/native_job_at_stop.txt",
        "3e79c9b4fd9e7e2ad172293b04f84ef73677499d05714df968af655f46ad8de7"),
    "12374_completed": (
        "results/literature_diagnosis/qualified-reader-retention-20261025/conf/cluster/sacct_12374.txt",
        "509f39267779ea64e0b5f5c6b20925b753ba0a5c9eed2f4c8a42d49561e0d4f3"),
    "16542_cancelled_sacct": (
        "results/motion_v2/v2_3/deployment2/remote_evidence/settlement/final_sacct.stdout.txt",
        "c6e562f1d80ec1ea4d4e157775087ae0c305f7004c64fd34b72f40e0731ef676"),
    "16542_cancelled_control": (
        "results/motion_v2/v2_3/deployment2/remote_evidence/settlement/final_scontrol.stdout.txt",
        "87a38c8ba2d7ec1e7b7f357f8e761b8c7c49010cc583bcc8b3ee8a7344dca604"),
}
RUNNER_COLUMNS = "JobIDRaw,State,Start,ElapsedRaw,AllocTRES"
SETTLEMENT_COLUMNS = "JobIDRaw,JobName,User,Account,Partition,State,ExitCode,Start,End,ElapsedRaw,AllocTRES,ReqTRES"
HISTORICAL_EXPECTED = {
    "account": "csci_ga_2271-2026fa", "partition": "c12m85-a100-1", "owner": "xy3257",
    "time_limit_seconds": 7200, "hard_cap_seconds": 14400,
    "wrapper_path": "/scratch/xy3257/motion-v2-20261031-deployment2/repo/experiments/motion_v2/run_step2_v2_3.sbatch",
    "workdir": "/scratch/xy3257/motion-v2-20261031-deployment2/repo",
}
BOUND_BYTES = {
    "artifact_sha256.json": "f6d943e9d67e9ff5b8af2d766eda4124adbe1c4ec678cb74ef47803323cb1a1f",
    "status.json": "8b89cd9e23f5d8a057da94693c36dfe184a6f4ae060452f81ab3d43ea4507df4",
    "direction_bound.npz": "0a99ba0570ef116a03c24775d311ba25886434e7f1e2c87637bc6e1a800427ea",
}


def load_module(name):
    spec = importlib.util.spec_from_file_location("offline_replay_" + name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runtime():
    return (load_module("slurm_records"), load_module("submit_step2_v2_3_1"),
            load_module("run_step2"))


def fixture_text(name):
    relative, expected = FIXTURES[name]
    raw = (REPO / relative).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected
    return raw.decode("utf-8")


def captured(call):
    try:
        return {"status": "passed", "result": call()}
    except (ValueError, RuntimeError) as error:
        return {"status": "rejected", "error_type": type(error).__name__, "error": str(error)}


def check_row(field, observed, requirement, scope, actual, expected="passed"):
    return {"field": field, "observed": observed, "requirement": requirement,
            "scope": scope, "actual": actual, "expected": expected,
            "matches_expected": actual == expected}


def missing_row(field, scope, reason):
    return check_row(field, None, reason, scope, "not_applicable_missing_evidence",
                     "not_applicable_missing_evidence")


def record_case(name, schema, fields, checks, validators):
    relative, checksum = FIXTURES[name]
    return {"name": name, "path": relative, "sha256": checksum, "schema": schema,
            "parsed_fields": fields, "checks": checks, "runtime_validators": validators}


def test_real_fixture_replay(runtime):
    """Use actual shared runtime validators; the JSON retains expected rejections."""
    records, submit, runner = runtime
    cases = []
    pending_text = fixture_text("16542_pending")
    pending = records.parse_control(pending_text)
    submission = captured(lambda: submit.validate_submission_control(pending_text, "16542", HISTORICAL_EXPECTED))
    live_pending = captured(lambda: runner.validate_running_control(pending_text, "16542"))
    identity_fields = {"JobId": "16542", "Restarts": "0", "Account": HISTORICAL_EXPECTED["account"],
                       "Partition": HISTORICAL_EXPECTED["partition"], "Command": HISTORICAL_EXPECTED["wrapper_path"],
                       "WorkDir": HISTORICAL_EXPECTED["workdir"], "UserId": "xy3257(4882486)"}
    checks = [check_row(key, pending.get(key), "Historical submission identity; shared submission validator",
                        "submission", "passed" if pending.get(key) == value and submission["status"] == "passed" else "failed")
              for key, value in identity_fields.items()]
    for key, value in (("CPUs/Task", "4"), ("ReqB:S:C:T", "0:0:*:*"), ("NtasksPerN:B:S:C", "1:0:*:*")):
        checks.append(check_row(key, pending.get(key), "Preserve slash/colon key independently", "parser",
                                "passed" if pending.get(key) == value else "failed"))
    checks.extend([
        check_row("NumNodes", pending.get("NumNodes"), "One node: both endpoints must equal 1", "submission/runner",
                  captured(lambda: records.require_one_node(pending["NumNodes"]))["status"]),
        check_row("TimeLimit", pending.get("TimeLimit"), "Historical request 7200 seconds", "submission",
                  "passed" if records.parse_duration(pending["TimeLimit"]) == 7200 else "failed"),
        check_row("ReqTRES", pending.get("ReqTRES"), "Exactly one requested GPU", "submission",
                  captured(lambda: records.one_gpu(pending["ReqTRES"]))["status"]),
        check_row("TresPerNode", pending.get("TresPerNode"), "Recorded typed A100 request", "submission",
                  "passed" if pending.get("TresPerNode") == "gres/gpu:a100:1" else "failed"),
        check_row("JobState", pending.get("JobState"), "PENDING is legal for submission but is not a running allocation",
                  "runner", live_pending["status"], "rejected"),
        check_row("AllocTRES", pending.get("AllocTRES"), "PENDING has no allocation; do not run science",
                  "runner", captured(lambda: records.one_gpu(pending["AllocTRES"]))["status"], "rejected"),
        check_row("StartTime", pending.get("StartTime"), "Unknown is legal in the pending submission", "submission",
                  "passed" if pending.get("StartTime") == "Unknown" and submission["status"] == "passed" else "failed"),
        check_row("RunTime", pending.get("RunTime"), "Pending job has zero elapsed runtime", "parser/runner field",
                  "passed" if records.parse_duration(pending["RunTime"]) == 0 else "failed"),
        check_row("Requeue", pending.get("Requeue"), "Evidence only, not a rejection criterion", "submission/runner",
                  "passed" if pending.get("Requeue") == "1" else "failed"),
        check_row("NumCPUs", pending.get("NumCPUs"), "Record actual CPUs; no exact-four allocated-CPU gate", "record_only",
                  "passed" if pending.get("NumCPUs") == "4" else "failed"),
    ])
    cases.append(record_case("16542_pending", "scontrol single-line key=value", pending, checks,
                             {"validate_submission_control": submission, "validate_running_control": live_pending}))

    cpu_text = fixture_text("16512_cpu_running")
    cpu = records.parse_control(cpu_text)
    live_cpu = captured(lambda: runner.validate_running_control(cpu_text, "16512"))
    cpu_gpu = captured(lambda: records.one_gpu(cpu["AllocTRES"]))
    checks = []
    for key, value in (("JobId", "16512"), ("JobState", "RUNNING"), ("Restarts", "0"),
                       ("NumNodes", "1"), ("NumCPUs", "24"), ("CPUs/Task", "4"),
                       ("ReqB:S:C:T", "0:0:*:*"), ("NtasksPerN:B:S:C", "1:0:*:*"), ("Requeue", "1")):
        checks.append(check_row(key, cpu.get(key), "Preserve actual auxiliary CPU record; CPUs>4 allowed",
                                "parser/auxiliary", "passed" if cpu.get(key) == value else "failed"))
    checks.extend([
        check_row("NumNodes", cpu["NumNodes"], "One node", "shared node parser",
                  captured(lambda: records.require_one_node(cpu["NumNodes"]))["status"]),
        check_row("ReqTRES", cpu["ReqTRES"], "CPU-only request, no GPU", "auxiliary",
                  "passed" if records.gpu_count(cpu["ReqTRES"]) == 0 else "failed"),
        check_row("AllocTRES", cpu["AllocTRES"], "CPU-only allocation must fail a scientific GPU gate", "runner",
                  cpu_gpu["status"], "rejected"),
        check_row("TimeLimit", cpu["TimeLimit"], "Auxiliary one-hour request, not the scientific wrapper limit", "auxiliary",
                  "passed" if records.parse_duration(cpu["TimeLimit"]) == 3600 else "failed"),
        check_row("StartTime", cpu["StartTime"], "Original running CPU start is retained", "parser",
                  "passed" if cpu["StartTime"] == "2026-10-31T20:42:02" else "failed"),
        check_row("RunTime", cpu["RunTime"], "Parse recorded auxiliary elapsed duration", "parser/runner field",
                  "passed" if records.parse_duration(cpu["RunTime"]) == 501 else "failed"),
    ])
    for key in ("Account", "Partition", "UserId", "Command", "WorkDir"):
        checks.append(check_row(key, cpu.get(key), "Auxiliary context is not a scientific submission identity",
                                "submission", "not_applicable_context", "not_applicable_context"))
    cases.append(record_case("16512_cpu_running", "scontrol multiline key=value", cpu, checks,
                             {"validate_running_control": live_cpu, "one_gpu": cpu_gpu}))

    history = records.parse_accounting_row(fixture_text("12374_completed"), "12374")
    history_gpu = captured(lambda: records.one_gpu(history["AllocTRES"]))
    checks = []
    for key, value in (("JobID", "12374"), ("JobName", "cv-qualified-reader-conf"),
                       ("Partition", "c12m85-a100-1"), ("State", "COMPLETED"), ("ExitCode", "0:0"),
                       ("Elapsed", "00:06:02"), ("ElapsedRaw", "362"),
                       ("Start", "2026-10-26T20:10:28"), ("End", "2026-10-26T20:16:30")):
        checks.append(check_row(key, history.get(key), "Preserve historical header and exact top-level row",
                                "parser/history", "passed" if history.get(key) == value else "failed"))
    checks.append(check_row("AllocTRES", history["AllocTRES"], "GPU count=1 despite cpu=12; generic TRES allowed",
                            "shared GPU validator", history_gpu["status"]))
    checks.append(check_row("State", history["State"], "Historical COMPLETED is not subjected to current RUNNING gate",
                            "runner live-state", "not_applicable_context", "not_applicable_context"))
    for key in ("UserId", "Account", "NumNodes", "Restarts", "TimeLimit", "RunTime", "Command", "WorkDir", "ReqTRES"):
        checks.append(missing_row(key, "submission/runner", "This historical sacct schema contains no such field"))
    checks.append(missing_row("visible CUDA device count/model", "runner physical GPU gate",
                               "Historical text fixture cannot verify the current physical A100 device"))
    cases.append(record_case("12374_completed", "pipe header with 10 columns; top-level + .batch + .extern",
                             history, checks, {"one_gpu": history_gpu}))

    cancelled = records.parse_accounting_row(fixture_text("16542_cancelled_sacct"), "16542", SETTLEMENT_COLUMNS)
    final_text = fixture_text("16542_cancelled_control")
    final_control = records.parse_control(final_text)
    settled = captured(lambda: submit.validate_settlement_record(cancelled, final_text, HISTORICAL_EXPECTED))
    checks = [check_row(key, cancelled.get(key), "Unallocated cancellation evidence", "settlement",
                        "passed" if cancelled.get(key) == value else "failed")
              for key, value in (("JobIDRaw", "16542"), ("State", "CANCELLED by 4882486"),
                                 ("Start", "None"), ("ElapsedRaw", "0"), ("AllocTRES", ""))]
    checks.append(check_row("Restarts", final_control.get("Restarts"), "Native restart count remains zero", "settlement",
                            "passed" if final_control.get("Restarts") == "0" else "failed"))
    checks.append(check_row("StartTime", final_control.get("StartTime"),
                            "scontrol cancellation timestamp does not override sacct Start=None", "settlement",
                            "passed" if final_control.get("StartTime") == final_control.get("EndTime") else "failed"))
    cases.append(record_case("16542_cancelled_sacct", "12 explicit columns, headerless top-level row",
                             cancelled, checks, {"validate_settlement_record": settled}))
    cases.append(record_case("16542_cancelled_control", "scontrol single-line terminal state", final_control,
                             [check_row("AllocTRES", final_control.get("AllocTRES"), "Zero allocated GPU at cancellation", "settlement",
                                        "passed" if records.gpu_count(final_control["AllocTRES"]) == 0 else "failed")],
                             {"validate_settlement_record": settled}))
    successful = (all(check["matches_expected"] for case in cases for check in case["checks"]) and
                  submission["status"] == "passed" and live_pending["status"] == "rejected" and
                  live_cpu["status"] == "rejected" and history_gpu["status"] == "passed" and
                  settled["status"] == "passed" and settled.get("result", {}).get("actual_a100_gpu_seconds") == 0)
    report = {"schema_version": 1, "version": "v2.3.1", "status": "passed" if successful else "failed",
              "fixtures_unmodified": True, "native_commands_executed": 0, "gpu_seconds": 0,
              "historical_context": HISTORICAL_EXPECTED, "cases": cases,
              "interpretation": "Expected rejection of a CPU or pending record by scientific GPU/RUNNING gates is a passing software test; N/A means absent evidence or inapplicable historical context."}
    output = os.environ.get("MOTION_SLURM_REPLAY_OUTPUT")
    assert output, "Use the registered replay driver so the field report is retained"
    with (Path(output) / "fixture_replay.json").open("x", encoding="utf-8") as destination:
        json.dump(report, destination, ensure_ascii=False, indent=2, allow_nan=False)
        destination.write("\n")
    assert successful, "Unexpected real-fixture replay result; inspect fixture_replay.json"


@pytest.mark.parametrize("key", ["JobId", "JobState", "Restarts", "Account", "Partition", "NumNodes",
                                  "TimeLimit", "UserId", "Command", "WorkDir", "ReqTRES"])
def test_missing_submission_required_field_is_rejected(runtime, key):
    records, submit, _ = runtime
    fields = records.parse_control(fixture_text("16542_pending"))
    del fields[key]
    # Explicit synthetic mutation: the retained real fixture is never rewritten.
    synthetic = " ".join(name + "=" + value for name, value in fields.items())
    with pytest.raises((ValueError, RuntimeError, KeyError)):
        submit.validate_submission_control(synthetic, "16542", HISTORICAL_EXPECTED)


@pytest.mark.parametrize("key,value", [("JobId", "999"), ("Restarts", "1"), ("Account", "different"),
                                        ("Partition", "different"), ("NumNodes", "1-2"),
                                        ("TimeLimit", "03:00:00"), ("UserId", "different(1)"),
                                        ("Command", "/different/wrapper"), ("WorkDir", "/different"),
                                        ("ReqTRES", "gres/gpu=2")])
def test_mismatched_submission_required_field_is_rejected(runtime, key, value):
    records, submit, _ = runtime
    fields = records.parse_control(fixture_text("16542_pending"))
    fields[key] = value
    synthetic = " ".join(name + "=" + item for name, item in fields.items())
    with pytest.raises((ValueError, RuntimeError)):
        submit.validate_submission_control(synthetic, "16542", HISTORICAL_EXPECTED)


@pytest.mark.parametrize("key,value", [("ElapsedRaw", "1"), ("Start", "2026-10-31T21:38:00"),
                                        ("State", "COMPLETED"), ("User", "different"),
                                        ("ReqTRES", "gres/gpu=2")])
def test_zero_allocation_settlement_requires_consistent_evidence(runtime, key, value):
    records, submit, _ = runtime
    row = records.parse_accounting_row(fixture_text("16542_cancelled_sacct"), "16542", SETTLEMENT_COLUMNS)
    row[key] = value
    with pytest.raises((ValueError, RuntimeError)):
        submit.validate_settlement_record(row, fixture_text("16542_cancelled_control"), HISTORICAL_EXPECTED)


@pytest.mark.parametrize("key,value", [("Restarts", "1"), ("NodeList", "allocated-node"),
                                        ("RunTime", "00:00:01"), ("AllocTRES", "gres/gpu=1")])
def test_zero_allocation_settlement_rejects_contradictory_scontrol(runtime, key, value):
    records, submit, _ = runtime
    row = records.parse_accounting_row(fixture_text("16542_cancelled_sacct"), "16542", SETTLEMENT_COLUMNS)
    control = records.parse_control(fixture_text("16542_cancelled_control"))
    control[key] = value
    synthetic = " ".join(name + "=" + item for name, item in control.items())
    with pytest.raises((ValueError, RuntimeError)):
        submit.validate_settlement_record(row, synthetic, HISTORICAL_EXPECTED)


@pytest.mark.parametrize("value", ["1", "1-1"])
def test_one_node_forms(runtime, value):
    records, _, _ = runtime
    assert records.parse_node_range(value) == (1, 1)
    assert records.require_one_node(value) == 1


@pytest.mark.parametrize("value", ["", "0", "2", "1-2", "0-1", "2-1", "1-1-1", "x", "1.0"])
def test_non_single_or_malformed_node_rejected(runtime, value):
    with pytest.raises((ValueError, RuntimeError)):
        runtime[0].require_one_node(value)


@pytest.mark.parametrize("text", ["", "JobId=1 JobId=2", "JobId=1 Restarts=0 Restarts=1"])
def test_control_empty_or_duplicate_rejected(runtime, text):
    with pytest.raises((ValueError, RuntimeError)):
        runtime[0].parse_control(text)


@pytest.mark.parametrize("text", ["UNLIMITED", "Partition_Limit", "", "03:60:00", "03:00:60", "x"])
def test_invalid_or_unbounded_duration_rejected(runtime, text):
    with pytest.raises((ValueError, RuntimeError)):
        runtime[0].parse_duration(text)


@pytest.mark.parametrize("suffix", ["", "|"])
def test_headerless_empty_last_field_and_optional_extra_delimiter(runtime, suffix):
    row = runtime[0].parse_accounting_row("42|PENDING|Unknown|0|" + suffix + "\n", "42", RUNNER_COLUMNS)
    assert row == {"JobIDRaw": "42", "State": "PENDING", "Start": "Unknown", "ElapsedRaw": "0", "AllocTRES": ""}


def test_accounting_missing_exact_row_and_duplicates(runtime):
    records = runtime[0]
    assert records.parse_accounting_row("", "42", RUNNER_COLUMNS) is None
    assert records.parse_accounting_row("42.batch|RUNNING|2026-10-31T00:00:00|1|gres/gpu=1\n", "42", RUNNER_COLUMNS) is None
    with pytest.raises((ValueError, RuntimeError)):
        records.parse_accounting_row("42|RUNNING|2026-10-31T00:00:00|1|gres/gpu=1\n" * 2, "42", RUNNER_COLUMNS)
    with pytest.raises((ValueError, RuntimeError)):
        records.parse_accounting_row("42|RUNNING|missing-columns\n", "42", RUNNER_COLUMNS)
    with pytest.raises((ValueError, RuntimeError)):
        records.parse_accounting_row("JobID|State|State\n42|RUNNING|RUNNING\n", "42")


@pytest.mark.parametrize("text", ["cpu=12,gres/gpu=1,mem=24G,node=1", "gres/gpu:a100=1",
                                   "cpu=24,gres/gpu=1,gres/gpu:a100=1"])
def test_gpu_count_ignores_cpu_and_does_not_double_count(runtime, text):
    assert runtime[0].gpu_count(text) == 1
    assert runtime[0].one_gpu(text) == 1


@pytest.mark.parametrize("text", ["", "(null)", "None", "Unknown", "cpu=24,mem=16G,node=1"])
def test_zero_gpu_records_fail_scientific_gate(runtime, text):
    assert runtime[0].gpu_count(text) == 0
    with pytest.raises((ValueError, RuntimeError)):
        runtime[0].one_gpu(text)


@pytest.mark.parametrize("text", ["gres/gpu=2", "gres/gpu:v100=1", "gres/gpu=1,gres/gpu:a100=2",
                                   "gres/gpu=1,gres/gpu=1", "gres/gpu=nan", "gres/gpu=-1"])
def test_invalid_gpu_records_rejected(runtime, text):
    with pytest.raises((ValueError, RuntimeError)):
        runtime[0].one_gpu(text)


def running_control(**changes):
    """Explicit synthetic control for timing tests; never relabel a real fixture."""
    fields = {"JobId": "42", "JobState": "RUNNING", "Restarts": "0", "NumNodes": "1-1",
              "NumCPUs": "12", "CPUs/Task": "4", "AllocTRES": "cpu=12,gres/gpu=1,node=1",
              "ReqTRES": "cpu=4,gres/gpu=1,node=1", "StartTime": "2026-10-31T00:00:00",
              "RunTime": "00:00:01", "TimeLimit": "03:00:00", "Requeue": "1"}
    for key, value in changes.items():
        if value is None:
            fields.pop(key, None)
        else:
            fields[key] = value
    return " ".join(key + "=" + value for key, value in fields.items())


def accounting_line(state="RUNNING", start="2026-10-31T00:00:00", elapsed="1", tres="cpu=12,gres/gpu=1,node=1"):
    return "|".join(("42", state, start, elapsed, tres)) + "\n"


class FakeNative:
    def __init__(self, rows, control=None, command_seconds=0):
        self.rows = list(rows)
        self.control = running_control() if control is None else control
        self.command_seconds = command_seconds
        self.now = 0.0
        self.calls = []
        self.sleeps = []
        self.sacct_calls = 0
        self.probes = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        assert seconds > 0
        self.sleeps.append(seconds)
        self.now += seconds

    def command(self, argv, timeout):
        assert timeout > 0
        self.calls.append({"argv": list(argv), "timeout": timeout, "started": self.now})
        if self.command_seconds > timeout:
            self.now += timeout
            raise subprocess.TimeoutExpired(argv, timeout)
        self.now += self.command_seconds
        if argv[0] == "scontrol":
            return self.control
        assert argv[0] == "sacct", "A test must never issue another native command"
        index = min(self.sacct_calls, len(self.rows) - 1)
        self.sacct_calls += 1
        return self.rows[index]

    def invoke(self, runner, startup=120):
        return runner.native_allocation(startup, job_id="42", command_fn=self.command,
                                        clock_fn=self.clock, sleep_fn=self.sleep, record_probe=self.probes.append)


def test_startup_missing_then_running(runtime):
    native = FakeNative(["", accounting_line()])
    result = native.invoke(runtime[2])
    assert result["job_id"] == "42"
    assert native.sacct_calls == 2 and native.now == 5
    assert result["elapsed_seconds_at_probe"] >= 6
    assert result["startup_accounting_wait_seconds"] == 5
    assert [probe["status"] for probe in native.probes] == ["waiting", "passed"]


@pytest.mark.parametrize("state", ["PENDING", "CONFIGURING"])
@pytest.mark.parametrize("start,tres", [("Unknown", ""), ("None", "(null)")])
def test_startup_transient_blank_allocation_then_running(runtime, state, start, tres):
    native = FakeNative([accounting_line(state, start, "0", tres), accounting_line()])
    result = native.invoke(runtime[2])
    assert result["job_id"] == "42" and native.now == 5
    assert native.sacct_calls == 2 and native.probes[-1]["status"] == "passed"


@pytest.mark.parametrize("changes", [{"Restarts": "1"}, {"Restarts": None}, {"JobId": "43"},
                                     {"NumNodes": "1-2"}, {"AllocTRES": "gres/gpu=2"},
                                     {"AllocTRES": "gres/gpu:v100=1"}, {"JobState": "PENDING"},
                                     {"TimeLimit": "04:00:00"}, {"StartTime": "Unknown"}])
def test_invalid_control_fails_before_sacct_without_wait(runtime, changes):
    native = FakeNative([""], control=running_control(**changes))
    with pytest.raises((ValueError, RuntimeError)):
        native.invoke(runtime[2])
    assert native.sacct_calls == 0 and native.sleeps == [] and native.now == 0


@pytest.mark.parametrize("row", [accounting_line("PENDING", "None", "0", "gres/gpu=2"),
                                  accounting_line("CONFIGURING", "Unknown", "0", "gres/gpu:v100=1"),
                                  accounting_line("PENDING", "2026-10-30T00:00:00", "0", ""),
                                  accounting_line("RUNNING", "None", "0", ""),
                                  accounting_line("PENDING", "None", "bad", ""),
                                  "42|RUNNING|malformed\n"])
def test_invalid_accounting_field_is_not_retryable(runtime, row):
    native = FakeNative([row, accounting_line()])
    with pytest.raises((ValueError, RuntimeError)):
        native.invoke(runtime[2])
    assert native.sacct_calls == 1 and native.sleeps == [] and native.now == 0


@pytest.mark.parametrize("state", ["COMPLETED", "CANCELLED by 4882486", "FAILED", "TIMEOUT"])
def test_terminal_accounting_stops_immediately(runtime, state):
    native = FakeNative([accounting_line(state), accounting_line()])
    with pytest.raises((ValueError, RuntimeError)):
        native.invoke(runtime[2])
    assert native.sacct_calls == 1 and native.sleeps == [] and native.now == 0


@pytest.mark.parametrize("row", ["", accounting_line("PENDING", "Unknown", "0", "")])
def test_startup_exact_120_second_deadline(runtime, row):
    native = FakeNative([row])
    with pytest.raises((ValueError, RuntimeError)):
        native.invoke(runtime[2])
    assert native.now == 120
    assert all(call["started"] < 120 for call in native.calls)
    assert sum(native.sleeps) == 120


def test_command_latency_is_inside_startup_deadline(runtime):
    native = FakeNative([""], command_seconds=7)
    with pytest.raises((ValueError, RuntimeError, subprocess.TimeoutExpired)):
        native.invoke(runtime[2])
    assert native.now == 120
    assert all(call["timeout"] <= 120 - call["started"] for call in native.calls)
    assert sum(native.sleeps) < 120


def test_periodic_native_check_has_no_grace(runtime):
    native = FakeNative(["", accounting_line()])
    with pytest.raises((ValueError, RuntimeError)):
        native.invoke(runtime[2], startup=0)
    assert native.sacct_calls == 1 and native.sleeps == [] and native.now == 0


def test_reused_bound_provenance_json_and_bytes_only():
    """Check saved provenance/byte identity, never decode arrays or recompute B."""
    directory = REPO / "results/motion_v2/v2_3/direction_bound"
    for name, checksum in BOUND_BYTES.items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == checksum
    status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    inventory = json.loads((directory / "artifact_sha256.json").read_text(encoding="utf-8"))
    assert inventory == {name: checksum for name, checksum in BOUND_BYTES.items() if name != "artifact_sha256.json"}
    original_path = REPO / "experiments/motion_v2/protocol_step2_v2_3.json"
    original_sha = hashlib.sha256(original_path.read_bytes()).hexdigest()
    assert original_sha == "031e37b3f83d3ec5c05242a72069fc8336b32e771c3e2a8a2a3f6f0302e10875"
    original = json.loads(original_path.read_text(encoding="utf-8"))
    assert status["protocol_sha256"] == original_sha
    helper = "experiments/motion_v2/prepare_direction_bound_v2_3.py"
    assert status["source_sha256"] == {helper: original["source_sha256"][helper]}
    assert status["manifest_sha256"] == original["dataset_manifest_sha256"]
    assert status["status"] == "passed" and status["actual_a100_gpu_seconds"] == 0
    assert status["training_or_model_imported"] is False and status["test_opened_or_generated"] is False


def test_no_scientific_dependency_imported(runtime):
    assert all(name not in sys.modules for name in ("torch", "numpy", "transformers", "sklearn"))
