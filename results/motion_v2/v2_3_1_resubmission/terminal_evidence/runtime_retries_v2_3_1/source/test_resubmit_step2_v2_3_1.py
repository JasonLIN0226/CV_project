"""Pure runtime admission tests: no Slurm, models, arrays, or science execution.

Run only after the new software protocol records these source hashes.  These
tests exercise the actual retry helper, not a second implementation of its gate.
"""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCIENTIFIC_HEAD = "6dc5ed253dfa305b26be8f19d40edcd6a5878ea7"
FROZEN_RUNTIME_FILES = {
    "run_step2_v2_3_1.sbatch": "7543fd9282b89639daf8054019d01977dda092a1f78c7c052d3e687fd9378b84",
    "run_step2.py": "b6004fe74e87c5d6f8a9392dba08d6a6b76c8365664b7bfe3295c23872ba63a7",
    "submit_step2_v2_3_1.py": "970533810a79155525474749e4ae41f555a60d0c1abd2bd0554ac958b283075e",
    "slurm_records.py": "720ce2094378c148abbb30b0b4cf8493748bfdf2ad177c58a3269fc49aeddddf",
    "config_step2_v2_3_1.json": "5156c095d4e805a60d1f7b29c1491e8fed51b8d612b0173b80d4349d803a5bcd",
    "protocol_step2_v2_3_1.json": "b88c85ca78a91b722c48bdd8465ce88b324ac281447c4608d1447f1a6b508c60",
    "protocol_step2_deployment_v2_3_1.json": "1c6151d7a6d47833e7baa80389820bcc661ab7dbeeb0a6bfcafb5d181ccb693a",
    "test_slurm_records_v2_3_1.py": "22e9036a8eda581cb5e32002c5986eca4e68294d2c1c9f5b3d47c50dafda966f",
    "replay_slurm_v2_3_1.py": "f9b6ebc04dcbe68759fdf1956122834dfd0b76cb4907f0c0919438f1c236e5ed",
}
FOUR_FACTS = (
    "no_batch_step", "stdout_stderr_absent", "scientific_marker_absent",
    "scientific_output_absent",
)
ACCOUNTING_COLUMNS = "JobIDRaw,State,Start,ElapsedRaw,AllocTRES"
TOP_LEVEL_ROW = (
    "16550|CANCELLED by 0|2026-10-31T22:07:12|769|"
    "billing=12,cpu=12,gres/gpu=1,mem=16G,node=1")
TERMINAL_FIXTURES = {
    "accounting.json": "57d3cd0ebbe345c7cd3e1779d691af87b5b9ebf7d50afecff4d5307b081fe01a",
    "final_scontrol.stdout.txt": "38ba2a0bfdcb44cd3b94aba51dd1b79912520a82c1ab0acf7fd1fcc3fd89739d",
}


@pytest.fixture
def runtime(monkeypatch):
    def no_external_commands(*args, **kwargs):
        raise AssertionError("Pure admission must not invoke external commands")

    monkeypatch.setattr(subprocess, "run", no_external_commands)
    monkeypatch.setattr(subprocess, "Popen", no_external_commands)
    spec = importlib.util.spec_from_file_location(
        "offline_retry_v2_3_1", HERE / "resubmit_step2_v2_3_1.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def records(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "offline_retry_records_v2_3_1", HERE / "slurm_records.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def terminal_evidence():
    directory = REPO / "results/motion_v2/v2_3_1/deployment3/remote_evidence/settlement"
    content = {}
    for name, expected in TERMINAL_FIXTURES.items():
        raw = (directory / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected
        content[name] = raw.decode("utf-8")
    receipt = json.loads(content["accounting.json"])
    assert receipt["scontrol"]["stdout"] == content["final_scontrol.stdout.txt"]
    return receipt["allocation_row"], content["final_scontrol.stdout.txt"]


def history_record(attempt=1, job_id="16550", seconds=769):
    return {
        "attempt": attempt,
        "job_id": job_id,
        "terminal": True,
        "restarts": 0,
        "actual_a100_gpu_seconds": seconds,
        **{name: True for name in FOUR_FACTS},
    }


def admit(runtime, attempt=2, history=None, **overrides):
    values = {
        "scientific_head": SCIENTIFIC_HEAD,
        "frozen_source_verified": True,
        "frozen_preparation_verified": True,
    }
    values.update(overrides)
    return runtime.validate_retry_admission(
        attempt, [history_record()] if history is None else history, **values)


@pytest.mark.parametrize("name", tuple(FROZEN_RUNTIME_FILES))
def test_existing_runtime_science_and_replay_bytes_unchanged(name):
    """Byte checks do not execute the old replay or any scientific function."""
    assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == FROZEN_RUNTIME_FILES[name]


def test_second_attempt_has_complete_three_hour_budget(runtime):
    result = admit(runtime)
    assert result == {
        "eligible": True,
        "attempt": 2,
        "cumulative_a100_gpu_seconds": 769,
        "maximum_additional_a100_gpu_seconds": 10800,
        "projected_cumulative_a100_gpu_seconds": 11569,
        "user_total_cap_seconds": 14400,
        "maximum_total_attempts": 3,
    }


def test_third_attempt_exact_four_hour_cap_is_admissible(runtime):
    history = [history_record(), history_record(2, "16600", 2831)]
    result = admit(runtime, 3, history)
    assert result["cumulative_a100_gpu_seconds"] == 3600
    assert result["projected_cumulative_a100_gpu_seconds"] == 14400
    assert result["maximum_additional_a100_gpu_seconds"] == 10800


def test_third_attempt_one_second_over_cap_is_rejected(runtime):
    history = [history_record(), history_record(2, "16600", 2832)]
    with pytest.raises(ValueError):
        admit(runtime, 3, history)


def test_zero_allocation_second_job_still_consumes_an_attempt(runtime):
    history = [history_record(), history_record(2, "16600", 0)]
    assert admit(runtime, 3, history)["cumulative_a100_gpu_seconds"] == 769
    history.append(history_record(3, "16601", 0))
    with pytest.raises(ValueError):
        admit(runtime, 4, history)


def test_admission_does_not_mutate_evidence(runtime):
    history = [history_record(), history_record(2, "16600", 1200)]
    before = deepcopy(history)
    admit(runtime, 3, history)
    assert history == before


@pytest.mark.parametrize("attempt", [0, 1, 4, -1, True, False, 2.0, "2", None])
def test_attempt_number_is_strict_and_includes_first_job(runtime, attempt):
    with pytest.raises(ValueError):
        admit(runtime, attempt)


@pytest.mark.parametrize("attempt,history", [
    (2, []),
    (3, [history_record()]),
    (2, [history_record(), history_record(2, "16600", 0)]),
    (3, [history_record(), history_record(3, "16600", 0)]),
    (3, [history_record(2), history_record(1, "16600", 0)]),
    (3, [history_record(), history_record(1, "16600", 0)]),
])
def test_history_cannot_skip_omit_or_repeat_attempts(runtime, attempt, history):
    with pytest.raises(ValueError):
        admit(runtime, attempt, deepcopy(history))


@pytest.mark.parametrize("job_id", ["16551", "16550.batch", "", None])
def test_first_job_identity_is_fixed(runtime, job_id):
    record = history_record(job_id=job_id)
    with pytest.raises(ValueError):
        admit(runtime, history=[record])


@pytest.mark.parametrize("seconds", [0, 768, 770, -1, True, 769.0, "769", None])
def test_first_job_actual_allocation_is_fixed_769_seconds(runtime, seconds):
    with pytest.raises(ValueError):
        admit(runtime, history=[history_record(seconds=seconds)])


def test_job_id_cannot_be_reused_for_two_attempts(runtime):
    history = [history_record(), history_record(2, "16550", 0)]
    with pytest.raises(ValueError):
        admit(runtime, 3, history)


@pytest.mark.parametrize("seconds", [-1, True, False, 1.5, "0", None])
def test_later_gpu_seconds_require_a_nonnegative_integer(runtime, seconds):
    history = [history_record(), history_record(2, "16600", seconds)]
    with pytest.raises(ValueError):
        admit(runtime, 3, history)


@pytest.mark.parametrize("field", FOUR_FACTS)
@pytest.mark.parametrize("value", [False, None, "true", "proven", 1])
def test_every_absence_fact_requires_proven_true(runtime, field, value):
    history = [history_record()]
    history[0][field] = value
    with pytest.raises(ValueError):
        admit(runtime, history=history)


@pytest.mark.parametrize("field", FOUR_FACTS)
def test_missing_absence_fact_cannot_authorize_retry(runtime, field):
    history = [history_record()]
    del history[0][field]
    with pytest.raises(ValueError):
        admit(runtime, history=history)


@pytest.mark.parametrize("field", FOUR_FACTS)
def test_latest_attempt_also_requires_all_four_facts(runtime, field):
    history = [history_record(), history_record(2, "16600", 0)]
    history[1][field] = False
    with pytest.raises(ValueError):
        admit(runtime, 3, history)


@pytest.mark.parametrize("terminal", [False, None, "CANCELLED", "True", 1])
def test_unknown_or_nonterminal_submission_never_authorizes_retry(runtime, terminal):
    history = [history_record()]
    history[0]["terminal"] = terminal
    with pytest.raises(ValueError):
        admit(runtime, history=history)


@pytest.mark.parametrize("restarts", [1, -1, True, False, "0", None])
def test_restart_count_must_be_integer_zero(runtime, restarts):
    history = [history_record()]
    history[0]["restarts"] = restarts
    with pytest.raises(ValueError):
        admit(runtime, history=history)


@pytest.mark.parametrize("field", ["attempt", "job_id", "terminal", "restarts", "actual_a100_gpu_seconds"])
def test_missing_history_identity_or_accounting_rejects(runtime, field):
    history = [history_record()]
    del history[0][field]
    with pytest.raises(ValueError):
        admit(runtime, history=history)


@pytest.mark.parametrize("head", ["6dc5ed2", "HEAD", "", None, "0" * 40])
def test_exact_frozen_remote_head_is_required(runtime, head):
    with pytest.raises(ValueError):
        admit(runtime, scientific_head=head)


@pytest.mark.parametrize("field", ["frozen_source_verified", "frozen_preparation_verified"])
@pytest.mark.parametrize("value", [False, None, "true", 1])
def test_source_and_preparation_receipts_must_be_verified(runtime, field, value):
    with pytest.raises(ValueError):
        admit(runtime, **{field: value})


def test_complete_accounting_top_row_without_steps(runtime, records):
    result = runtime.inspect_duplicate_accounting(
        TOP_LEVEL_ROW + "\n", "16550", ACCOUNTING_COLUMNS, records)
    assert result["no_batch_step"] is True
    assert result["batch_step_ids"] == []
    assert result["allocation_row"]["JobIDRaw"] == "16550"
    assert result["allocation_row"]["ElapsedRaw"] == "769"


def test_batch_step_is_start_evidence_not_an_extra_gpu_bill(runtime, records):
    raw = TOP_LEVEL_ROW + "\n" + TOP_LEVEL_ROW.replace("16550|", "16550.batch|", 1)
    result = runtime.inspect_duplicate_accounting(raw, "16550", ACCOUNTING_COLUMNS, records)
    assert result["no_batch_step"] is False
    assert result["batch_step_ids"] == ["16550.batch"]
    assert result["allocation_row"]["JobIDRaw"] == "16550"
    assert result["allocation_row"]["ElapsedRaw"] == "769"


def test_nested_batch_step_also_blocks_retry(runtime, records):
    raw = TOP_LEVEL_ROW + "\n" + TOP_LEVEL_ROW.replace("16550|", "16550.batch.0|", 1)
    result = runtime.inspect_duplicate_accounting(raw, "16550", ACCOUNTING_COLUMNS, records)
    assert result["no_batch_step"] is False
    assert result["batch_step_ids"] == ["16550.batch.0"]


def test_extern_step_is_preserved_without_being_called_batch(runtime, records):
    raw = TOP_LEVEL_ROW + "\n" + TOP_LEVEL_ROW.replace("16550|", "16550.extern|", 1)
    result = runtime.inspect_duplicate_accounting(raw, "16550", ACCOUNTING_COLUMNS, records)
    assert result["no_batch_step"] is True
    assert result["batch_step_ids"] == []
    assert result["allocation_row"]["ElapsedRaw"] == "769"


@pytest.mark.parametrize("raw", [
    "",
    TOP_LEVEL_ROW.replace("16550|", "16550.batch|", 1),
    TOP_LEVEL_ROW + "\n" + TOP_LEVEL_ROW,
    TOP_LEVEL_ROW + "\n" + TOP_LEVEL_ROW.replace("16550|", "16551|", 1),
    TOP_LEVEL_ROW + "\n" + TOP_LEVEL_ROW.replace("16550|", "16551.batch|", 1),
    "16550|CANCELLED by 0|2026-10-31T22:07:12|769",
    TOP_LEVEL_ROW + "|unexpected-field",
])
def test_ambiguous_or_incomplete_native_records_never_prove_no_batch(runtime, records, raw):
    with pytest.raises(ValueError):
        runtime.inspect_duplicate_accounting(raw, "16550", ACCOUNTING_COLUMNS, records)


def test_accounting_keeps_empty_final_allocation_field(runtime, records):
    raw = "16550|CANCELLED by 0|None|0|"
    result = runtime.inspect_duplicate_accounting(raw, "16550", ACCOUNTING_COLUMNS, records)
    assert result["allocation_row"]["AllocTRES"] == ""
    assert result["allocation_row"]["ElapsedRaw"] == "0"
    assert result["no_batch_step"] is True


def purged_lookup(**overrides):
    result = {"returncode": 1, "timed_out": False, "stdout": "",
              "stderr": "slurm_load_jobs error: Invalid job id specified\n"}
    result.update(overrides)
    return result


def retained(runtime, records, terminal_evidence, **overrides):
    row, text = terminal_evidence
    values = {
        "lookup": purged_lookup(),
        "fresh_row": deepcopy(row),
        "saved_row": deepcopy(row),
        "saved_control_text": text,
        "expected_control_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "parser": records,
    }
    values.update(overrides)
    return runtime.retained_control_after_purge(**values)


@pytest.mark.parametrize("stderr", [
    "Invalid job id specified",
    "slurm_load_jobs error: Invalid job id specified\n",
    "scontrol: error: Invalid job id specified\n",
])
def test_explicit_purge_can_reuse_exact_hash_locked_terminal_control(
        runtime, records, terminal_evidence, stderr):
    row, text = terminal_evidence
    result = retained(runtime, records, terminal_evidence, lookup=purged_lookup(stderr=stderr))
    assert result == {
        "control_text": text,
        "control_sha256": TERMINAL_FIXTURES["final_scontrol.stdout.txt"],
        "evidence_kind": "preserved_terminal_scontrol_after_explicit_purge",
    }
    assert row["ElapsedRaw"] == "769"
    assert row["State"] == "CANCELLED by 0"


@pytest.mark.parametrize("stderr", [
    "", "Permission denied", "slurm_load_jobs error: Unable to contact slurm controller",
    "Socket timed out", "slurm_load_jobs error: Invalid job id specified; permission denied",
    "Invalid job id specified\nUnexpected additional error",
])
def test_other_scontrol_errors_never_trigger_terminal_fallback(
        runtime, records, terminal_evidence, stderr):
    with pytest.raises(ValueError):
        retained(runtime, records, terminal_evidence, lookup=purged_lookup(stderr=stderr))


@pytest.mark.parametrize("returncode", [0, None, True, False, "1", 1.0])
def test_purge_requires_an_actual_failed_native_command(
        runtime, records, terminal_evidence, returncode):
    with pytest.raises(ValueError):
        retained(runtime, records, terminal_evidence, lookup=purged_lookup(returncode=returncode))


def test_timed_out_command_cannot_be_interpreted_as_purged(runtime, records, terminal_evidence):
    with pytest.raises(ValueError):
        retained(runtime, records, terminal_evidence, lookup=purged_lookup(timed_out=True))


def test_partial_scontrol_output_cannot_be_ignored(runtime, records, terminal_evidence):
    with pytest.raises(ValueError):
        retained(runtime, records, terminal_evidence, lookup=purged_lookup(stdout="JobId=16550 "))


@pytest.mark.parametrize("field", [
    "JobIDRaw", "JobName", "User", "Account", "Partition", "State", "ExitCode",
    "Start", "End", "ElapsedRaw", "AllocTRES", "ReqTRES",
])
def test_every_fresh_accounting_field_must_equal_preserved_accounting(
        runtime, records, terminal_evidence, field):
    fresh = deepcopy(terminal_evidence[0])
    fresh[field] = "changed native field"
    with pytest.raises(ValueError):
        retained(runtime, records, terminal_evidence, fresh_row=fresh)


def test_missing_fresh_accounting_field_rejects_fallback(runtime, records, terminal_evidence):
    fresh = deepcopy(terminal_evidence[0])
    del fresh["End"]
    with pytest.raises(ValueError):
        retained(runtime, records, terminal_evidence, fresh_row=fresh)


@pytest.mark.parametrize("expected_hash", [None, "", "0" * 64, "f" * 63, "F" * 64])
def test_preserved_control_requires_the_exact_registered_sha(
        runtime, records, terminal_evidence, expected_hash):
    with pytest.raises(ValueError):
        retained(runtime, records, terminal_evidence, expected_control_sha256=expected_hash)


@pytest.mark.parametrize("before,after", [
    ("JobId=16550 ", "JobId=16551 "),
    ("Restarts=0 ", "Restarts=1 "),
    ("Restarts=0 ", ""),
    ("JobState=CANCELLED ", "JobState=FAILED "),
])
def test_hash_valid_control_still_needs_same_job_state_and_zero_restarts(
        runtime, records, terminal_evidence, before, after):
    assert before in terminal_evidence[1]
    modified = terminal_evidence[1].replace(before, after, 1)
    with pytest.raises(ValueError):
        retained(runtime, records, terminal_evidence, saved_control_text=modified,
                 expected_control_sha256=hashlib.sha256(modified.encode("utf-8")).hexdigest())


def test_matching_nonterminal_control_is_not_terminal_proof(runtime, records, terminal_evidence):
    row = deepcopy(terminal_evidence[0])
    row["State"] = "RUNNING"
    modified = terminal_evidence[1].replace("JobState=CANCELLED ", "JobState=RUNNING ", 1)
    with pytest.raises(ValueError):
        retained(runtime, records, terminal_evidence, fresh_row=deepcopy(row), saved_row=deepcopy(row),
                 saved_control_text=modified,
                 expected_control_sha256=hashlib.sha256(modified.encode("utf-8")).hexdigest())


def test_retained_fallback_does_not_mutate_native_evidence(runtime, records, terminal_evidence):
    fresh, saved = deepcopy(terminal_evidence[0]), deepcopy(terminal_evidence[0])
    before = deepcopy(fresh)
    retained(runtime, records, terminal_evidence, fresh_row=fresh, saved_row=saved)
    assert fresh == saved == before
