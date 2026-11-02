"""Pure fake-clock/native/GPU tests; no subprocess, Slurm, CUDA, or study data."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SPEC = importlib.util.spec_from_file_location("motion_step34_runtime_under_test", Path(__file__).with_name("step34_runtime_recovery1.py"))
RUNTIME = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNTIME)
GPU_NAME = "NVIDIA A100-SXM4-40GB"
GPU_UUID = "GPU-12345678-1234-1234-1234-123456789abc"


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def duration(seconds):
    return f"{seconds // 3600:02}:{seconds // 60 % 60:02}:{seconds % 60:02}"


def allocation(elapsed=10, **changes):
    fields = {"JobId": "12345", "JobState": "RUNNING", "NumNodes": "1", "Restarts": "0",
              "UserId": "xy3257(123)", "Account": "csci_ga_2271-2026fa", "Partition": "synthetic-a100",
              "NodeList": "synthetic-node", "StartTime": "2026-11-01T01:00:00",
              "RunTime": duration(int(elapsed)), "TimeLimit": "02:55:00", "Requeue": "1",
              "AllocTRES": "cpu=12,mem=16G,node=1,gres/gpu=1,gres/gpu:a100=1"}
    fields.update(changes)
    return {"job_id": fields["JobId"], "start": fields["StartTime"], "native_start_time": fields["StartTime"],
            "elapsed_seconds_at_probe": elapsed, "alloc_tres": fields["AllocTRES"],
            "native_time_limit_seconds": 10500, "restarts": int(fields["Restarts"]), "requeue": fields["Requeue"],
            "scontrol": " ".join(key + "=" + value for key, value in fields.items()), "sacct": "synthetic-only"}


class Native:
    def __init__(self, clock):
        self.clock, self.calls = clock, []
        self.changes = {}
        self.error = None
        self.delay = 0.0

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        self.clock.advance(self.delay)
        if self.error is not None:
            raise self.error
        result = allocation(10 + int(self.clock()), **self.changes)
        if kwargs.get("record_probe") is not None:
            kwargs["record_probe"]({"status": "passed", "synthetic": True})
        return result


def guard(tmp_path, *, supplied=False, gpu_probe=None):
    clock = Clock()
    native = Native(clock)
    instance = RUNTIME.BudgetGuard(tmp_path / "runtime", clock_fn=clock, native_fn=native,
                                   allocation=allocation() if supplied else None, gpu_probe=gpu_probe)
    return instance, clock, native


def fake_torch(*, count=1, available=True, name=GPU_NAME, uuid=GPU_UUID):
    return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: available, device_count=lambda: count,
        get_device_name=lambda index: name, get_device_properties=lambda index: SimpleNamespace(uuid=uuid)))


def test_initial_native_time_and_registered_budget(tmp_path):
    budget, clock, native = guard(tmp_path)
    snapshot = budget.snapshot()
    assert snapshot["initial_estimate_seconds"] == 8500
    assert snapshot["prior_a100_gpu_seconds"] == 3753
    assert snapshot["actual_allocated_gpu_seconds_so_far"] == 10
    assert snapshot["phase_wall_seconds"]["initialization"] == 10
    assert snapshot["cumulative_a100_gpu_seconds_so_far"] == 3763
    assert snapshot["effective_allocation_cap_seconds"] == 10500
    assert snapshot["projected_allocation_seconds"] == 8500
    assert native.calls[0]["startup_wait_seconds"] == 120
    assert len(native.calls) == 1
    assert json.loads((tmp_path / "runtime/allocation_startup_checks.jsonl").read_text())["synthetic"] is True


def test_allowed_infrastructure_charge_is_in_live_guard(tmp_path):
    clock=Clock()
    budget=RUNTIME.BudgetGuard(tmp_path/'runtime',allocation=allocation(),clock_fn=clock,
                              native_fn=Native(clock),prior_a100_seconds=3900)
    assert budget.snapshot()['cumulative_a100_gpu_seconds_so_far']==3910
    assert budget.limit==10500


@pytest.mark.parametrize('prior',[3752,3901,True,3753.0])
def test_invalid_recovery_charge_rejected_before_output(tmp_path,prior):
    with pytest.raises(ValueError):
        RUNTIME.BudgetGuard(tmp_path/'runtime',prior_a100_seconds=prior)
    assert not (tmp_path/'runtime').exists()


def test_runtime_native_checks_have_no_startup_grace(tmp_path):
    budget, clock, native = guard(tmp_path)
    clock.advance(29)
    budget()
    assert len(native.calls) == 1
    clock.advance(1)
    budget()
    assert len(native.calls) == 2
    assert native.calls[-1] == {"startup_wait_seconds": 0}
    assert budget.snapshot()["native_allocation_checks"] == 2


def test_supplied_allocation_does_not_repeat_startup(tmp_path):
    budget, _, native = guard(tmp_path, supplied=True)
    assert not native.calls
    budget.check(force_native=True)
    assert native.calls == [{"startup_wait_seconds": 0}]


def test_stage_progress_rate_and_clock_time_included(tmp_path):
    budget, clock, _ = guard(tmp_path)
    budget.set_phase("encoding_diagnostics", 100)
    clock.advance(4)
    budget.progress(25)
    assert budget.remaining_estimate() == pytest.approx(12 + 1000 + 3400 + 2300 + 450)
    assert budget.phase_totals()["encoding_diagnostics"] == 4
    assert budget.elapsed() == 14


def test_projected_overrun_stops_without_reducing_work(tmp_path):
    budget, clock, native = guard(tmp_path)
    budget.set_phase("encoding_diagnostics", 1000)
    clock.advance(800)
    with pytest.raises(RUNTIME.BudgetStop, match="Projected"):
        budget.progress(100)
    assert budget.phase_progress["encoding_diagnostics"]["total"] == 1000
    calls = len(native.calls)
    with pytest.raises(RUNTIME.BudgetStop):
        budget()
    assert len(native.calls) == calls


def test_fixed_ten_percent_warmup_does_not_extrapolate_model_load(tmp_path):
    budget, clock, _ = guard(tmp_path)
    budget.set_phase("encoding_diagnostics", 1000)
    clock.advance(40)
    budget.progress(1)
    assert budget.remaining_estimate() == 900 + 1000 + 3400 + 2300 + 450
    budget.progress(99)
    assert budget.remaining_estimate() == 8050
    budget.progress(100)
    assert budget.remaining_estimate() == pytest.approx(360 + 7150)
    snapshot = budget.snapshot()
    assert snapshot["dynamic_rate_warmup_fraction"] == .1
    assert snapshot["dynamic_rate_threshold_units"]["encoding_diagnostics"] == 100


def test_warmup_does_not_suspend_projected_or_actual_limits(tmp_path):
    budget, clock, _ = guard(tmp_path)
    budget.set_phase("encoding_diagnostics", 1000)
    clock.advance(2500)
    with pytest.raises(RUNTIME.BudgetStop, match="Projected"):
        budget.progress(1)


def test_actual_limit_stops_even_if_other_work_is_finished(tmp_path):
    budget, clock, _ = guard(tmp_path)
    for phase in RUNTIME.PHASES[1:]:
        budget.set_phase(phase, 1)
        budget.progress(1)
    assert budget.remaining_estimate() == 0
    clock.advance(10490)
    with pytest.raises(RUNTIME.BudgetStop, match="Actual"):
        budget()


def test_runtime_command_latency_is_counted(tmp_path):
    budget, clock, native = guard(tmp_path)
    clock.advance(30)
    native.delay = 4
    budget()
    assert budget.elapsed() == 44
    assert sum(budget.phase_totals().values()) == 44


def test_higher_native_elapsed_adds_auditable_floor(tmp_path):
    budget, clock, native = guard(tmp_path)
    clock.advance(30)
    budget.native_fn = lambda **kwargs: allocation(50)
    budget()
    assert budget.elapsed() == 50
    assert sum(budget.phase_totals().values()) == 50


@pytest.mark.parametrize("changes", [{"Restarts": "1"}, {"NumNodes": "2"}, {"JobState": "PENDING"},
                                    {"AllocTRES": "cpu=12,node=1,gres/gpu=2,gres/gpu:a100=2"},
                                    {"AllocTRES": "cpu=12,node=1,gres/gpu=1,gres/gpu:v100=1"}])
def test_bad_native_allocation_stops_immediately(tmp_path, changes):
    budget, clock, native = guard(tmp_path)
    native.changes = changes
    clock.advance(30)
    with pytest.raises((ValueError, RuntimeError)):
        budget()
    assert len(native.calls) == 2
    assert json.loads((tmp_path / "runtime/native_allocation_checks.jsonl").read_text().splitlines()[-1])["status"] == "failed"


@pytest.mark.parametrize("field,value", [("JobId", "54321"), ("StartTime", "2026-11-01T01:01:00"),
                                       ("UserId", "other(999)"), ("Account", "other-account"),
                                       ("Partition", "other-partition"), ("NodeList", "other-node")])
def test_native_identity_cannot_change(tmp_path, field, value):
    budget, _, native = guard(tmp_path)
    native.changes[field] = value
    with pytest.raises(RuntimeError, match="identity changed"):
        budget.check(force_native=True)


def test_requeue_is_recorded_without_becoming_a_stop_rule(tmp_path):
    budget, _, native = guard(tmp_path)
    native.changes["Requeue"] = "0"
    budget.check(force_native=True)
    assert budget.snapshot()["requeue_is_report_only"] is True


def test_missing_runtime_accounting_never_retries(tmp_path):
    budget, _, native = guard(tmp_path)
    native.error = RuntimeError("Native accounting is not ready; runtime monitoring has no grace")
    with pytest.raises(RuntimeError, match="no grace"):
        budget.check(force_native=True)
    assert len(native.calls) == 2
    with pytest.raises(RUNTIME.BudgetStop):
        budget()
    assert len(native.calls) == 2


def test_new_limit_stricter_than_unchanged_step2_limit():
    record = allocation(TimeLimit="03:00:00")
    record["native_time_limit_seconds"] = 10800
    with pytest.raises(RuntimeError, match="10500"):
        RUNTIME.validate_allocation(record)


@pytest.mark.parametrize("field,bad", [("native_time_limit_seconds", True), ("restarts", False),
                                     ("elapsed_seconds_at_probe", float("nan")),
                                     ("elapsed_seconds_at_probe", -1)])
def test_malformed_native_values_rejected(field, bad):
    record = allocation()
    record[field] = bad
    with pytest.raises((ValueError, RuntimeError)):
        RUNTIME.validate_allocation(record)


def test_total_work_and_phase_order_cannot_change(tmp_path):
    budget, _, _ = guard(tmp_path)
    with pytest.raises(ValueError, match="skipped"):
        budget.set_phase("students", 54000)
    budget.set_phase("encoding_diagnostics", 100)
    with pytest.raises(ValueError, match="before"):
        budget.set_phase("labels", 20)
    with pytest.raises(ValueError, match="cannot change"):
        budget.set_phase("encoding_diagnostics", 50)


@pytest.mark.parametrize("bad", [True, 1.0, "1", -1, 101])
def test_invalid_progress_rejected(tmp_path, bad):
    budget, _, _ = guard(tmp_path)
    budget.set_phase("encoding_diagnostics", 100)
    with pytest.raises(ValueError):
        budget.progress(bad)


def test_student_callback_accepts_fixed_work_and_monotone_updates(tmp_path):
    budget, _, _ = guard(tmp_path)
    for phase in ("encoding_diagnostics", "labels"):
        budget.set_phase(phase, 1)
        budget.progress(1)
    budget(stage="students", completed=0, total=54000)
    budget(stage="students", completed=100, total=54000)
    assert budget.phase_progress["students"]["completed"] == 100
    assert budget.snapshot()["dynamic_rate_threshold_units"]["students"] == 5400
    assert budget.remaining_estimate() == 3400 + 2300 + 450
    with pytest.raises(ValueError):
        budget(stage="students", completed=99, total=54000)
    with pytest.raises(ValueError):
        budget(stage="students", completed=100, total=27000)


def test_interruption_remains_stopped(tmp_path):
    budget, _, _ = guard(tmp_path)
    budget.interrupted = "synthetic signal"
    with pytest.raises(RUNTIME.BudgetStop, match="signal"):
        budget()
    budget.interrupted = None
    with pytest.raises(RUNTIME.BudgetStop, match="signal"):
        budget()


def test_runtime_output_is_exclusive(tmp_path):
    budget, _, _ = guard(tmp_path)
    original = (tmp_path / "runtime/allocation_start.json").read_bytes()
    with pytest.raises(FileExistsError):
        RUNTIME.BudgetGuard(tmp_path / "runtime", allocation=allocation(), native_fn=lambda **kwargs: allocation())
    assert (tmp_path / "runtime/allocation_start.json").read_bytes() == original


def test_single_a100_gpu_verified_without_commands_or_cuda():
    calls = []
    def command(arguments, timeout):
        calls.append((arguments, timeout))
        return GPU_NAME + ", " + GPU_UUID + "\n"
    result = RUNTIME.verify_gpu(torch_module=fake_torch(), command_fn=command, environ={"CUDA_VISIBLE_DEVICES": "0"})
    assert result["gpu_uuid"] == GPU_UUID
    assert calls == [(["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader", "--id=" + GPU_UUID], 30)]


@pytest.mark.parametrize("fake", [fake_torch(count=0), fake_torch(count=2), fake_torch(available=False),
                                fake_torch(name="NVIDIA V100")])
def test_gpu_torch_rejection_precedes_smi(fake):
    def forbidden(*args, **kwargs):
        raise AssertionError("No nvidia-smi call is permitted")
    with pytest.raises(RuntimeError):
        RUNTIME.verify_gpu(torch_module=fake, command_fn=forbidden, environ={})


@pytest.mark.parametrize("text", ["", "NVIDIA V100, " + GPU_UUID,
                                 GPU_NAME + ", GPU-aaaaaaaa-1234-1234-1234-123456789abc",
                                 GPU_NAME + ", " + GPU_UUID + "\n" + GPU_NAME + ", " + GPU_UUID])
def test_nvidia_smi_mismatch_rejected(text):
    with pytest.raises(RuntimeError):
        RUNTIME.verify_gpu(torch_module=fake_torch(), command_fn=lambda *args, **kwargs: text, environ={})


def test_uuid_missing_uses_one_visible_identifier():
    calls = []
    def command(arguments, timeout):
        calls.append(arguments)
        return GPU_NAME + ", " + GPU_UUID
    result = RUNTIME.verify_gpu(torch_module=fake_torch(uuid=None), command_fn=command,
                                 environ={"CUDA_VISIBLE_DEVICES": "0"})
    assert calls[0][-1] == "--id=0"
    assert result["torch_gpu_uuid"] is None


def test_periodic_gpu_identity_change_stops(tmp_path):
    current = {"device_count": 1, "gpu_name": GPU_NAME, "gpu_uuid": GPU_UUID, "cuda_visible_devices": "0"}
    budget, _, _ = guard(tmp_path, gpu_probe=lambda: copy.deepcopy(current))
    budget.check(force_native=True)
    current["gpu_uuid"] = "GPU-aaaaaaaa-1234-1234-1234-123456789abc"
    with pytest.raises(RuntimeError, match="Visible GPU identity"):
        budget.check(force_native=True)
