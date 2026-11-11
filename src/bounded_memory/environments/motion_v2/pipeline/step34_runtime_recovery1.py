"""Operational budget guard for Motion step 3/4; stdlib-only at import.

The original step-2 native Slurm reader is loaded lazily and remains unchanged.
No experiment, command, CUDA import, or filesystem write occurs on import.
"""

from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time


ALLOCATION_CAP_SECONDS = 10500
PRIOR_A100_SECONDS = 3753
TOTAL_A100_CAP_SECONDS = 14400
NATIVE_INTERVAL_SECONDS = 30
DYNAMIC_RATE_WARMUP_FRACTION = 0.1
PHASE_ESTIMATES = {"initialization": 450, "encoding_diagnostics": 900, "labels": 1000,
                   "students": 3400, "online": 2300, "finalize": 450}
PHASES = tuple(PHASE_ESTIMATES)


class BudgetStop(RuntimeError):
    """Stop without shrinking the registered work or automatically retrying."""


def _reference():
    key = "_motion_step34_frozen_native_step2"
    if key not in sys.modules:
        path = Path(__file__).with_name("run_step2.py")
        spec = importlib.util.spec_from_file_location(key, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"


def _write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(_json(value))


def _append(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(_json(value))


def _seconds(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(name + " must be a finite nonnegative number")
    return float(value)


def validate_allocation(allocation, previous=None):
    """Validate the stricter new cap plus the original native identity policy."""
    if not isinstance(allocation, dict):
        raise ValueError("Native allocation must be a dictionary")
    reference = _reference()
    records = reference.local_module("slurm_records")
    job_id = allocation.get("job_id", "")
    if not isinstance(job_id, str) or not re.fullmatch(r"[0-9]+", job_id):
        raise RuntimeError("Native job id is missing")
    fields = reference.validate_running_control(allocation.get("scontrol", ""), job_id)
    limit = allocation.get("native_time_limit_seconds")
    if type(limit) is not int or not 0 < limit <= ALLOCATION_CAP_SECONDS:
        raise RuntimeError("Native time limit must be positive and at most 10500 seconds")
    if records.parse_duration(fields["TimeLimit"]) != limit:
        raise RuntimeError("Native time-limit records disagree")
    if type(allocation.get("restarts")) is not int or allocation["restarts"] != 0:
        raise RuntimeError("Native Restarts must remain exactly zero")
    if allocation.get("start") != fields["StartTime"] or allocation.get("native_start_time") != fields["StartTime"]:
        raise RuntimeError("Native start identities disagree")
    records.one_gpu(allocation.get("alloc_tres", ""))
    _seconds(allocation.get("elapsed_seconds_at_probe"), "native elapsed")
    for field in ("UserId", "Account", "Partition", "NodeList"):
        if not fields.get(field) or fields[field] in {"None", "Unknown", "(null)"}:
            raise RuntimeError("Native identity field is unavailable: " + field)
    identity = {name: allocation[name] for name in ("job_id", "start", "native_start_time", "alloc_tres", "native_time_limit_seconds")}
    identity.update({name: fields[name] for name in ("UserId", "Account", "Partition", "NodeList")})
    if previous is not None and identity != previous:
        raise RuntimeError("Native scientific allocation identity changed")
    return identity


def normalize_gpu_uuid(value):
    """Canonicalize torch2.5.1 _CUuuid.__str__ and nvidia-smi GPU UUIDs.

    Both representations must contain the same complete 128-bit UUID. This
    does not accept arbitrary prefixes, partial IDs, MIG IDs or byte reprs.
    """
    text = str(value)
    if not re.fullmatch(r"(?:GPU-)?[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", text, re.IGNORECASE):
        raise RuntimeError("Unsupported GPU UUID representation: " + repr(text))
    return "GPU-" + (text[4:] if text[:4].upper() == "GPU-" else text).lower()


def verify_gpu(*, torch_module=None, command_fn=None, environ=None, record_probe=None):
    """Check one torch-visible A100 against its nvidia-smi name/UUID.

    Dependency injection is for offline tests only. Real torch is imported only
    when explicitly invoked inside the scientific allocation. Query by the
    torch UUID when exposed; otherwise use one CUDA_VISIBLE_DEVICES identifier.
    """
    if torch_module is None:
        import torch as torch_module
    environment = os.environ if environ is None else environ
    command_fn = _reference().command if command_fn is None else command_fn
    evidence = {"torch_version": str(getattr(torch_module, "__version__", "unknown")),
                "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES")}
    try:
        cuda = torch_module.cuda
        available, count = cuda.is_available(), cuda.device_count()
        evidence.update(cuda_available=available, device_count=count)
        if not available or count != 1:
            raise RuntimeError("Exactly one torch-visible GPU is required")
        name = cuda.get_device_name(0)
        evidence["gpu_name"] = name
        if not isinstance(name, str) or "A100" not in name.upper():
            raise RuntimeError("The visible GPU must be an A100")
        properties = cuda.get_device_properties(0)
        raw_uuid = getattr(properties, "uuid", None)
        evidence.update(torch_uuid_type=type(raw_uuid).__module__ + "." + type(raw_uuid).__qualname__,
                        torch_uuid_repr=repr(raw_uuid), torch_uuid_str=str(raw_uuid))
        gpu_uuid = normalize_gpu_uuid(raw_uuid) if raw_uuid is not None else None
        evidence["torch_gpu_uuid"] = gpu_uuid
        visible = evidence["cuda_visible_devices"]
        if visible is not None and (not visible.strip() or "," in visible):
            raise RuntimeError("CUDA_VISIBLE_DEVICES must name exactly one device")
        arguments = ["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"]
        identifier = gpu_uuid or visible
        if identifier is not None:
            arguments.append("--id=" + identifier)
        evidence["nvidia_smi_command"] = arguments
        raw = command_fn(arguments, timeout=30)
        evidence["nvidia_smi"] = raw
        rows = [line.strip() for line in raw.splitlines() if line.strip()]
        if len(rows) != 1 or len(rows[0].split(",")) != 2:
            raise RuntimeError("nvidia-smi must certify exactly one selected GPU")
        smi_name, smi_raw_uuid = (item.strip() for item in rows[0].split(","))
        smi_uuid = normalize_gpu_uuid(smi_raw_uuid)
        if smi_name != name or "A100" not in smi_name.upper():
            raise RuntimeError("Torch and nvidia-smi GPU names/types disagree")
        if gpu_uuid is not None and smi_uuid != gpu_uuid:
            raise RuntimeError("Torch and nvidia-smi GPU UUIDs disagree")
        if visible is not None and visible.upper().startswith("GPU-") and normalize_gpu_uuid(visible) != smi_uuid:
            raise RuntimeError("CUDA_VISIBLE_DEVICES and nvidia-smi UUIDs disagree")
        evidence.update(status="passed", gpu_uuid=smi_uuid)
        return evidence
    except Exception as error:
        evidence.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if record_probe is not None:
            record_probe(evidence)


class BudgetGuard:
    def __init__(self, output, *, allocation=None, native_fn=None, clock_fn=None, gpu_probe=None,
                 prior_a100_seconds=PRIOR_A100_SECONDS):
        if type(prior_a100_seconds) is not int or not PRIOR_A100_SECONDS <= prior_a100_seconds <= TOTAL_A100_CAP_SECONDS - ALLOCATION_CAP_SECONDS:
            raise ValueError("Prior allocation charge cannot fit the full registered wall limit")
        self.prior_a100_seconds = prior_a100_seconds
        self.output = Path(output)
        if self.output.exists() or self.output.is_symlink():
            raise FileExistsError("Runtime evidence directory must be new; no resume or overwrite")
        self.output.mkdir(parents=True, exist_ok=False)
        self.clock = time.monotonic if clock_fn is None else clock_fn
        self.native_fn = _reference().native_allocation if native_fn is None else native_fn
        self.gpu_probe = gpu_probe
        self.gpu_identity = None
        self.interrupted = None
        self.stopped_reason = None
        self.startup_path = self.output / "allocation_startup_checks.jsonl"
        self.native_path = self.output / "native_allocation_checks.jsonl"
        with self.startup_path.open("x", encoding="utf-8"):
            pass
        with self.native_path.open("x", encoding="utf-8"):
            pass
        if allocation is None:
            allocation = self.native_fn(startup_wait_seconds=120,
                                        record_probe=lambda event: _append(self.startup_path, event))
        self.identity = validate_allocation(allocation)
        self.allocation = dict(allocation)
        self.started = self.clock()
        self.initial_elapsed = _seconds(allocation["elapsed_seconds_at_probe"], "initial native elapsed")
        self.elapsed_offset = self.initial_elapsed
        self.last_native_probe = self.started
        self.native_probe_count = 1
        self.phase = "initialization"
        self.phase_started = self.started
        self.phase_totals_saved = {name: 0.0 for name in PHASES}
        self.phase_totals_saved[self.phase] = self.initial_elapsed
        self.phase_progress = {name: {"completed": 0, "total": None, "finished": False} for name in PHASES}
        _write_new(self.output / "allocation_start.json", allocation)
        _append(self.native_path, {"stage": "allocation_start", "status": "passed", "allocation": allocation})
        self.check()

    def elapsed(self):
        difference = self.clock() - self.started
        if difference < 0:
            raise RuntimeError("Monotonic clock moved backwards")
        return self.elapsed_offset + difference

    @property
    def limit(self):
        return min(ALLOCATION_CAP_SECONDS, TOTAL_A100_CAP_SECONDS - self.prior_a100_seconds,
                   self.allocation["native_time_limit_seconds"])

    def phase_totals(self):
        values = dict(self.phase_totals_saved)
        values[self.phase] += self.clock() - self.phase_started
        return values

    def remaining_estimate(self):
        totals = self.phase_totals()
        remaining = 0.0
        for name in PHASES:
            progress = self.phase_progress[name]
            if progress["finished"]:
                continue
            if name != self.phase:
                remaining += PHASE_ESTIMATES[name]
            elif name == "initialization":
                remaining += max(0.0, PHASE_ESTIMATES[name] - totals[name])
            elif progress["completed"] >= self.dynamic_rate_threshold(name):
                remaining += totals[name] / progress["completed"] * (progress["total"] - progress["completed"])
            else:
                # Encoder/model loading can dominate the first few units. The
                # frozen estimate remains in force until 10% of work completes;
                # actual allocation time and projection checks never pause.
                remaining += PHASE_ESTIMATES[name]
        return remaining

    def dynamic_rate_threshold(self, phase):
        total = self.phase_progress[phase]["total"]
        if phase == "initialization" or total is None:
            return None
        return max(1, math.ceil(total * DYNAMIC_RATE_WARMUP_FRACTION))

    def _stop(self, reason):
        self.stopped_reason = reason
        raise BudgetStop(reason)

    def check_native(self, force=False):
        if not force and self.clock() - self.last_native_probe < NATIVE_INTERVAL_SECONDS:
            return
        evidence = {"stage": self.phase, "probe_utc": datetime.now(timezone.utc).isoformat()}
        try:
            allocation = self.native_fn(startup_wait_seconds=0)
            evidence["allocation"] = allocation
            validate_allocation(allocation, self.identity)
            now = self.clock()
            new_offset = max(self.elapsed_offset, float(allocation["elapsed_seconds_at_probe"]) - (now - self.started))
            self.phase_totals_saved[self.phase] += new_offset - self.elapsed_offset
            self.elapsed_offset = new_offset
            if self.gpu_probe is not None:
                gpu = self.gpu_probe()
                evidence["visible_gpu"] = gpu
                identity = {key: gpu[key] for key in ("device_count", "gpu_name", "gpu_uuid", "cuda_visible_devices")}
                if self.gpu_identity is not None and identity != self.gpu_identity:
                    raise RuntimeError("Visible GPU identity changed during scientific work")
                self.gpu_identity = identity
            self.last_native_probe = self.clock()
            self.native_probe_count += 1
            evidence["status"] = "passed"
        except Exception as error:
            self.stopped_reason = "Native/GPU verification failed: " + str(error)
            evidence.update(status="failed", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            _append(self.native_path, evidence)

    def check(self, force_native=False):
        if self.stopped_reason or self.interrupted:
            self._stop(self.stopped_reason or str(self.interrupted))
        self.check_native(force=force_native)
        elapsed = self.elapsed()
        if elapsed >= self.limit:
            self._stop("Actual allocation time reached its registered cap during " + self.phase)
        if elapsed + self.remaining_estimate() > self.limit:
            self._stop("Projected allocation time exceeds its registered cap during " + self.phase)

    def set_phase(self, name, total=None):
        self.check()
        if name not in PHASES:
            raise ValueError("Unknown registered phase")
        if name != "initialization" and (type(total) is not int or total <= 0):
            raise ValueError("Scientific phase requires a positive integer work total")
        if name == self.phase:
            previous_total = self.phase_progress[name]["total"]
            if previous_total is not None and previous_total != total:
                raise ValueError("Registered phase work total cannot change")
            self.phase_progress[name]["total"] = total
            return
        if PHASES.index(name) != PHASES.index(self.phase) + 1:
            raise ValueError("Registered phases cannot be skipped, repeated, or reordered")
        prior = self.phase_progress[self.phase]
        if self.phase != "initialization" and prior["completed"] != prior["total"]:
            raise ValueError("Cannot leave a phase before its registered work completes")
        self.phase_totals_saved = self.phase_totals()
        prior["finished"] = True
        self.phase = name
        self.phase_started = self.clock()
        self.phase_progress[name]["total"] = total
        self.check()

    def progress(self, completed):
        record = self.phase_progress[self.phase]
        if type(completed) is not int or record["total"] is None or not record["completed"] <= completed <= record["total"]:
            raise ValueError("Progress must be a monotone integer within the frozen phase total")
        record["completed"] = completed
        self.check()

    def __call__(self, *, stage=None, completed=None, total=None):
        if stage is not None:
            if stage != self.phase or (total is not None and self.phase_progress[self.phase]["total"] is None):
                self.set_phase(stage, total)
            elif total is not None and total != self.phase_progress[self.phase]["total"]:
                raise ValueError("Registered work total changed")
        elif total is not None:
            raise ValueError("A work total requires its registered stage")
        if completed is not None:
            self.progress(completed)
        else:
            self.check()

    def snapshot(self):
        return {"actual_allocated_gpu_seconds_so_far": self.elapsed(),
                "prior_a100_gpu_seconds": self.prior_a100_seconds,
                "cumulative_a100_gpu_seconds_so_far": self.prior_a100_seconds + self.elapsed(),
                "estimated_remaining_seconds": self.remaining_estimate(),
                "projected_allocation_seconds": self.elapsed() + self.remaining_estimate(),
                "phase_wall_seconds": self.phase_totals(), "phase": self.phase,
                "phase_progress": {key: dict(value) for key, value in self.phase_progress.items()},
                "phase_initial_estimates": dict(PHASE_ESTIMATES),
                "dynamic_rate_warmup_fraction": DYNAMIC_RATE_WARMUP_FRACTION,
                "dynamic_rate_threshold_units": {name: self.dynamic_rate_threshold(name) for name in PHASES},
                "warmup_remaining_rule": "Keep the full registered phase estimate until ceil(total*0.1) units complete; actual/projected cap checks remain active",
                "initial_estimate_seconds": sum(PHASE_ESTIMATES.values()),
                "native_allocation_checks": self.native_probe_count, "native_restarts_required": 0,
                "native_check_interval_seconds": NATIVE_INTERVAL_SECONDS,
                "startup_accounting_grace_seconds": 120, "runtime_accounting_grace_seconds": 0,
                "native_time_limit_seconds": self.allocation["native_time_limit_seconds"],
                "effective_allocation_cap_seconds": self.limit, "total_cap_seconds": TOTAL_A100_CAP_SECONDS,
                "gpu_count": 1, "stopped_reason": self.stopped_reason,
                "requeue": self.allocation.get("requeue"), "requeue_is_report_only": True,
                "automatic_retry_or_resume": False}
