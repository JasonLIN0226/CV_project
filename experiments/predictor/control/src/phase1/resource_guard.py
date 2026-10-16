"""Fail-closed resource accounting for the authorized full Phase1 sequence.

Preparation only: no Slurm submission, data generation, or model execution.
All rate measurements must later come from the native course allocation.
"""
import math
from datetime import datetime, timezone

CAP = 28800
PRIOR_GPU_SECONDS = {7477: 100, 7480: 126, 7485: 6021, 7524: 164, 7535: 3386}
TASKS = ("gate", "motion")


def positive(value, name, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Missing numeric native measurement: " + name)
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError("Invalid native measurement: " + name)
    return float(value)


def reconcile_prior(rows):
    """Require all three actual allocation records; batch rows are excluded.

    The caller must read these rows from sacct on the current course node.
    An empty/missing record cannot be interpreted as zero GPU consumption.
    CPU jobs are not accepted into this GPU-only ledger.
    """
    observed = {}
    for row in rows:
        job = row["job_id"]
        if job not in PRIOR_GPU_SECONDS or job in observed:
            raise ValueError("Unexpected, duplicate, or step-level allocation record")
        if row["account"] != "csci_ga_2271-2026fa" or row["allocated_gpus"] != 1:
            raise ValueError("Course / single GPU allocation proof missing")
        state = row["state"].split()[0]
        if state not in {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY"}:
            raise ValueError("Prior allocation has no actual terminal state")
        elapsed = positive(row["elapsed_raw"], "elapsed_raw", allow_zero=True)
        if elapsed != PRIOR_GPU_SECONDS[job]:
            raise ValueError("Actual prior wall time differs from the frozen ledger")
        observed[job] = int(elapsed)
    if observed != PRIOR_GPU_SECONDS:
        raise ValueError("Incomplete native prior allocation ledger")
    return {"used_seconds": sum(observed.values()),
            "remaining_seconds": CAP - sum(observed.values()), "cap_seconds": CAP}


def elapsed_from_allocation(start_utc, now_utc=None):
    """Charge startup, preloading, proofs, and partial work from Slurm StartTime."""
    start = datetime.fromisoformat(start_utc.replace("Z", "+00:00"))
    now = datetime.now(timezone.utc) if now_utc is None else datetime.fromisoformat(now_utc.replace("Z", "+00:00"))
    if start.tzinfo is None or now.tzinfo is None or now < start:
        raise ValueError("Actual UTC allocation start/time proof required")
    return math.ceil((now - start).total_seconds())


def full_sequence_projection(*, new_gpu_seconds_charged, candidate_epoch_seconds,
                             reference_epoch_seconds, development_epochs_done,
                             encoder_frames_per_second, confirmation_frame_bound,
                             encoder_equivalence_passed, fixed_overhead_seconds=900,
                             safety_multiplier=1.15):
    """Reserve complete development AND fresh 12-model confirmation before continuing.

    Rates are conservative maxima over measured complete training + validation
    epochs. Counts remain 6 development models and 12 fresh confirmation models,
    each 120 epochs. Only epochs actually retained toward development may be
    subtracted. Profiling and encoding already used are charged independently.
    This function never authorizes a submission or declares scientific success.
    """
    charged = positive(new_gpu_seconds_charged, "all new allocated GPU seconds", True)
    overhead = positive(fixed_overhead_seconds, "remaining preparation/proof/I/O reserve")
    safety = positive(safety_multiplier, "safety multiplier")
    if safety < 1.10 or overhead < 600:
        raise ValueError("Resource safety reserves cannot be reduced to fit the budget")
    if encoder_equivalence_passed is not True:
        raise ValueError("Native actual encoder / duplicate execution proof required")
    throughput = positive(encoder_frames_per_second, "actual encoder throughput")
    frames = positive(confirmation_frame_bound, "complete fresh RGB encoding bound")
    if int(frames) != frames or frames < 72928:
        raise ValueError("Cannot use a reduced or incomplete fresh frame bound")
    if set(candidate_epoch_seconds) != set(TASKS) or set(reference_epoch_seconds) != set(TASKS):
        raise ValueError("Both tasks and fresh reference rates must be measured")
    if set(development_epochs_done) != set(TASKS):
        raise ValueError("Complete per-task development coverage ledger required")
    development = confirmation = 0.0
    for task in TASKS:
        candidate = positive(candidate_epoch_seconds[task], task + " complete candidate epoch")
        reference = positive(reference_epoch_seconds[task], task + " complete reference epoch")
        counts = development_epochs_done[task]
        if set(counts) != {17, 29, 43}:
            raise ValueError("All original seeds must remain in the resource plan")
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 or n > 120 for n in counts.values()):
            raise ValueError("Invalid retained development epoch count")
        development += (360 - sum(counts.values())) * candidate
        confirmation += 360 * (candidate + reference)
    encoding = frames / throughput
    remaining_projection = safety * (development + confirmation + encoding) + overhead
    total_projection = sum(PRIOR_GPU_SECONDS.values()) + charged + remaining_projection
    return {"cost_prediction_only": True, "phase1_go": False,
            "development_models": 6, "confirmation_models": 12, "each_epochs": 120,
            "prior_actual_gpu_seconds": sum(PRIOR_GPU_SECONDS.values()),
            "new_actual_gpu_seconds_charged": charged,
            "remaining_development_raw_seconds": development,
            "complete_confirmation_training_raw_seconds": confirmation,
            "complete_confirmation_encoding_raw_seconds": encoding,
            "safety_multiplier": safety, "remaining_fixed_reserve_seconds": overhead,
            "projected_cumulative_gpu_seconds": total_projection,
            "cap_seconds": CAP, "fits_shared_cap": total_projection <= CAP,
            "if_rejected": "Stop with true partial evidence; do not shorten scope or relax gates"}
