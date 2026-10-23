#!/usr/bin/env python3
"""Restart only the CPU guardians after the verified resume boundary.

The original guardians exited cleanly when resume_cpu failed during its
next-stage submission.  Their terminal receipts are archived rather than
discarded, and the frozen heartbeat protocol creates a fresh independent pair.
"""
import json
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path("/scratch/xl4358/phase1_independent_confirmation_execution_v10_checkpoint_resume")
sys.path[:0] = [str(ROOT), str(ROOT / "original_src")]

import orchestrator as workflow
from startup_client import ensure


failure = ROOT / "stage_failures" / "resume_cpu.json"
if failure.exists():
    proof = json.loads((ROOT / "interruption_resume_CPU_proof.json").read_text())
    completion = json.loads((ROOT / "stage_completed" / "resume_cpu.json").read_text())
    error = json.loads(failure.read_text()).get("error", "")
    if not (
        proof.get("reused_committed_frames") == 62418
        and proof.get("original_CPU_fit_job") == 9029
        and proof.get("original_partial_GPU_job") == 9030
        and proof.get("original_Gate_reference17_epochs") == 80
        and proof.get("no_model_forward_no_encoding_no_training") is True
        and proof.get("tests_unopened") is True
        and completion.get("job") == 9073
        and "FileNotFoundError" in error
        and "sequence_budget.json" in error
    ):
        raise SystemExit("Resume failure is not the verified post-proof path bug")
    target = ROOT / "execution_repair" / "archive" / "resume_cpu.path_failure_9073.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        digest = hashlib.sha256(failure.read_bytes()).hexdigest()[:12]
        target = target.with_suffix(target.suffix + "." + digest)
    shutil.move(failure, target)


for name in (
    "watchdog_submission.json",
    "watchdog_cleanup_submission.json",
    "watchdog_status.json",
    "guardian_peer_status.json",
    "watchdog_terminal.json",
    "guardian_peer_terminal.json",
    "watchdog_submission_attempt.json",
    "watchdog_cleanup_submission_attempt.json",
    "cpu_incarnation_primary.json",
    "cpu_incarnation_peer.json",
    "receipts/train_gpu_submission_attempt.json",
):
    source = ROOT / name
    if source.exists():
        target = ROOT / (name + ".attempt9074_9075")
        if target.exists():
            digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
            target = ROOT / (name + ".attempt9074_9075." + digest)
        shutil.move(source, target)

protocol, source = workflow.source_check()
report = ensure(ROOT, protocol, workflow.GPU_LIMITS)
print(json.dumps(report, indent=2))
