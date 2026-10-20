#!/usr/bin/env python3
"""Submit the frozen V10 train_gpu stage after the verified resume boundary."""
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/scratch/xl4358/phase1_independent_confirmation_execution_v10_checkpoint_resume")
sys.path[:0] = [str(ROOT), str(ROOT / "original_src")]

import orchestrator as workflow


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


protocol, source = workflow.source_check()
source_sha = sha256(ROOT / "source_lock.json")
data = Path(workflow.config()["output"])

proof = ROOT / "interruption_resume_CPU_proof.json"
completion = ROOT / "stage_completed" / "resume_cpu.json"
if not proof.is_file() or not completion.is_file():
    raise SystemExit("Verified resume proof is missing")

data_link = ROOT / "fresh"
if data_link.is_symlink():
    if data_link.resolve() != data.resolve():
        raise SystemExit("Existing fresh alias points to the wrong data root")
elif data_link.exists():
    raise SystemExit("Unexpected real fresh directory in execution root")
else:
    data_link.symlink_to(data, target_is_directory=True)

report = workflow.SequenceGuard(data).check()
if report["remaining_cap_seconds"] < workflow.GPU_LIMITS["train_gpu"]:
    raise SystemExit("Complete training does not fit the remaining GPU budget")
workflow.startup_require_live(ROOT, source_sha)

receipt_path = ROOT / "receipts" / "train_gpu_submission.json"
attempt_path = ROOT / "receipts" / "train_gpu_submission_attempt.json"
if receipt_path.exists() or attempt_path.exists():
    raise SystemExit("An existing or unknown train_gpu submission must be inspected")

wrapper = ROOT / "execution_repair" / "resume_train_direct.py"
wrapper_sha = sha256(wrapper)
with attempt_path.open("x") as handle:
    json.dump(
        {
            "stage": "train_gpu",
            "unknown_do_not_retry": True,
            "resume_cpu_job": 9073,
            "wrapper_sha256": wrapper_sha,
        },
        handle,
    )

args = [
    "/opt/slurm/bin/sbatch",
    "--parsable",
    "--account=csci_ga_2271-2026fa",
    "--partition=c12m85-a100-1",
    "--cpus-per-task=4",
    "--mem=60G",
    "--time=180",
    "--no-requeue",
    "--job-name=p1-train_gpu-4133e433r1",
    "--output=" + str(ROOT / "train_gpu-r1-%j.out"),
    "--error=" + str(ROOT / "train_gpu-r1-%j.err"),
    "--gres=gpu:1",
    "--export=NIL",
    "--wrap=/scratch/xl4358/phase1_env/bin/python -u " + str(wrapper),
]
raw = subprocess.run(
    args, check=True, capture_output=True, text=True, timeout=30
).stdout.strip()
job = int(raw.split(";")[0])
receipt = {
    "stage": "train_gpu",
    "job": job,
    "parent": None,
    "stdout": raw,
    "source_lock_sha256": source_sha,
    "actual_submission_UTC": datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat(),
    "resume_cpu_job": 9073,
    "fresh_alias": str(data_link),
    "data_root": str(data),
    "repair_wrapper_sha256": wrapper_sha,
    "repair_reason": (
        "The frozen resume_cpu stage completed its proof, but automatic "
        "train_gpu submission used ROOT/fresh instead of config.output."
    ),
}
receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")

guard = workflow.SequenceGuard(data)
guard.state.setdefault("GPU_job_ids", []).append(job)
guard.save()
print(json.dumps({"receipt": receipt, "budget_before": report}, indent=2))
