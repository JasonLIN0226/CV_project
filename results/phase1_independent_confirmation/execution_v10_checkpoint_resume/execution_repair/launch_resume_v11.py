#!/usr/bin/env python3
"""Launch one execution-only continuation after the OOD SIGTERM."""
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "original_src")]

import orchestrator as workflow
from startup_client import ensure as ensure_guardians


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archive(relative, suffix="sigterm9086"):
    source = ROOT / relative
    if not source.exists():
        return None
    target = source.with_name(source.name + "." + suffix)
    if target.exists():
        target = source.with_name(
            source.name + "." + suffix + "." + sha256(source)[:12]
        )
    shutil.move(source, target)
    return str(target.relative_to(ROOT))


protocol, source = workflow.source_check()
source_sha = sha256(ROOT / "source_lock.json")
proof = json.loads((ROOT / "interruption_resume_CPU_proof.json").read_text())
completion = json.loads((ROOT / "stage_completed/resume_cpu.json").read_text())
if (
    proof.get("reused_committed_frames") != 62418
    or proof.get("original_CPU_fit_job") != 9029
    or proof.get("original_partial_GPU_job") != 9030
    or proof.get("original_Gate_reference17_epochs") != 80
    or proof.get("no_model_forward_no_encoding_no_training") is not True
    or proof.get("tests_unopened") is not True
    or completion.get("job") != 9073
):
    raise SystemExit("Verified resume/source boundary is missing")

data = Path(workflow.config()["output"])
link = ROOT / "fresh"
if link.is_symlink():
    if link.resolve() != data.resolve():
        raise SystemExit("fresh alias points to the wrong data root")
elif link.exists():
    raise SystemExit("Unexpected physical fresh directory")
else:
    link.symlink_to(data, target_is_directory=True)

old_status = json.loads((ROOT / "watchdog_status.json").read_text())
old_job = old_status.get("GPU_jobs", {}).get("9086", {})
if old_job.get("terminal") is not False or old_job.get("charged_seconds") != 2126:
    raise SystemExit("Unexpected 9086 ledger state")

active = subprocess.run(
    [
        "/opt/slurm/bin/squeue",
        "-h",
        "-u",
        "xl4358",
        "--name=p1-train_gpu-4133e433",
        "-o",
        "%i",
    ],
    check=True,
    capture_output=True,
    text=True,
    timeout=20,
).stdout.split()
if active:
    raise SystemExit("A train_gpu allocation is already active: " + ",".join(active))

archived = {}
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
    "group_cleanup_attempts.json",
    "watchdog_cleanup_attempts.json",
    "stage_attempts/train_gpu.json",
    "receipts/train_gpu_submission_attempt.json",
    "receipts/train_gpu_submission.json",
):
    target = archive(name)
    if target:
        archived[name] = target

wrapper = ROOT / "execution_repair" / "train_gpu_resume_v11.py"
training_resume = ROOT / "execution_repair" / "fresh_training_resume.py"
launcher = Path(__file__).resolve()
manifest = {
    "status": "EXECUTION_ONLY_OOD_SIGTERM_RECOVERY",
    "source_lock_sha256": source_sha,
    "old_gpu_job": 9086,
    "old_gpu_terminal": "CANCELLED for SIGNAL Terminated at 23:21:02 UTC",
    "old_gpu_charged_seconds": 2126,
    "old_ood_session": "deb2a5fd-d331-4f48-9606-11fcc50dbe5a",
    "old_ood_job": 9077,
    "old_ood_terminal": "CANCELLED for SIGNAL Terminated at 23:11:02 UTC",
    "archived": archived,
    "launcher_sha256": sha256(launcher),
    "wrapper_sha256": sha256(wrapper),
    "training_resume_sha256": sha256(training_resume),
    "scientific_source_changes": False,
    "sample_or_gate_changes": False,
    "encode_train_rerun": False,
    "fit_cpu_rerun": False,
}
(ROOT / "execution_repair/resume_v11_manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n"
)

guardian_report = ensure_guardians(ROOT, protocol, workflow.GPU_LIMITS)
workflow.startup_require_live(ROOT, source_sha)
guard = workflow.SequenceGuard(data)
budget_before = guard.check()

attempt_path = ROOT / "receipts" / "train_gpu_resume_v11_submission_attempt.json"
with attempt_path.open("x") as handle:
    json.dump(
        {
            "stage": "train_gpu",
            "source_lock_sha256": source_sha,
            "unknown_do_not_retry": True,
            "reason": "OOD parent session SIGTERM",
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
    "--job-name=p1-train_gpu-4133e433",
    "--output=" + str(ROOT / "train_gpu-r2-%j.out"),
    "--error=" + str(ROOT / "train_gpu-r2-%j.err"),
    "--gres=gpu:1",
    "--export=NIL",
    "--wrap=/scratch/xl4358/phase1_env/bin/python -u " + str(wrapper),
]
submitted = subprocess.run(
    args, check=True, capture_output=True, text=True, timeout=30
).stdout.strip()
job = int(submitted.split(";")[0])

receipt = {
    "stage": "train_gpu",
    "job": job,
    "parent": None,
    "stdout": submitted,
    "source_lock_sha256": source_sha,
    "actual_submission_UTC": datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat(),
    "resume_cpu_job": 9073,
    "fresh_alias": str(link),
    "data_root": str(data),
    "repair_wrapper_sha256": sha256(wrapper),
    "training_resume_sha256": sha256(training_resume),
    "repair_reason": (
        "The OOD parent session was SIGTERMed at 23:11:02 UTC; its 10-minute "
        "cleanup cancelled the completed-model training allocation at 23:21:02 UTC."
    ),
}
(ROOT / "receipts/train_gpu_submission.json").write_text(
    json.dumps(receipt, indent=2) + "\n"
)
guard.state.setdefault("GPU_job_ids", []).append(job)
guard.save()
(ROOT / "execution_repair/resume_v11_submitted.json").write_text(
    json.dumps(
        {
            "receipt": receipt,
            "guardian_report": guardian_report,
            "budget_before": budget_before,
            "manifest": manifest,
        },
        indent=2,
    )
    + "\n"
)
print(json.dumps({"job": job, "receipt": receipt}, indent=2))
