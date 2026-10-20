#!/usr/bin/env python3
"""Use the live four-hour control allocation for guardian A.

The cloud CPU partition remained in NOT_RESPONDING/POWERING_UP for more than
ten minutes. This execution-only launcher starts the unchanged frozen
watchdog in the live control allocation and submits the unchanged peer to the
interactive CPU partition.
"""
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "original_src")]

import orchestrator as workflow
from heartbeat import fresh as heartbeat_fresh
from watchdog import native_job, write


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archive(relative):
    source = ROOT / relative
    if not source.exists():
        return None
    target = source.with_name(source.name + ".interactive_recovery")
    if target.exists():
        target = source.with_name(
            source.name + ".interactive_recovery." + sha256(source)[:12]
        )
    shutil.move(source, target)
    return str(target.relative_to(ROOT))


protocol, source = workflow.source_check()
source_sha = sha256(ROOT / "source_lock.json")
job_id = int(os.environ["SLURM_JOB_ID"])
if job_id != 9105:
    raise SystemExit("This launcher must run in the verified control allocation")
native = native_job(job_id)
if (
    native["JobState"] != "RUNNING"
    or native["Account"] != "csci_ga_2271-2026fa"
    or "gres/gpu=" in native.get("AllocTRES", "")
):
    raise SystemExit("Control allocation is not the expected course CPU job")

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
    raise SystemExit("GPU training already active: " + ",".join(active))

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
    "stage_failures/train_gpu.json",
    "receipts/train_gpu_resume_v11_submission_attempt.json",
    "receipts/train_gpu_submission.json",
):
    target = archive(name)
    if target:
        archived[name] = target

request = {
    "source_lock_sha256": source_sha,
    "GPU_cap_seconds": protocol["GPU_cap_seconds"],
    "prior_GPU_charge_seconds": protocol.get("interruption_resume", {}).get(
        "execution_settled_GPU_seconds", protocol["prior_settled_seconds"]
    ),
    "startup_limit_seconds": 300,
    "stage_allocation_limits": workflow.GPU_LIMITS,
    "GPU_job_names": {
        stage: "p1-" + stage + "-" + source_sha[:8]
        for stage in workflow.GPU_LIMITS
    },
}
write(ROOT / "watchdog_request.json", request)
write(
    ROOT / "watchdog_submission.json",
    {
        "job": job_id,
        "source_lock_sha256": source_sha,
        "stdout": str(job_id),
        "CPU_only": True,
    },
)

with (ROOT / "watchdog-9105-control.out").open("wb") as out, (
    ROOT / "watchdog-9105-control.err"
).open("wb") as err:
    primary_process = subprocess.Popen(
        ["/bin/bash", str(ROOT / "watchdog.sh")],
        stdout=out,
        stderr=err,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )

peer = subprocess.run(
    [
        "/opt/slurm/bin/sbatch",
        "--parsable",
        "--account=csci_ga_2271-2026fa",
        "--partition=interactive",
        "--cpus-per-task=1",
        "--mem=1500M",
        "--time=04:00:00",
        "--no-requeue",
        "--export=NIL",
        "--job-name=p1-cleanup-" + source_sha[:8],
        "--output=" + str(ROOT / "cleanup-control-%j.out"),
        "--error=" + str(ROOT / "cleanup-control-%j.err"),
        str(ROOT / "cleanup.sh"),
    ],
    check=True,
    capture_output=True,
    text=True,
    timeout=30,
).stdout.strip()
peer_job = int(peer.split(";")[0])
write(
    ROOT / "watchdog_cleanup_submission.json",
    {
        "job": peer_job,
        "source_lock_sha256": source_sha,
        "stdout": peer,
        "CPU_only": True,
    },
)

deadline = time.monotonic() + 900
while True:
    if primary_process.poll() is not None:
        raise SystemExit("Primary guardian exited before readiness")
    if (
        (ROOT / "watchdog_status.json").exists()
        and (ROOT / "guardian_peer_status.json").exists()
    ):
        break
    if time.monotonic() >= deadline:
        raise TimeoutError("Interactive guardian pair did not become ready")
    time.sleep(3)

workflow.startup_require_live(ROOT, source_sha)
guard = workflow.SequenceGuard(workflow.config()["output"])
budget_before = guard.check()

attempt_path = ROOT / "receipts" / "train_gpu_resume_v11_submission_attempt.json"
with attempt_path.open("x") as handle:
    json.dump(
        {
            "stage": "train_gpu",
            "source_lock_sha256": source_sha,
            "unknown_do_not_retry": True,
            "reason": "OOD parent SIGTERM; execution wrapper parent-chain fix",
        },
        handle,
    )

wrapper = ROOT / "execution_repair" / "train_gpu_resume_v11.py"
training_resume = ROOT / "execution_repair" / "fresh_training_resume.py"
submitted = subprocess.run(
    [
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
    ],
    check=True,
    capture_output=True,
    text=True,
    timeout=30,
).stdout.strip()
gpu_job = int(submitted.split(";")[0])
receipt = {
    "stage": "train_gpu",
    "job": gpu_job,
    "parent": None,
    "stdout": submitted,
    "source_lock_sha256": source_sha,
    "actual_submission_UTC": datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat(),
    "resume_cpu_job": 9073,
    "fresh_alias": str(ROOT / "fresh"),
    "data_root": str(Path(workflow.config()["output"])),
    "repair_wrapper_sha256": sha256(wrapper),
    "training_resume_sha256": sha256(training_resume),
    "guardian_primary_job": job_id,
    "guardian_peer_job": peer_job,
    "repair_reason": (
        "The prior OOD job was SIGTERMed at 23:11:02 UTC and cancelled the "
        "child GPU at 23:21:02 UTC. The first recovery GPU failed before model "
        "loading because its execution wrapper retained resume_cpu in CHAIN."
    ),
}
(ROOT / "receipts/train_gpu_submission.json").write_text(
    json.dumps(receipt, indent=2) + "\n"
)
guard.state.setdefault("GPU_job_ids", []).append(gpu_job)
guard.save()
provenance = {
    "status": "V11_INTERACTIVE_GUARDIANS_AND_GPU_SUBMITTED",
    "source_lock_sha256": source_sha,
    "archived": archived,
    "primary_guardian": job_id,
    "peer_guardian": peer_job,
    "gpu_job": gpu_job,
    "receipt": receipt,
    "budget_before": budget_before,
    "launcher_sha256": sha256(Path(__file__)),
    "wrapper_sha256": sha256(wrapper),
    "training_resume_sha256": sha256(training_resume),
    "no_scientific_source_change": True,
    "no_reencoding": True,
}
(ROOT / "execution_repair/resume_v11b_submitted.json").write_text(
    json.dumps(provenance, indent=2) + "\n"
)
print(json.dumps({"gpu_job": gpu_job, "primary": job_id, "peer": peer_job}))
