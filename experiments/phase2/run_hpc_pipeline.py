"""Run bounded independent encoding/data work inside one existing course A100 job.

This runner does not install software, submit/requeue jobs, train a predictor,
produce deletion supervision, or verify baseline-model planning.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
TOTAL_SECONDS = 2400
REFERENCE = ROOT / "results/reference/gate/resolved_config.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    temporary = Path(str(path) + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def code_hashes():
    paths = list((ROOT / "experiments/phase2").glob("*.py"))
    paths += list((ROOT / "experiments/predictor/control/src/phase1").glob("*.py"))
    paths += [REFERENCE, ROOT / "experiments/predictor/control/source_lock.json",
              ROOT / "experiments/phase2/encoder_pipeline.sbatch"]
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(paths)}


def verify_checkout(expected):
    if not re.fullmatch(r"[0-9a-fA-F]{40}", expected):
        raise ValueError("--expected-commit must be a complete 40-character commit SHA")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=10).strip()
    status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"],
                                     cwd=ROOT, text=True, timeout=10)
    if head.lower() != expected.lower():
        raise ValueError("Checkout HEAD differs from the explicitly expected commit")
    if status:
        raise ValueError("Checkout must be clean, including untracked files; keep run logs outside the repository")
    return {"head": head, "clean": True, "source_sha256": code_hashes()}


def terminate_group(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        # The encoder-pilot parent handles TERM and stops its own worker group.
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def validate_data(task_root, output):
    """Validate the exact new small dataset; never regenerate existing data."""
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "experiments/predictor/control/src"))
    from experiments.phase2.prepare_controlled_data import make_config
    from experiments.phase2.prepare_encoded_samples import make_encoder_config, verify_prepared_inputs, verify_retained_source
    data_root = task_root / "data"
    raw = json.loads((data_root / "resolved_config.json").read_text())
    expected = make_config(json.loads(REFERENCE.read_text()), data_root,
                           seed=20261020, train_pairs=16, eval_pairs=4)
    if raw["data"] != expected["data"]:
        raise ValueError("Existing dataset does not match the explicitly expected new seed/counts/dynamics")
    config = make_encoder_config(raw, data_root, task_root / "cache/ijepa",
                                 device="cuda", local_files_only=True)
    source = verify_retained_source()
    manifest, index, roles = verify_prepared_inputs(data_root, config)
    record = {"status": "existing_or_new_data_validated_without_regeneration",
              "data_seed": 20261020, "train_pairs_per_task": 16, "eval_pairs_per_task_split": 4,
              "dataset_signature": manifest["signature"], "episodes": len(index["records"]),
              "split_roles": roles, "source_proof": source,
              "preparation_manifest_sha256": sha256(data_root / "preparation_manifest.json"),
              "predictor_trained": False}
    write_json(output, record)
    print(json.dumps(record), flush=True)


def run(args):
    began = time.monotonic()
    deadline = began + TOTAL_SECONDS
    task_root = args.task_root.expanduser().resolve()
    job = os.environ.get("SLURM_JOB_ID", "")
    if not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", job):
        raise ValueError("Require the current existing Slurm allocation's numeric SLURM_JOB_ID")
    output = task_root / "runs" / job
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "pipeline_manifest.json"
    manifest = {"schema_version": 1, "status": "starting", "started_utc": now(),
                "host": socket.gethostname(), "slurm_job_id": job,
                "slurm_account": os.environ.get("SLURM_JOB_ACCOUNT"),
                "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
                "task_root": str(task_root), "repository_root": str(ROOT),
                "python": sys.executable, "expected_commit": args.expected_commit,
                "global_wall_timeout_seconds": TOTAL_SECONDS, "stages": [],
                "scope": "new_development_data_frozen_ijepa_encoding_and_sample_audit_only",
                "predictor_trained": False, "learned_retention_scorer_trained": False,
                "deletion_supervision_generated": False, "baseline_model_planning_verified": False,
                "scientific_phase2_completed": False, "submitted_or_requeued_job": False}
    write_json(manifest_path, manifest)
    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    active = None

    def interrupt(signum, _frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    def stage(name, command, maximum=None):
        nonlocal active
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Global pipeline wall-time budget exhausted before " + name)
        timeout = min(remaining, maximum) if maximum else remaining
        log_path = output / (name + ".log")
        record = {"name": name, "command": command, "status": "running", "started_utc": now(),
                  "timeout_seconds": timeout, "log": str(log_path), "exit_code": None}
        manifest["stages"].append(record)
        write_json(manifest_path, manifest)
        started = time.monotonic()
        try:
            with log_path.open("x") as log:
                active = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                          stderr=subprocess.STDOUT, start_new_session=True)
                record["pid"] = active.pid
                record["process_group"] = active.pid
                write_json(manifest_path, manifest)
                code = active.wait(timeout=timeout)
            record["exit_code"] = code
            if code:
                raise RuntimeError(f"{name} exited {code}; inspect {log_path}")
            record["status"] = "completed"
        except subprocess.TimeoutExpired:
            record["status"] = "timed_out"
            raise
        except KeyboardInterrupt:
            record["status"] = "interrupted"
            raise
        except Exception:
            record["status"] = "failed"
            raise
        finally:
            terminate_group(active)
            if active is not None:
                record["exit_code"] = active.returncode
            active = None
            record.update(ended_utc=now(), elapsed_seconds=time.monotonic() - started)
            if log_path.is_file():
                record["log_sha256"] = sha256(log_path)
            write_json(manifest_path, manifest)

    for sig in old_handlers:
        signal.signal(sig, interrupt)
    try:
        if not task_root.is_dir() or not (task_root / "cache/ijepa").is_dir():
            raise ValueError("CPU preparation must first provide task_root/cache/ijepa")
        manifest["checkout"] = verify_checkout(args.expected_commit)
        manifest["status"] = "running"
        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED="1", HF_HUB_DISABLE_XET="1", HF_HUB_OFFLINE="1",
                   TRANSFORMERS_OFFLINE="1", HF_HOME=str(task_root / "cache/huggingface"),
                   HF_HUB_CACHE=str(task_root / "cache/huggingface/hub"),
                   XDG_CACHE_HOME=str(task_root / "cache/xdg"), TORCH_HOME=str(task_root / "cache/torch"),
                   OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", SDL_VIDEODRIVER="dummy")
        py = sys.executable
        stage("pusht_encoder", [py, str(ROOT / "experiments/phase2/run_encoder_pilot.py"),
              "--output", str(output / "pusht"), "--cache-dir", str(task_root / "cache"),
              "--timeout-seconds", "600"], maximum=660)
        hardware = json.loads((output / "pusht/hardware.json").read_text())
        pilot = json.loads((output / "pusht/run_manifest.json").read_text())
        if pilot.get("status") != "completed" or not pilot.get("encoding_verified"):
            raise RuntimeError("PushT encoder stage did not verify its real encoding output")
        if hardware.get("slurm_job_id") != job or "A100" not in hardware.get("torch_device_name", "").upper():
            raise RuntimeError("Encoder hardware evidence does not match this job and A100")
        manifest["verified_hardware"] = hardware
        data_root = task_root / "data"
        if not data_root.exists() or not any(data_root.iterdir()):
            stage("prepare_controlled_data", [py, str(ROOT / "experiments/phase2/prepare_controlled_data.py"),
                  "--output", str(data_root), "--seed", "20261020", "--train-pairs", "16", "--eval-pairs", "4"], maximum=300)
        else:
            manifest["stages"].append({"name": "prepare_controlled_data", "status": "existing_data_preserved_pending_validation",
                                       "path": str(data_root), "regenerated": False})
        stage("validate_controlled_data", [py, str(Path(__file__).resolve()), "--task-root", str(task_root),
              "--validate-data-output", str(output / "controlled_data_validation.json")], maximum=180)
        stage("encoded_samples", [py, str(ROOT / "experiments/phase2/prepare_encoded_samples.py"),
              "--data-root", str(data_root), "--cache-dir", str(task_root / "cache/ijepa"),
              "--device", "cuda", "--output", str(output / "encoded"), "--local-files-only"])
        encoded = json.loads((output / "encoded/manifest.json").read_text())
        if encoded.get("status") != "official_encoding_and_sample_audit_completed":
            raise RuntimeError("Encoded-sample stage ended without the complete official-encoder sample audit")
        if code_hashes() != manifest["checkout"]["source_sha256"]:
            raise RuntimeError("Scientific source files changed during execution")
        manifest.update(status="completed", official_encoding_and_sample_audit_completed=True,
                        sample_rows=encoded.get("sample_rows"), encoded_episodes=encoded.get("encoded_episodes"))
    except KeyboardInterrupt:
        manifest.update(status="interrupted", error=traceback.format_exc())
    except (subprocess.TimeoutExpired, TimeoutError):
        manifest.update(status="timed_out", error=traceback.format_exc())
    except Exception:
        manifest.update(status="failed", error=traceback.format_exc())
    finally:
        terminate_group(active)
        manifest.update(ended_utc=now(), driver_wall_seconds=time.monotonic() - began,
                        resource_accounting_note="Driver wall time only; obtain scheduler allocation time separately, including pre-driver/idle time.")
        write_json(manifest_path, manifest)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    print(json.dumps({"status": manifest["status"], "manifest": str(manifest_path),
                      "scientific_phase2_completed": False}), flush=True)
    return 0 if manifest["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", required=True, type=Path)
    parser.add_argument("--expected-commit", help="Required for the pipeline: complete verified Git HEAD SHA")
    parser.add_argument("--validate-data-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.validate_data_output is not None:
        if args.validate_data_output.exists():
            parser.error("validation output already exists")
        validate_data(args.task_root.expanduser().resolve(), args.validate_data_output)
    else:
        if not args.expected_commit:
            parser.error("--expected-commit is required")
        raise SystemExit(run(args))


if __name__ == "__main__":
    main()
