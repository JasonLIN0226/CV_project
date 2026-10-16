"""Separate CPU-only operational fallback; original frozen encoding core unchanged."""
from fractions import Fraction
from pathlib import Path
import json
import math
import os
import resource
import shutil
import signal
import socket
import sys
import time

import numpy as np
import torch

from .encoders import FrozenIJepa
from .v5_encoding import (COURSE, file_sha, no_symlink, atomic_json, encoding_plan,
                          work_projection, prefetch, encode_record, checkpoint_files)

CPU_OUTPUT_ROOT = Path("/scratch/xl4358/phase1_v5_real_encoding_cpu")


def require_cpu_resource(account, job, host, partition, cpus, gpu_ids,
                         gpu_count, cuda_available, cuda_count):
    if (account != COURSE or not job or not str(job).isdigit()
            or not host.startswith("b-") or partition != "n2c48m24"
            or str(cpus) != "4" or gpu_ids or str(gpu_count) not in ("", "0")
            or cuda_available or cuda_count != 0):
        raise RuntimeError("allocated course n2c48m24, requested4 CPU and zero GPU required")


def require_cpu_protocol(protocol):
    expected_encoder = {
        "backend": "ijepa", "repository": "facebook/ijepa_vith14_1k",
        "revision": "f157467ea509bc356ff9f61fd3c0d840eec5e04e",
        "pooling": "grid2", "dtype": "float32", "batch_size": 2,
        "cache_dir": "/scratch/xl4358/phase1_hf_cache/hub",
        "local_files_only": True, "frozen": True}
    expected_budget = {
        "course": COURSE, "partition": "n2c48m24", "gpu_count": 0,
        "cpus_requested": 4, "memory_gb": 16, "time_limit": "04:00:00",
        "soft_stop_seconds": 13800, "first_preflight_frames": 512,
        "actual_frames": 50688, "one_science_job_at_a_time": True}
    budget = protocol.get("budget", {})
    if (protocol.get("encoder") != expected_encoder
            or protocol.get("expected_device") != "cpu"
            or protocol.get("cache_namespace") != str(CPU_OUTPUT_ROOT)
            or any(budget.get(k) != v for k, v in expected_budget.items())
            or protocol.get("decoder", {}).get("threads") != 4):
        raise ValueError("fixed CPU resource, full-data encoder and budget protocol changed")


def cpu_projection(setup_seconds, work_seconds):
    if any(not math.isfinite(x) or x < 0 for x in (setup_seconds, work_seconds)):
        raise ValueError("invalid measured CPU projection times")
    return work_projection(setup_seconds, work_seconds, 50688, 512)


def peak_rss_bytes():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw if sys.platform == "darwin" else raw * 1024)


def main(control, rgb_root, output, av, launch_started=None):
    from v5_causal_frame_stream import single_frame_rgb
    if (os.getenv("SLURM_JOB_ACCOUNT") != COURSE or not os.getenv("SLURM_JOB_ID")
            or not socket.gethostname().startswith("b-")):
        raise RuntimeError("scientific encoding requires allocated course Cloud")
    require_cpu_resource(
        os.getenv("SLURM_JOB_ACCOUNT"), os.getenv("SLURM_JOB_ID"),
        socket.gethostname(), os.getenv("SLURM_JOB_PARTITION"),
        os.getenv("SLURM_CPUS_PER_TASK"), os.getenv("SLURM_JOB_GPUS", ""),
        os.getenv("SLURM_GPUS_ON_NODE", "0"),
        torch.cuda.is_available(), torch.cuda.device_count())
    for path in (control, rgb_root, output):
        no_symlink(path)
    protocol = json.loads((control / "protocol.json").read_text())
    require_cpu_protocol(protocol)
    if output.parent != CPU_OUTPUT_ROOT or output.name != "job_" + os.getenv("SLURM_JOB_ID"):
        raise ValueError("fresh CPU cache must bind its actual allocated job namespace")
    if output.exists():
        raise ValueError("fresh independent encoding output required")
    if hasattr(os, "sched_getaffinity"):
        affinity = sorted(os.sched_getaffinity(0))
        if len(affinity) < 4:
            raise RuntimeError("fewer than four available CPU cores")
        os.sched_setaffinity(0, affinity[:4])
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    os.umask(0o077)
    output.mkdir(parents=True)
    features_root = output / "features"
    features_root.mkdir()
    manifest = json.loads((control / "manifest.json").read_text())
    audited = json.loads((control / "pts_report.json").read_text())
    plan = encoding_plan(manifest, audited)
    started = time.monotonic() if launch_started is None else launch_started
    report_path = output / "encoding_report.json"
    report = {"schema_version": 1, "status": "RUNNING_FROZEN_SINGLE_FRAME_ENCODING",
              "job_id": os.getenv("SLURM_JOB_ID"), "course": COURSE, "host": socket.gethostname(),
              "device": "cpu", "allocated_gpu_count": 0,
              "partition": os.getenv("SLURM_JOB_PARTITION"), "cpus_requested": 4,
              "runtime": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                          "numpy": np.__version__, "av": av.__version__, "cpu_threads":4,
                          "cpu_interop_threads": 1},
              "protocol_sha256": file_sha(control/"protocol.json"),
              "source_lock_sha256": file_sha(control/"source_lock.json"),
              "manifest_sha256": file_sha(control/"manifest.json"),
              "pts_report_sha256": file_sha(control/"pts_report.json"),
              "new_weights_trained": False, "predictor_trained": False, "task_labels_read": False,
              "sealed_rgb_or_numeric_labels_read": False, "phase1_passed": False,
              "frames_encoded": 0, "forwards": 0, "planned_frames": 50688,
              "completed_recordings": [], "preflight_passed": False}
    atomic_json(report_path, report)
    deadline = started + protocol["budget"]["soft_stop_seconds"]

    def check_deadline():
        if time.monotonic() >= deadline:
            raise RuntimeError("fixed four-hour encoding budget exhausted")

    def interrupt(signum, _frame):
        report.update(status="INTERRUPTED", signal=signum, elapsed_seconds=time.monotonic()-started)
        atomic_json(report_path, report)
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)

    def progress(n):
        report["frames_encoded"] += n
        report["forwards"] += 1
        if report["frames_encoded"] % 128 == 0:
            report["elapsed_seconds"] = time.monotonic() - started
            atomic_json(report_path, report)
            print(json.dumps({"frames_encoded": report["frames_encoded"], "planned_frames": 50688}), flush=True)

    try:
        cfg = protocol["encoder"]
        report["actual_official_checkpoint_files"] = checkpoint_files(cfg["cache_dir"], cfg["repository"], cfg["revision"])
        encoder = FrozenIJepa(cfg, torch.device("cpu"))
        if encoder.metadata["resolved_revision"] != cfg["revision"]:
            raise ValueError("resolved official revision mismatch")
        if any(p.requires_grad or p.dtype != torch.float32 or p.device.type != "cpu"
               for p in encoder.model.parameters()) or encoder.model.training:
            raise ValueError("frozen eval float32 model binding failed")
        atomic_json(output/"encoder_metadata.json", encoder.metadata)
        report["encoder_metadata_sha256"] = file_sha(output/"encoder_metadata.json")
        setup_seconds = time.monotonic() - started
        report["setup_seconds"] = setup_seconds
        work_started = time.monotonic()
        for index, row in enumerate(plan):
            check_deadline()
            path = rgb_root / row["path"]
            no_symlink(path)
            if path.stat().st_size != row["bytes"] or file_sha(path) != row["sha256"]:
                raise ValueError("actual RGB bytes changed")
            if shutil.disk_usage(output).free < 5_000_000_000:
                raise RuntimeError("insufficient feature storage headroom")
            clocks = [Fraction(k, 2) for k in range(row["count"])]
            final = features_root / f"recording_{index:03d}.npy"
            meta = encode_record(encoder, single_frame_rgb(av, path, clocks, threads=4),
                                 final, row["count"], row["timestamp_sha256"], progress, check_deadline)
            if file_sha(path) != row["sha256"]:
                raise ValueError("RGB bytes changed during encoding")
            report["completed_recordings"].append({"recording": row["recording"], "participant": row["participant"],
               "split": row["split"], "rgb_path": row["path"], "rgb_sha256": row["sha256"],
               "episodes": row["episodes"], "feature": meta})
            report["elapsed_seconds"] = time.monotonic()-started
            if index == 0:
                if row["count"] != 512 or report["frames_encoded"] != 512:
                    raise ValueError("fixed first512 preflight changed")
                projection = cpu_projection(setup_seconds, time.monotonic()-work_started)
                report["first512_projected_total_seconds"] = projection
                if projection > protocol["budget"]["soft_stop_seconds"]:
                    report["status"] = "STOPPED_FIXED_FIRST512_PROJECTION_EXCEEDS_BUDGET"
                    atomic_json(report_path, report)
                    print(json.dumps({"status": report["status"], "projection_seconds": projection}), flush=True)
                    return 2
                report["preflight_passed"] = True
            atomic_json(report_path, report)
        if report["frames_encoded"] != 50688 or report["forwards"] != 25344 or len(report["completed_recordings"]) != 78:
            raise ValueError("incomplete full development encoding")
        report.update(status="ALL78_FROZEN_CAUSAL_FEATURES_COMPLETE_NOT_PHASE1_GO",
                      elapsed_seconds=time.monotonic()-started,
                      peak_process_rss_bytes=peak_rss_bytes())
        atomic_json(report_path, report)
        print(json.dumps({"status": report["status"], "frames_encoded": 50688}), flush=True)
        return 0
    except Exception as error:
        report.update(status="FAILED", error_type=type(error).__name__, error=str(error)[:300],
                      elapsed_seconds=time.monotonic()-started)
        atomic_json(report_path, report)
        print(json.dumps({"status": "FAILED", "error_type": type(error).__name__, "error": str(error)[:300]}), flush=True)
        return 1

