"""Pinned v5 single-image encoding; no labels, predictor, or train/test fitting."""
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import os
import queue
import shutil
import signal
import socket
import threading
import time

import numpy as np
import torch

from .encoders import FrozenIJepa

COURSE = "csci_ga_2271-2026fa"


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def no_symlink(path):
    if any(p.is_symlink() for p in (Path(path), *Path(path).parents)):
        raise ValueError("payload/source symlink paths are forbidden")


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def encoding_plan(manifest, audited):
    """Immutable development names and actual eligible episodes, never labels."""
    from v5_video_causality import selected_video_names, episode_starts
    allowed = selected_video_names(manifest)
    if len(allowed) != 78 or len(audited["videos"]) != 78:
        raise ValueError("development scope changed")
    videos = {v["path"]: v for v in audited["videos"]}
    if len(videos) != 78 or set(videos) != {x["path"] for x in manifest["development_rgb"]}:
        raise ValueError("actual PTS scope incomplete or duplicated")
    plan = []
    for row in manifest["development_rgb"]:
        video = videos[row["path"]]
        for key in ("participant", "split", "path", "sha256", "bytes"):
            if video[key] != row[key]:
                raise ValueError("actual audit and manifest binding differs")
        if video["first_presentation_seconds"] != [0, 1] or video["presentation_grid"] != "exact60fps":
            raise ValueError("unregistered presentation clock")
        starts = episode_starts(Fraction(*video["last_presentation_seconds"]))
        if not starts or [Fraction(*e["start_seconds"]) for e in video["episodes"]] != starts:
            raise ValueError("episode selection changed")
        count = len(starts) * 256
        if count != video["sampled_frame_count"]:
            raise ValueError("audited sample count changed")
        plan.append({**row, "episodes": video["episodes"], "count": count,
                     "timestamp_sha256": video["selected_timestamps_sha256"]})
    if sum(v["count"] for v in plan) != 50688:
        raise ValueError("actual development observation count changed")
    return plan


def work_projection(setup_seconds, first_work_seconds, planned_frames, observed_frames):
    if observed_frames != 512 or planned_frames != 50688:
        raise ValueError("fixed first512/all50688 preflight changed")
    return setup_seconds + first_work_seconds * planned_frames / observed_frames


def prefetch(source, capacity=4):
    """Bounded next-frame I/O. Encoder still receives independent single RGBs."""
    items = queue.Queue(capacity)
    stop = threading.Event()
    sentinel = object()

    def put(item):
        while not stop.is_set():
            try:
                items.put(item, timeout=0.1)
                return
            except queue.Full:
                pass

    def produce():
        try:
            for item in source:
                if stop.is_set():
                    break
                put((True, item))
        except BaseException as error:
            put((False, error))
        finally:
            if hasattr(source, "close"):
                source.close()
            put((True, sentinel))

    worker = threading.Thread(target=produce, daemon=True, name="v5-causal-rgb")
    worker.start()
    try:
        while True:
            okay, item = items.get()
            if not okay:
                raise item
            if item is sentinel:
                return
            yield item
    finally:
        stop.set()
        worker.join(timeout=3)


def encode_record(encoder, observed, destination, expected_count, expected_timestamp_hash,
                  progress, check_deadline):
    """Incomplete arrays retain .partial.npy; only exact completed data publishes."""
    if expected_count % 256 or not 0 < expected_count <= 1024:
        raise ValueError("unregistered episode size")
    final = Path(destination)
    partial = final.with_name(final.stem + ".partial.npy")
    no_symlink(final)
    if final.exists() or partial.exists():
        raise ValueError("encoding output already exists; do not silently redo it")
    data = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32,
                                   shape=(expected_count, 5120))
    digest = hashlib.sha256()
    batch = []
    encoded = 0
    rows_seen = 0
    iterator = prefetch(iter(observed), capacity=4)
    try:
        for clock, stamp, rgb in iterator:
            check_deadline()
            if rows_seen >= expected_count:
                raise ValueError("extra observations")
            expected_clock = Fraction(rows_seen, 2)
            if clock != expected_clock or stamp != expected_clock:
                raise ValueError("full decoding differs from locked nearest-past PTS plan")
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
                raise ValueError("encoder requires one uint8 RGB image per observation")
            digest.update(f"{rows_seen//256}:{rows_seen%256}:{stamp.numerator}/{stamp.denominator}\n".encode())
            batch.append(rgb)
            rows_seen += 1
            if len(batch) == 2:
                features = np.asarray(encoder.encode(np.stack(batch)))
                if features.shape != (2, 5120) or features.dtype != np.float32 or not np.isfinite(features).all():
                    raise ValueError("invalid full frozen encoder features")
                data[encoded:encoded+2] = features
                encoded += 2
                batch.clear()
                progress(2)
        if rows_seen != expected_count or encoded != expected_count or batch:
            raise ValueError("incomplete causal frame stream")
        if digest.hexdigest() != expected_timestamp_hash:
            raise ValueError("all selected timestamps disagree with actual PTS audit")
        data.flush()
        del data
        partial.replace(final)
        return {"path": final.name, "shape": [expected_count, 5120], "dtype": "float32",
                "bytes": final.stat().st_size, "sha256": file_sha(final),
                "timestamp_sha256": digest.hexdigest(), "forwards": expected_count//2}
    finally:
        iterator.close()


def checkpoint_files(cache_root, repository, revision):
    """Hash actual cached official bytes; no remote download or invented hashes."""
    root = Path(cache_root) / ("models--" + repository.replace("/", "--")) / "snapshots" / revision
    if not root.is_dir():
        raise ValueError("pinned official checkpoint is not present in the existing cache")
    paths = sorted(root.glob("*.safetensors")) + sorted(root.glob("*.bin"))
    if not paths:
        raise ValueError("pinned checkpoint has no cached weight files")
    paths += [root / "config.json", root / "preprocessor_config.json"]
    return {p.name: {"bytes": p.stat().st_size, "sha256": file_sha(p)} for p in paths}


def main(control, rgb_root, output, av):
    from v5_causal_frame_stream import single_frame_rgb
    if (os.getenv("SLURM_JOB_ACCOUNT") != COURSE or not os.getenv("SLURM_JOB_ID")
            or not socket.gethostname().startswith("b-")):
        raise RuntimeError("scientific encoding requires allocated course Cloud")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one allocated CUDA device required")
    if "A100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("fixed v5 A100 resource changed")
    for path in (control, rgb_root, output):
        no_symlink(path)
    if output.exists():
        raise ValueError("fresh independent encoding output required")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if hasattr(os, "sched_getaffinity"):
        affinity = sorted(os.sched_getaffinity(0))
        if len(affinity) < 4:
            raise RuntimeError("fewer than four available CPU cores")
        os.sched_setaffinity(0, affinity[:4])
    os.umask(0o077)
    output.mkdir(parents=True)
    features_root = output / "features"
    features_root.mkdir()
    protocol = json.loads((control / "protocol.json").read_text())
    manifest = json.loads((control / "manifest.json").read_text())
    audited = json.loads((control / "pts_report.json").read_text())
    plan = encoding_plan(manifest, audited)
    started = time.monotonic()
    report_path = output / "encoding_report.json"
    report = {"schema_version": 1, "status": "RUNNING_FROZEN_SINGLE_FRAME_ENCODING",
              "job_id": os.getenv("SLURM_JOB_ID"), "course": COURSE, "host": socket.gethostname(),
              "device": torch.cuda.get_device_name(0),
              "runtime": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                          "numpy": np.__version__, "av": av.__version__, "cpu_threads":4},
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
        encoder = FrozenIJepa(cfg, torch.device("cuda:0"))
        if encoder.metadata["resolved_revision"] != cfg["revision"]:
            raise ValueError("resolved official revision mismatch")
        if any(p.requires_grad or p.dtype != torch.float32 for p in encoder.model.parameters()) or encoder.model.training:
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
                projection = work_projection(setup_seconds, time.monotonic()-work_started, 50688, 512)
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
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(0))
        atomic_json(report_path, report)
        print(json.dumps({"status": report["status"], "frames_encoded": 50688}), flush=True)
        return 0
    except Exception as error:
        report.update(status="FAILED", error_type=type(error).__name__, error=str(error)[:300],
                      elapsed_seconds=time.monotonic()-started)
        atomic_json(report_path, report)
        print(json.dumps({"status": "FAILED", "error_type": type(error).__name__, "error": str(error)[:300]}), flush=True)
        return 1
