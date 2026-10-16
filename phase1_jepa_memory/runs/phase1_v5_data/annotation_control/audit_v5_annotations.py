#!/usr/bin/env python3
"""Data-only annotation acquisition: quarantine before numeric interpretation.

This program never computes task scores, label/frame joins, or fitted models.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import time

from fetch_v5_assembly_rgb import COURSE, read_manifest, sha256_file, atomic_report
from v5_video_causality import selected_video_names

PTS_REPORT_SHA = "c85481d98636e2e6c0b8911ba8d920195a6b76166af940b70b97333e818da996"


def safe_path(path):
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("symlink paths forbidden")


def git_blob(path):
    payload = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()


def action_catalog(raw_rows):
    actions = {}
    for row in raw_rows:
        item = tuple(int(row[k]) for k in ("action_id", "verb_id", "noun_id"))
        action, verb, noun = item
        if action in actions or not 0 <= verb < 24 or not 0 <= noun < 90:
            raise ValueError("invalid action vocabulary")
        actions[action] = (verb, noun)
    return actions


def development_segments(raw_rows, selected, catalog):
    """Do not access label fields of any unselected recording or camera."""
    for raw in raw_rows:
        video = raw.get("video")
        if video not in selected:
            continue
        start, end = int(raw["start_frame"]), int(raw["end_frame"])
        if start < 0 or end < start:
            raise ValueError("invalid selected interval")
        available = all(raw.get(k, "") != "" for k in ("action_id", "verb_id", "noun_id"))
        labels = None
        if available:
            labels = tuple(int(raw[k]) for k in ("action_id", "verb_id", "noun_id"))
            if labels == (-1, -1, -1):
                labels = None
            elif labels[0] not in catalog or catalog[labels[0]] != labels[1:]:
                raise ValueError("selected labels disagree with action vocabulary")
        yield {"video": video, **selected[video], "start_frame": start,
               "end_frame": end, "labels": labels}


def range_compatible(start, end, raw_frames):
    """Only reject a range impossible under either 0/1-index hypothesis.

    This conservative bound never authorizes choosing a timestamp origin.
    """
    return 0 <= start <= end <= (raw_frames + 1) // 2


def check_file(path, meta):
    safe_path(path)
    if path.stat().st_size != meta["size"]:
        raise ValueError("annotation size mismatch")
    expected = meta.get("lfs")
    if expected:
        if sha256_file(path) != expected["sha256"]:
            raise ValueError("annotation checksum mismatch")
    elif git_blob(path) != meta["blob_id"]:
        raise ValueError("annotation Git blob mismatch")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pts-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_ACCOUNT") != COURSE
            or not socket.gethostname().startswith("b-")):
        raise RuntimeError("annotation payload acquisition requires allocated course Cloud")
    manifest = read_manifest(args.manifest)
    if sha256_file(args.pts_report) != PTS_REPORT_SHA:
        raise ValueError("actual PTS report binding mismatch")
    pts = json.loads(args.pts_report.read_text())
    if pts["status"] != "PTS_AND_ELIGIBILITY_VERIFIED_ANNOTATION_LABELS_PENDING":
        raise ValueError("PTS prerequisite incomplete")
    selected = selected_video_names(manifest)
    raw_frames = {v["path"].removeprefix("recordings/"): v["packet_frames"]
                  for v in pts["videos"]}
    if set(raw_frames) != set(selected):
        raise ValueError("selected video scope mismatch")
    safe_path(args.output)
    if args.output.exists():
        raise ValueError("immutable output already exists; do not duplicate a run")
    os.umask(0o077)
    args.output.mkdir(parents=True)
    quarantine = args.output / "quarantine"
    quarantine.mkdir()
    report_path = args.output / "annotation_availability_report.json"
    report = {"status": "RUNNING_DATA_ONLY", "job_id": os.environ["SLURM_JOB_ID"],
              "course": COURSE, "science_result": False, "models_trained": False,
              "features_extracted": False, "label_frame_join_executed": False,
              "label_origin_verified": False, "phase1_passed": False,
              "pts_report_sha256": PTS_REPORT_SHA, "source_files": [],
              "official_partition_label_payloads_downloaded_to_quarantine": False,
              "sealed_participant_label_rows_numerically_interpreted": False,
              "sealed_participant_labels_exported_or_evaluated": False}
    atomic_report(report_path, report)
    started = time.monotonic()

    def terminate(signum, _frame):
        report.update(status="INTERRUPTED", signal=signum, elapsed_seconds=time.monotonic()-started)
        atomic_report(report_path, report)
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        from huggingface_hub import get_token, hf_hub_download
        if not get_token():
            raise RuntimeError("existing HPC native login required")
        files = manifest["annotation_catalog_metadata"] + manifest["annotation_label_source_metadata"]
        paths = {}
        for meta in files:
            if time.monotonic()-started > 780:
                raise RuntimeError("fixed annotation preflight budget exhausted")
            path = Path(hf_hub_download(manifest["repo_id"], meta["path"], repo_type="dataset",
                        revision=manifest["revision"], local_dir=quarantine))
            if path != quarantine / meta["path"]:
                raise ValueError("unexpected annotation location")
            check_file(path, meta)
            paths[meta["path"]] = path
            report["source_files"].append({"path": meta["path"], "bytes": meta["size"],
                                           "sha256": sha256_file(path)})
            atomic_report(report_path, report)
        report["official_partition_label_payloads_downloaded_to_quarantine"] = True
        with paths["annotations/fine-grained-annotations/actions.csv"].open(newline="") as handle:
            catalog = action_catalog(csv.DictReader(handle))
        if set(catalog) != set(range(1380)):
            raise ValueError("official action dictionary incomplete")
        report["action_vocabulary"] = {"actions": len(catalog), "verbs": len({v[0] for v in catalog.values()}),
                                       "nouns": len({v[1] for v in catalog.values()})}
        summaries = {video: {**info, "segments": 0, "labels_available": 0, "labels_unavailable": 0,
                            "min_start": None, "max_end": None, "range_incompatible": 0,
                            "official_sources": {}} for video, info in selected.items()}
        export = args.output / "development_intervals_unaligned.jsonl"
        with export.open("x") as output:
            for meta in manifest["annotation_label_source_metadata"]:
                with paths[meta["path"]].open(newline="") as handle:
                    rows = csv.DictReader(handle)
                    if not {"video", "start_frame", "end_frame"} <= set(rows.fieldnames or []):
                        raise ValueError("official CSV schema mismatch")
                    for segment in development_segments(rows, selected, catalog):
                        if time.monotonic()-started > 780:
                            raise RuntimeError("fixed annotation preflight budget exhausted")
                        stat = summaries[segment["video"]]
                        stat["segments"] += 1
                        field = "labels_available" if segment["labels"] is not None else "labels_unavailable"
                        stat[field] += 1
                        start, end = segment["start_frame"], segment["end_frame"]
                        stat["min_start"] = start if stat["min_start"] is None else min(stat["min_start"], start)
                        stat["max_end"] = end if stat["max_end"] is None else max(stat["max_end"], end)
                        stat["range_incompatible"] += not range_compatible(start, end, raw_frames[segment["video"]])
                        stat["official_sources"][meta["path"]] = stat["official_sources"].get(meta["path"], 0)+1
                        output.write(json.dumps(segment, separators=(",", ":"))+"\n")
        report["development_recordings"] = summaries
        report["export"] = {"path": str(export), "bytes": export.stat().st_size,
                            "sha256": sha256_file(export), "development_only": True, "aligned": False}
        complete = all(v["segments"] > 0 and v["labels_unavailable"] == 0 and v["range_incompatible"] == 0
                       for v in summaries.values())
        report.update(status="DEVELOPMENT_LABEL_AVAILABILITY_AND_RANGES_VERIFIED_ORIGIN_PENDING"
                      if complete else "DATA_PREFLIGHT_REJECTED_LABELS_OR_RANGES",
                      elapsed_seconds=time.monotonic()-started,
                      total_development_segments=sum(v["segments"] for v in summaries.values()),
                      readout_authorized=False)
        atomic_report(report_path, report)
        print(json.dumps({k: report[k] for k in ("status", "job_id", "total_development_segments", "elapsed_seconds")}), flush=True)
        return 0 if complete else 2
    except Exception as error:
        report.update(status="FAILED", error_type=type(error).__name__, elapsed_seconds=time.monotonic()-started)
        atomic_report(report_path, report)
        print(json.dumps({"status": "FAILED", "error_type": type(error).__name__}), flush=True)
        return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"status": "PRECHECK_FAILED", "error_type": type(error).__name__}), flush=True)
        raise SystemExit(1)
