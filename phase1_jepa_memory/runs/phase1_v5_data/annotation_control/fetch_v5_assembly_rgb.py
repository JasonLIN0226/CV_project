#!/usr/bin/env python3
"""Fetch only the pinned v5 development RGB files on allocated course Cloud."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import socket
import time

MANIFEST_SHA = "eb4ff098e79b4bf1038db1115c8aa9a7b6872c390ef38347c9e2ad84752b4d01"
COURSE = "csci_ga_2271-2026fa"


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(path):
    if sha256_file(path) != MANIFEST_SHA:
        raise ValueError("pinned manifest checksum mismatch")
    m = json.loads(path.read_text())
    rows = m["development_rgb"]
    if m["repo_id"] != "cvml-nus/assembly101" or len(rows) != 78:
        raise ValueError("unexpected development scope")
    names = set()
    counts = {"train": 0, "validation": 0}
    for r in rows:
        p = PurePosixPath(r["path"])
        if p.is_absolute() or ".." in p.parts or r["split"] not in counts:
            raise ValueError("illegal path or sealed split")
        expected = f"recordings/{r['recording']}/C10095_rgb.mp4"
        if r["path"] != expected or expected in names or len(r["sha256"]) != 64:
            raise ValueError("unexpected camera, duplicate, or checksum")
        names.add(expected)
        counts[r["split"]] += 1
    if counts != {"train": 58, "validation": 20}:
        raise ValueError("development recording counts changed")
    return m


def atomic_report(path, report):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(report, indent=2) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("check-access", "fetch"), required=True)
    args = parser.parse_args()
    m = read_manifest(args.manifest)
    # Neither mode is an experiment. Full transfer is restricted to a real batch allocation.
    if args.mode == "fetch" and (
        not os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_JOB_ACCOUNT") != COURSE
        or not socket.gethostname().startswith("b-")
    ):
        raise RuntimeError("fetch requires allocated course Cloud, not Mac/Torch")
    from huggingface_hub import get_token, get_hf_file_metadata, hf_hub_download, hf_hub_url
    if not get_token():
        raise RuntimeError("native Hugging Face login required")
    first = m["development_rgb"][0]
    head = get_hf_file_metadata(hf_hub_url(
        m["repo_id"], first["path"], repo_type="dataset", revision=m["revision"]
    ), timeout=15)
    if head.size != first["bytes"] or head.etag != first["sha256"] or head.commit_hash != m["revision"]:
        raise RuntimeError("authorized HEAD does not match pinned asset")
    if args.mode == "check-access":
        print(json.dumps({"status": "AUTHORIZED_HEAD_MATCH", "payload_bytes": 0}))
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    if any(p.is_symlink() for p in (args.output, *args.output.parents)):
        raise RuntimeError("symlink output paths are forbidden")
    report = {"status": "RUNNING", "science_result": False,
              "job_id": os.environ["SLURM_JOB_ID"], "course": COURSE,
              "revision": m["revision"], "manifest_sha256": MANIFEST_SHA,
              "sealed_test_touched": False, "verified": [], "interrupted": False}
    report_path = args.output / "download_verification.json"
    atomic_report(report_path, report)

    def interrupted(signum, _frame):
        report.update(status="INTERRUPTED", interrupted=True, signal=signum)
        atomic_report(report_path, report)
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    started = time.monotonic()
    try:
        for index, row in enumerate(m["development_rgb"]):
            target = args.output / row["path"]
            if any(p.is_symlink() for p in (target, *target.parents)):
                raise RuntimeError("symlink target paths are forbidden")
            if shutil.disk_usage(args.output).free < row["bytes"] + 5_000_000_000:
                raise RuntimeError("insufficient transfer headroom")
            if not (target.is_file() and target.stat().st_size == row["bytes"]
                    and sha256_file(target) == row["sha256"]):
                downloaded = Path(hf_hub_download(
                    m["repo_id"], row["path"], repo_type="dataset", revision=m["revision"],
                    local_dir=args.output, force_download=target.exists(),
                ))
                if downloaded != target:
                    raise RuntimeError("unexpected download location")
            if target.stat().st_size != row["bytes"] or sha256_file(target) != row["sha256"]:
                raise RuntimeError("download size or SHA256 mismatch")
            report["verified"].append({"path": row["path"], "bytes": row["bytes"], "sha256": row["sha256"]})
            report["elapsed_seconds"] = time.monotonic() - started
            atomic_report(report_path, report)
            print(json.dumps({"verified_recordings": index + 1, "total": 78}), flush=True)
            if index == 0:
                # One real transfer is a transport preflight, not an outcome-based recording selection.
                projected = report["elapsed_seconds"] * m["rgb_total_bytes"] / row["bytes"]
                report["transport_projection_seconds"] = projected
                atomic_report(report_path, report)
                if projected > 3300:
                    raise RuntimeError("transport preflight exceeds fixed one-hour acquisition budget")
        report.update(status="COMPLETE_ALL_78_RGB_SHA256_MATCHED", elapsed_seconds=time.monotonic() - started)
        atomic_report(report_path, report)
        return 0
    except Exception as e:
        report.update(status="FAILED", error_type=type(e).__name__, elapsed_seconds=time.monotonic() - started)
        # Do not print exception strings: HTTP errors can contain signed URLs or credentials.
        atomic_report(report_path, report)
        print(json.dumps({"status": "FAILED", "error_type": type(e).__name__}), flush=True)
        return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"status": "PRECHECK_FAILED", "error_type": type(error).__name__}), flush=True)
        raise SystemExit(1)
