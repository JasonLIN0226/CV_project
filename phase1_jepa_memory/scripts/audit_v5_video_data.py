#!/usr/bin/env python3
"""Allocated CPU-only PTS/eligibility audit; no features, labels, or training."""
import argparse
from bisect import bisect_right
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import socket
import sys
import time
import zipfile

from v5_video_causality import as_pair, annotation_frame_at, episode_starts, query_steps

MANIFEST_SHA = "eb4ff098e79b4bf1038db1115c8aa9a7b6872c390ef38347c9e2ad84752b4d01"
DESIGN_SHA = "28c4b7b748db530db5a8fd341603a8ab1e0245f166a43999290a4f134e730fe3"
INTEGRITY_SHA = "db5034199cd96f9ce3e8077b1d950683399ccb34ddf35c5a3f93eea7f86d3672"
COURSE = "csci_ga_2271-2026fa"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, indent=2) + "\n")
    temp.replace(path)


def no_symlink(path):
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("symlink path forbidden")


def decode_at(av, path, clock, expected, threads):
    """Seek to an earlier keyframe and decode in presentation order.

    The later frame used to terminate the search is discarded, never an input.
    Decoder correctness is checked against the independently demuxed timeline.
    """
    with av.open(str(path), mode="r") as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = threads
        stream.codec_context.thread_type = "SLICE"
        container.seek(int(clock / stream.time_base), stream=stream,
                       backward=True, any_frame=False)
        chosen = None
        previous = None
        count = 0
        for frame in container.decode(stream):
            count += 1
            if frame.pts is None or frame.time_base is None or frame.is_corrupt:
                raise ValueError("missing/corrupt decoded presentation timestamp")
            stamp = Fraction(frame.pts) * frame.time_base
            if previous is not None and stamp <= previous:
                raise ValueError("decoded presentation order is not strict")
            previous = stamp
            if stamp > clock:
                break
            chosen = (stamp, frame.width, frame.height)
        if chosen is None or chosen[0] != expected or chosen[0] > clock:
            raise ValueError("causal decoded selection differs from packet timeline")
        return {"requested_seconds": as_pair(clock), "selected_seconds": as_pair(chosen[0]),
                "lag_seconds": as_pair(clock - chosen[0]), "width": chosen[1],
                "height": chosen[2], "decoded_frames_for_probe": count,
                "annotation_grid_index_zero_origin_30fps": annotation_frame_at(chosen[0])}


def audit_video(av, path, row, deadline):
    no_symlink(path)
    if not path.is_file() or path.stat().st_size != row["bytes"]:
        raise ValueError("actual video size differs from frozen integrity proof")
    # The full file hash is repeated here so PTS metadata binds current actual bytes.
    if sha(path) != row["sha256"]:
        raise ValueError("actual current video hash differs from pinned manifest")
    times = []
    dts_previous = None
    dts_missing = 0
    pts_missing = 0
    with av.open(str(path), mode="r") as container:
        if len(container.streams.video) != 1:
            raise ValueError("expected exactly one video stream")
        stream = container.streams.video[0]
        if stream.codec_context.name not in ("h264", "hevc"):
            raise ValueError("unregistered codec needs a separate audit")
        metadata = {"codec": stream.codec_context.name, "width": stream.width,
                    "height": stream.height, "time_base": as_pair(stream.time_base),
                    "average_rate": as_pair(stream.average_rate) if stream.average_rate else None,
                    "declared_frames": stream.frames, "start_time_ticks": stream.start_time,
                    "duration_ticks": stream.duration}
        for packet in container.demux(stream):
            if time.monotonic() > deadline:
                raise TimeoutError("fixed data audit deadline")
            if packet.size == 0:  # Decoder flush packets do not represent observations.
                continue
            if packet.pts is None or packet.time_base is None:
                pts_missing += 1
                continue
            times.append(Fraction(packet.pts) * packet.time_base)
            if packet.dts is None:
                dts_missing += 1
            else:
                dt = Fraction(packet.dts) * packet.time_base
                if dts_previous is not None and dt <= dts_previous:
                    raise ValueError("decode packet order is not strict")
                dts_previous = dt
    if pts_missing or dts_missing or not times:
        raise ValueError("missing packet timestamp; no gen_pts or guessed timestamps")
    sorted_times = sorted(times)
    if len(set(sorted_times)) != len(sorted_times):
        raise ValueError("duplicate packet PTS")
    origin = sorted_times[0]
    # Official 30-fps annotation mapping cannot be shifted by an inferred origin.
    if origin != 0:
        raise ValueError("nonzero origin needs explicit annotation-alignment evidence")
    if metadata["average_rate"] != [60, 1]:
        raise ValueError("actual nominal frame rate differs from official raw 60-fps mapping")
    expected_delta = Fraction(1, 60)
    if any(b - a != expected_delta for a, b in zip(sorted_times, sorted_times[1:])):
        raise ValueError("irregular 60-fps presentation grid needs explicit mapping review")
    if metadata["declared_frames"] and metadata["declared_frames"] != len(sorted_times):
        raise ValueError("declared frame count differs from MP4 packet timeline")
    starts = episode_starts(sorted_times[-1] - origin)
    selected_digest = hashlib.sha256()
    episodes = []
    for episode, start in enumerate(starts):
        lags = []
        for index in range(256):
            clock = origin + start + Fraction(index, 2)
            selected = sorted_times[bisect_right(sorted_times, clock) - 1]
            lag = clock - selected
            if lag < 0 or lag > Fraction(1, 30):
                raise ValueError("future or stale observation on the 2-fps grid")
            lags.append(lag)
            selected_digest.update(f"{episode}:{index}:{selected.numerator}/{selected.denominator}\n".encode())
        episodes.append({"episode": episode, "start_seconds": as_pair(origin + start),
                         "last_observation_seconds": as_pair(origin + start + Fraction(255, 2)),
                         "tokens": 256, "queries": len(query_steps()),
                         "maximum_selection_lag_seconds": as_pair(max(lags))})
    clocks = {origin, origin + Fraction(1, 120), origin + Fraction(1, 2)}
    for start in starts:
        clocks.update((origin + start, origin + start + Fraction(255, 2)))
    probes = []
    for clock in sorted(t for t in clocks if t <= sorted_times[-1]):
        if time.monotonic() > deadline:
            raise TimeoutError("fixed data audit deadline")
        expected = sorted_times[bisect_right(sorted_times, clock) - 1]
        probes.append(decode_at(av, path, clock, expected, 4))
    return {"recording": row["recording"], "participant": row["participant"],
            "split": row["split"], "path": row["path"], "bytes": row["bytes"],
            "sha256": row["sha256"], "stream": metadata,
            "packet_frames": len(sorted_times), "pts_reordering_present": times != sorted_times,
            "first_presentation_seconds": as_pair(origin),
            "last_presentation_seconds": as_pair(sorted_times[-1]),
            "presentation_grid": "exact60fps", "episodes": episodes,
            "sampled_frame_count": len(episodes) * 256,
            "selected_timestamps_sha256": selected_digest.hexdigest(), "decode_probes": probes,
            "eligible": bool(episodes), "raw60_to_30fps_clock_ratio_verified": True,
            "annotation_label_frame_origin_verified": False,
            "annotation_segment_endpoints_verified": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    args = parser.parse_args()
    if (not os.getenv("SLURM_JOB_ID") or os.getenv("SLURM_JOB_ACCOUNT") != COURSE
            or not socket.gethostname().startswith("b-")):
        raise RuntimeError("data audit requires actual allocated course Cloud")
    for p in (args.control, args.rgb_root, args.output, args.vendor):
        no_symlink(p)
    protocol = json.loads((args.control / "protocol.json").read_text())
    lock = json.loads((args.control / "source_lock.json").read_text())
    for name, digest in lock["files"].items():
        path = args.control / name
        no_symlink(path)
        if sha(path) != digest:
            raise ValueError("source/config byte lock mismatch")
    if sha(args.control / "manifest.json") != MANIFEST_SHA:
        raise ValueError("manifest changed")
    if sha(args.control / "design.json") != DESIGN_SHA:
        raise ValueError("design changed")
    if sha(args.rgb_root / "download_verification.json") != INTEGRITY_SHA:
        raise ValueError("actual acquisition proof changed")
    wheel = args.control / protocol["decoder"]["linux_wheel"]["filename"]
    if sha(wheel) != protocol["decoder"]["linux_wheel"]["sha256"]:
        raise ValueError("decoder wheel changed")
    # Import only bytes verified against the immutable official wheel.
    with zipfile.ZipFile(wheel) as archive:
        for entry in archive.infolist():
            relative = Path(entry.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe wheel entry")
            if not entry.is_dir():
                extracted = args.vendor / relative
                no_symlink(extracted)
                if sha(extracted) != hashlib.sha256(archive.read(entry)).hexdigest():
                    raise ValueError("decoder vendor bytes differ from pinned wheel")
    sys.path.insert(0, str(args.vendor))
    import av
    if av.__version__ != protocol["decoder"]["version"]:
        raise ValueError("unregistered decoder version")
    manifest = json.loads((args.control / "manifest.json").read_text())
    from v5_video_causality import selected_video_names
    names = selected_video_names(manifest)
    if len(names) != 78:
        raise ValueError("development scope changed")
    integrity = json.loads((args.rgb_root / "download_verification.json").read_text())
    if integrity["status"] != "COMPLETE_ALL_78_RGB_SHA256_MATCHED":
        raise ValueError("incomplete actual acquisition proof")
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "video_causality_report.json"
    started = time.monotonic()
    report = {"schema_version": 1, "status": "RUNNING", "science_result": False,
              "job_id": os.environ["SLURM_JOB_ID"], "course": COURSE,
              "host": socket.gethostname(), "protocol_sha256": sha(args.control / "protocol.json"),
              "source_lock_sha256": sha(args.control / "source_lock.json"),
              "manifest_sha256": MANIFEST_SHA, "design_sha256": DESIGN_SHA,
              "acquisition_report_sha256": INTEGRITY_SHA,
              "runtime": {"python": sys.version, "platform": platform.platform(),
                          "av": av.__version__, "ffmpeg_libraries": av.library_versions},
              "sealed_test_touched": False, "labels_downloaded": False,
              "features_extracted": False, "models_trained": False,
              "original_go": False, "videos": [], "failures": []}
    save(report_path, report)

    def interrupted(signum, _frame):
        report.update(status="INTERRUPTED", signal=signum, elapsed_seconds=time.monotonic() - started)
        save(report_path, report)
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    deadline = started + protocol["budget"]["soft_stop_seconds"]
    try:
        for i, row in enumerate(manifest["development_rgb"]):
            record_start = time.monotonic()
            try:
                result = audit_video(av, args.rgb_root / row["path"], row, deadline)
                result["elapsed_seconds"] = time.monotonic() - record_start
                report["videos"].append(result)
            except (ValueError, OSError) as error:
                report["failures"].append({"recording": row["recording"],
                                           "error_type": type(error).__name__, "reason": str(error)})
            report["elapsed_seconds"] = time.monotonic() - started
            save(report_path, report)
            print(json.dumps({"audited": i + 1, "total": 78, "failures": len(report["failures"])}), flush=True)
            if i == 0:
                # Fixed worst-case extrapolation from the first prespecified recording.
                report["first_recording_projection_seconds"] = report["elapsed_seconds"] * 78
                save(report_path, report)
                if report["first_recording_projection_seconds"] > protocol["budget"]["soft_stop_seconds"]:
                    raise TimeoutError("first-recording projection exceeds fixed data audit budget")
        eligible_participants = {v["participant"] for v in report["videos"] if v["eligible"]}
        required = {row["participant"] for row in manifest["development_rgb"]}
        report["missing_eligible_participants"] = sorted(required - eligible_participants)
        counts = {}
        for split in ("train", "validation"):
            subset = [v for v in report["videos"] if v["split"] == split]
            counts[split] = {"recordings": len(subset), "eligible_recordings": sum(v["eligible"] for v in subset),
                             "participants": len({v["participant"] for v in subset}),
                             "episodes": sum(len(v["episodes"]) for v in subset),
                             "sampled_frames": sum(v["sampled_frame_count"] for v in subset)}
        report.update(counts=counts, elapsed_seconds=time.monotonic() - started)
        report["status"] = ("PTS_AND_ELIGIBILITY_VERIFIED_ANNOTATION_LABELS_PENDING"
                            if not report["failures"] and not report["missing_eligible_participants"]
                            else "DATA_PREFLIGHT_REJECTED")
        save(report_path, report)
        return 0 if report["status"].startswith("PTS_AND") else 1
    except Exception as error:
        report.update(status="STOPPED", error_type=type(error).__name__, reason=str(error),
                      elapsed_seconds=time.monotonic() - started)
        save(report_path, report)
        print(json.dumps({"status": "STOPPED", "error_type": type(error).__name__}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
