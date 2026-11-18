"""Portable orchestration for the frozen v5 PushT family simulator and writer."""
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
import importlib.metadata as metadata
import json
import multiprocessing
from pathlib import Path
import platform
import subprocess
import time

import prepare_pusht_cog_marker_data as marker


def opencv_packages(version=metadata.version):
    found = {}
    for name in ("opencv-python", "opencv-python-headless"):
        try:
            found[name] = version(name)
        except metadata.PackageNotFoundError:
            continue
    if not found:
        raise RuntimeError("Install opencv-python or opencv-python-headless")
    return found


def provenance():
    import gym_pusht
    import gym_pusht.envs.pusht as installed

    here = Path(__file__).resolve()
    root = here.parents[2]
    sources = [here.with_name(name + ".py") for name in (
        "pusht", "pusht_cog_cue", "pusht_cog_coherent", "pusht_cog_matched", "pusht_cog_natural",
        "pusht_cog_marker", "prepare_pusht_cog_marker_data", "prepare_pusht_cog_v4_data")]
    sources += [here, root / "scripts/prepare_data/pusht.py"]
    hashes = {p.relative_to(root).as_posix(): marker.sha256(p) for p in sources}
    for module in (gym_pusht, installed):
        hashes[module.__name__] = marker.sha256(Path(module.__file__))
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    packages = {n: metadata.version(n) for n in
                ("numpy", "gym-pusht", "gymnasium", "pygame", "pymunk", "shapely")}
    return dict(git_base=commit, python=platform.python_version(), os=platform.system(),
                packages={**packages, **opencv_packages()}, source_sha256=hashes)


def collect_split(output, split, count, processes, ordered_map):
    accepted, attempted = 0, 0
    records, assignment, rejected = [], {}, []
    while accepted < count:
        batch = min(processes, count - accepted, marker.MAX_CANDIDATES_PER_SPLIT - attempted)
        if batch == 0:
            raise RuntimeError(f"Candidate limit reached for {split}")
        start = marker.SPLIT_BASE_SEEDS[split] + attempted
        jobs = [(split, seed) for seed in range(start, start + batch)]
        attempted += batch
        for candidate in ordered_map(marker.collect_candidate, jobs):
            if not candidate["valid"]:
                rejected.append(candidate)
                continue
            family = f"cog_{split}_{accepted:03d}"
            classes = marker.sibling_classes(candidate["seed"])
            family_rows = marker.write_family(output, family, candidate, classes)
            for row in family_rows:
                row["observations"] = Path(row["observations"]).as_posix()
            records.extend(family_rows)
            assignment[family] = classes
            accepted += 1
    return records, assignment, rejected


def run(output, processes, splits):
    if processes < 1 or not splits or len(set(splits)) != len(splits) or set(splits) - marker.SPLIT_COUNTS.keys():
        raise ValueError("Choose distinct v5 splits and a positive process count")
    counts = {s: marker.SPLIT_COUNTS[s] for s in splits}
    source = provenance()
    output.mkdir(parents=True, exist_ok=False)
    (output / "observations").mkdir()
    (output / "evaluation_only").mkdir()
    started = datetime.now(timezone.utc).isoformat()
    timer = time.monotonic()
    protocol = dict(schema_version=5, protocol_id="pusht-hidden-cog-marker-v5-20261109",
                    split_family_counts=counts, split_base_seeds={s: marker.SPLIT_BASE_SEEDS[s] for s in splits},
                    siblings_per_family=len(marker.cue.COG_CLASSES), observation_shape=[96, 96, 3],
                    recent_length=marker.RECENT_LENGTH, horizon=marker.cue.HORIZON,
                    action_semantics="absolute agent target XY in [0,512]",
                    normalization="actions_absolute / float32(256) - float32(1)",
                    marker=dict(frames=marker.cue.MARKER_FRAMES, radius_px_at_96=marker.cue.MARKER_RADIUS_PX,
                                start_range_inclusive=marker.cue.MARKER_START_RANGE, rgb=marker.cue.MARKER_COLOR))
    marker.write_json(output / "protocol.json", protocol)
    protocol_hash = marker.sha256(output / "protocol.json")
    marker.write_json(output / "protocol_frozen_before_collection.json",
                      dict(frozen_utc=started, before_environment_creation=True, protocol_sha256=protocol_hash, **source))
    records, assignment, rejected = [], {}, []
    context = multiprocessing.get_context("spawn").Pool(processes) if processes > 1 else nullcontext(None)
    with context as pool:
        for split, count in counts.items():
            rows, classes, failures = collect_split(output, split, count, processes, pool.map if pool else map)
            records.extend(rows)
            assignment.update(classes)
            rejected.extend(failures)
            print(f"{split}: {count} families, {len(rows)} trajectories, {len(failures)} rejected", flush=True)
    marker.write_json(output / "evaluation_only/sibling_classes.json", assignment)
    marker.write_json(output / "rejected_candidates.json", rejected)
    index = dict(schema_version=5, protocol_sha256=protocol_hash, records=records, recent_length=marker.RECENT_LENGTH,
                 horizon=marker.cue.HORIZON, class_assignment_sha256=marker.json_sha(assignment),
                 model_archive_field_whitelist=sorted(marker.FIELDS))
    marker.write_json(output / "observation_index.json", index)
    stored = json.loads((output / "observation_index.json").read_text())["records"]
    if any(r["target_frame"] != r["query_frame"] + marker.cue.HORIZON or r["target_frame"] >= r["frames"]
           for r in stored):
        raise ValueError("Missing or misaligned prediction target")
    if marker.sha256(output / "protocol.json") != protocol_hash:
        raise ValueError("Protocol changed during collection")
    audit = marker.audit_dataset(output, protocol, stored)
    marker.write_json(output / "structural_audit.json", audit)
    artifacts = {p.relative_to(output).as_posix(): dict(bytes=p.stat().st_size, sha256=marker.sha256(p))
                 for p in sorted(output.rglob("*")) if p.is_file()}
    receipt = dict(schema_version=5, status="CPU_RGB_action_preparation_completed", device="cpu",
                   started_utc=started, completed_utc=datetime.now(timezone.utc).isoformat(),
                   collection_seconds=time.monotonic() - timer, processes=processes, splits=list(splits),
                   counts=audit["counts"], structural_audit_passed=audit["passed"], protocol_sha256=protocol_hash,
                   rejected_candidates=len(rejected), rejected_by_reason=dict(Counter(
                       reason for candidate in rejected for reason in candidate["reasons"])),
                   encoder_run=False, normalization_fitted=False, predictor_trained=False, planning_run=False,
                   artifacts=artifacts, **source)
    # Only publish the completion receipt after the audit and artifact inventory succeed.
    partial = output / ".preparation_manifest.tmp"
    marker.write_json(partial, receipt)
    partial.replace(output / "preparation_manifest.json")
    return receipt
