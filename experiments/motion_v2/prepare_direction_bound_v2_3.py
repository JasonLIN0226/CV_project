"""Prepare the frozen validation direction bound on CPU before the A100 job.

Only evaluation_only validation target positions are decoded. No encoder,
predictor, Torch, test observation, or training implementation is imported.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback


sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
MANIFEST_SHA256 = "9ac43b81eb75456e17f3542e64e8a0ab58282207752ae3712836f6d2ea20032a"
BOUND_REGISTRATION = {"method": "paired_midpoint_position_mse", "split": "validation",
                      "pairs": 96, "horizons": [3, 6], "prepared_before_gpu": True}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def array_sha256(value):
    import numpy as np
    value = np.ascontiguousarray(value)
    checksum = hashlib.sha256(json.dumps({"shape": list(value.shape), "dtype": value.dtype.str},
                                        sort_keys=True, separators=(",", ":")).encode())
    checksum.update(value.tobytes())
    return checksum.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as source:
        return json.load(source)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as destination:
        json.dump(value, destination, indent=2, ensure_ascii=False, allow_nan=False)
        destination.write("\n")


def committed_file(path):
    path = Path(path).resolve()
    relative = path.relative_to(REPO).as_posix()
    result = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=REPO, check=True,
                            capture_output=True, timeout=30)
    if result.stdout != path.read_bytes():
        raise ValueError(f"File differs from committed HEAD: {relative}")
    return relative


def validate_protocol(protocol_path):
    committed_file(protocol_path)
    protocol = read_json(protocol_path)
    if (protocol.get("status") != "FROZEN_STEP2_PROTOCOL" or
            protocol.get("registered_before_run") is not True or
            protocol.get("dataset_manifest_sha256") != MANIFEST_SHA256 or
            protocol.get("direction_bound") != BOUND_REGISTRATION):
        raise ValueError("A committed v2.3 scientific protocol must register the direction bound")
    own_relative = Path(__file__).resolve().relative_to(REPO).as_posix()
    if own_relative not in protocol["source_sha256"]:
        raise ValueError("The scientific protocol must lock this CPU helper")
    for relative, expected in protocol["source_sha256"].items():
        path = (REPO / relative).resolve()
        path.relative_to(REPO)
        if sha256(path) != expected:
            raise ValueError(f"Registered source SHA256 mismatch: {relative}")
        committed_file(path)
    return protocol


def validation_pairs(manifest):
    records = manifest["records"]
    if (Counter((r["task"], r["split"]) for r in records) !=
            {("motion", "train"): 1024, ("motion", "validation"): 192} or
            len({r["id"] for r in records}) != 1216):
        raise ValueError("Only the frozen train/validation manifest is permitted")
    groups = {}
    for rec in records:
        if rec["split"] == "validation":
            groups.setdefault(rec["pair"], []).append(rec)
    pairs = list(groups.values())
    if len(pairs) != 96 or any(len(pair) != 2 or pair[0]["sibling"] != pair[1]["id"] or
                             pair[1]["sibling"] != pair[0]["id"] for pair in pairs):
        raise ValueError("Expected exactly 96 complete reciprocal validation sibling pairs")
    return pairs


def read_target_positions(dataset, manifest):
    import numpy as np
    dataset = Path(dataset).resolve()
    if sha256(dataset / "manifest.json") != MANIFEST_SHA256:
        raise ValueError("Frozen dataset manifest SHA256 mismatch")
    pairs = validation_pairs(manifest)
    positions = np.empty((96, 2, 2, 2), dtype=np.float64)
    source_hashes = {}
    for pi, pair in enumerate(pairs):
        for si, rec in enumerate(pair):
            if not re.fullmatch(r"[0-9a-f]{24}", rec["id"]):
                raise ValueError("Unexpected validation episode ID")
            relative = f"evaluation_only/validation/{rec['id']}.npz"
            if rec["eval"] != relative:
                raise ValueError("Only validation evaluation_only files may supply target positions")
            path = (dataset / relative).resolve()
            path.relative_to(dataset)
            if sha256(path) != rec["eval_sha256"]:
                raise ValueError(f"Evaluation source SHA256 mismatch: {rec['id']}")
            with np.load(path, allow_pickle=False) as source:
                states, visible = source["states"], source["visible"]
            if (states.shape != (rec["frames"], 2) or states.dtype != np.float32 or
                    visible.shape != (rec["frames"],) or visible.dtype != np.bool_ or
                    not np.isfinite(states).all()):
                raise ValueError("Invalid audited evaluation state/visibility schema")
            times = rec["query"] + np.asarray([3, 6])
            if times[-1] >= len(states) or not np.all(visible[times]):
                raise ValueError("Validation targets must be in-range and visible")
            positions[pi, si] = states[times]
            source_hashes[relative] = rec["eval_sha256"]
    pair_ids = np.asarray([pair[0]["pair"] for pair in pairs])
    episode_ids = np.asarray([[rec["id"] for rec in pair] for pair in pairs])
    return positions, pair_ids, episode_ids, source_hashes


def midpoint_bound(positions):
    import numpy as np
    values = np.asarray(positions, dtype=np.float64)
    if values.shape != (96, 2, 2, 2) or not np.isfinite(values).all():
        raise ValueError("Expected finite validation positions with shape [96,2,2,2]")
    midpoint = values.mean(axis=1, keepdims=True)
    result = np.square(values - midpoint).mean(axis=(1, 3))
    if not np.isfinite(result).all():
        raise ValueError("Non-finite direction bound")
    return result


def verify_prepared_bound(root, dataset, manifest, protocol_path, metrics):
    """Verify the CPU result against frozen sources; return evaluation-only arrays."""
    import numpy as np
    root = Path(root).resolve()
    expected_files = {"direction_bound.npz", "status.json", "artifact_sha256.json"}
    if {path.name for path in root.iterdir()} != expected_files:
        raise ValueError("Unexpected or incomplete direction-bound artifact directory")
    hashes = read_json(root / "artifact_sha256.json")
    if set(hashes) != expected_files - {"artifact_sha256.json"}:
        raise ValueError("Direction-bound artifact SHA256 inventory is incomplete")
    for relative, expected in hashes.items():
        if sha256(root / relative) != expected:
            raise ValueError(f"Direction-bound artifact SHA256 mismatch: {relative}")
    status = read_json(root / "status.json")
    protocol = read_json(protocol_path)
    own_relative = Path(__file__).resolve().relative_to(REPO).as_posix()
    # Runtime-only provenance compatibility: keep the original CPU B bytes and
    # original preparation receipt, explicitly locked by the new registration.
    preparation_protocol = protocol
    preparation_protocol_sha = sha256(protocol_path)
    reuse = protocol.get("reused_direction_bound")
    if reuse is not None:
        original_path = (REPO / reuse["original_protocol_path"]).resolve()
        original_path.relative_to(REPO)
        committed_file(original_path)
        if sha256(original_path) != reuse["original_protocol_sha256"]:
            raise ValueError("Original B preparation protocol hash mismatch")
        preparation_protocol = read_json(original_path)
        preparation_protocol_sha = reuse["original_protocol_sha256"]
        if (preparation_protocol["dataset_manifest_sha256"] != MANIFEST_SHA256 or
                preparation_protocol["direction_bound"] != protocol["direction_bound"] or
                reuse["original_helper_path"] != own_relative or
                preparation_protocol["source_sha256"][own_relative] != reuse["original_helper_sha256"] or
                set(reuse["artifact_sha256"]) != expected_files or
                any(sha256(root / name) != expected for name, expected in reuse["artifact_sha256"].items())):
            raise ValueError("Reused CPU bound is not the explicitly frozen original preparation")
    if (status.get("status") != "passed" or status.get("actual_a100_gpu_seconds") != 0 or
            status.get("evaluation_split") != "validation" or status.get("pairs") != 96 or
            status.get("episodes") != 192 or status.get("horizons") != [3, 6] or
            status.get("test_opened_or_generated") is not False or
            status.get("training_or_model_imported") is not False or
            status.get("protocol_sha256") != preparation_protocol_sha or
            status.get("manifest_sha256") != MANIFEST_SHA256 or
            status.get("source_sha256") != {own_relative: preparation_protocol["source_sha256"][own_relative]}):
        raise ValueError("Direction-bound status does not certify the registered CPU-only preparation")
    with np.load(root / "direction_bound.npz", allow_pickle=False) as saved:
        if set(saved.files) != {"true_positions", "B", "pair_ids", "episode_ids", "horizons"}:
            raise ValueError("Unexpected direction-bound NPZ schema")
        arrays = {key: saved[key] for key in saved.files}
    positions, pair_ids, episode_ids, source_hashes = read_target_positions(dataset, manifest)
    expected = {"true_positions": positions, "B": metrics.direction_lower_bound(positions),
                "pair_ids": pair_ids, "episode_ids": episode_ids, "horizons": np.asarray([3, 6])}
    if any(arrays[key].dtype != value.dtype or not np.array_equal(arrays[key], value)
           for key, value in expected.items()):
        raise ValueError("Prepared positions/B or pair/sibling alignment differ from actual validation targets")
    if (status.get("pair_ids_sha256") != json_digest(pair_ids.tolist()) or
            status.get("episode_ids_sha256") != json_digest(episode_ids.tolist()) or
            status.get("target_positions_sha256") != array_sha256(positions) or
            status.get("B_sha256") != array_sha256(arrays["B"]) or
            status.get("evaluation_source_sha256") != source_hashes):
        raise ValueError("Direction-bound semantic hashes differ from the verified arrays/sources")
    receipt = {"status": "passed", "artifact_directory": str(root),
               "artifact_sha256": {name: sha256(root / name) for name in sorted(expected_files)},
               "target_positions_sha256": status["target_positions_sha256"],
               "B_sha256": status["B_sha256"], "pairs": 96, "episodes": 192,
               "prepared_finished_utc": status["finished_utc"],
               "training_or_checkpoint_or_decoder_selection_uses_bound": False}
    if reuse is not None:
        receipt["reused_original_preparation"] = reuse
        receipt["original_receipt_rewritten"] = False
    return arrays["true_positions"], arrays["B"], receipt


def require_cpu_preparation():
    """Reject a GPU Slurm allocation without importing any GPU library."""
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id:
        return {"slurm_job_id": None, "mode": "local_cpu"}
    if not re.fullmatch(r"[0-9]+", job_id):
        raise ValueError("Unrecognized native CPU preparation job ID")
    control = subprocess.run(["scontrol", "show", "job", "--oneliner", job_id], check=True,
                             text=True, capture_output=True, timeout=30).stdout.strip()
    fields = dict(item.split("=", 1) for item in control.split() if "=" in item)
    if fields.get("JobId") != job_id or fields.get("JobState") != "RUNNING" or not fields.get("AllocTRES"):
        raise ValueError("Cannot confirm native CPU preparation allocation")
    for item in fields["AllocTRES"].split(","):
        if "=" in item:
            key, value = item.split("=", 1)
            if (key == "gres/gpu" or key.startswith("gres/gpu:")) and int(value) != 0:
                raise ValueError("Direction bound must be prepared before allocating a GPU")
    return {"slurm_job_id": job_id, "mode": "native_cpu", "scontrol": control}


def execute(args):
    protocol = validate_protocol(args.protocol)
    allocation = require_cpu_preparation()
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Direction-bound output must be new; no overwrite or retry")
    output.mkdir(parents=True, exist_ok=False)
    own_relative = Path(__file__).resolve().relative_to(REPO).as_posix()
    state = {"status": "stopped_execution_error", "actual_a100_gpu_seconds": 0,
             "protocol_sha256": sha256(args.protocol), "manifest_sha256": MANIFEST_SHA256,
             "source_sha256": {own_relative: protocol["source_sha256"][own_relative]},
             "evaluation_split": "validation", "pairs": 96, "episodes": 192, "horizons": [3, 6],
             "test_opened_or_generated": False, "training_or_model_imported": False,
             "allocation": allocation, "python": sys.version}
    try:
        import numpy as np
        state["numpy"] = np.__version__
        if np.__version__ != protocol["runtime_package_versions"]["numpy"]:
            raise ValueError("CPU NumPy version differs from the frozen scientific protocol")
        if sha256(args.dataset / "manifest.json") != MANIFEST_SHA256:
            raise ValueError("Frozen dataset manifest SHA256 mismatch")
        manifest = read_json(args.dataset / "manifest.json")
        positions, pair_ids, episode_ids, source_hashes = read_target_positions(args.dataset, manifest)
        bound = midpoint_bound(positions)
        with (output / "direction_bound.npz").open("xb") as destination:
            np.savez_compressed(destination, true_positions=positions, B=bound, pair_ids=pair_ids,
                                episode_ids=episode_ids, horizons=np.asarray([3, 6]))
        state.update(status="passed", pair_ids_sha256=json_digest(pair_ids.tolist()),
                     episode_ids_sha256=json_digest(episode_ids.tolist()),
                     target_positions_sha256=array_sha256(positions), B_sha256=array_sha256(bound),
                     evaluation_source_sha256=source_hashes,
                     B_mean={str(h): float(bound[:, i].mean()) for i, h in enumerate((3, 6))})
    except Exception as exc:
        state.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    finally:
        state["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "status.json", state)
        write_json(output / "artifact_sha256.json",
                   {path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file()})
    print(json.dumps({"status": state["status"], "actual_a100_gpu_seconds": 0,
                      "output": str(output)}, allow_nan=False), flush=True)
    return 0 if state["status"] == "passed" else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return execute(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
