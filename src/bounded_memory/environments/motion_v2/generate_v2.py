"""One-shot, train/validation-only Motion v2.1 dataset writer.

Run only after the step protocol and source hashes have been committed.
The output root must not exist; partial attempts are preserved, never reused.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np


# Support the registered ``python -I path/to/generate_v2.py`` invocation without
# importing the original Phase 1 package or placing repository modules on sys.path.
_SPEC = importlib.util.spec_from_file_location("motion_v2_envs", Path(__file__).with_name("envs_v2.py"))
envs_v2 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(envs_v2)

OBSERVATION_KEYS = frozenset({"rgb", "actions", "query"})
EVALUATION_KEYS = frozenset({
    "states", "velocity", "classes", "visible", "sufficient", "label", "evidence",
    "t_on", "t_c", "end", "L0", "L1", "v0", "v1", "speed", "x_end", "x_pre",
})


def file_hash(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def write_json_new(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def write_npz_new(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)


def generate(config, output):
    """Generate exactly 512 train and 96 validation pairs under output/dataset.

    No manifest/episode from any prior run is opened. No environment variable,
    config setting or CLI flag can expand this entry point to test splits.
    """
    envs_v2.validate_config(config)
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Output root already exists; no overwrite or implicit retry")
    code_paths = (Path(__file__).with_name("envs_v2.py"), Path(__file__))
    source_hashes = {path.name: file_hash(path) for path in code_paths}
    code_hash = envs_v2.digest(source_hashes)
    signature = envs_v2.digest({
        "data": config["data"], "generator_version": envs_v2.GENERATOR_VERSION,
        "implementation": code_hash,
    })
    # mkdir itself also refuses a race with another writer. The output is never
    # removed, including on interruption or an error partway through generation.
    output.mkdir(parents=True, exist_ok=False)
    root = output / "dataset"
    root.mkdir()
    resolved = dict(config)
    resolved["output"] = str(output)
    write_json_new(output / "resolved_config.json", resolved)
    began = time.perf_counter()
    records = []
    seen_pairs, seen_ids, seen_scene_seeds = set(), set(), set()
    for split in envs_v2.ALLOWED_SPLITS:
        for index in range(config["data"]["pairs"][split]):
            shared = envs_v2.scene(config, "motion", split, index)
            pair = envs_v2.digest([config["data"]["seed"], "motion", split, index])[:24]
            ids = [envs_v2.digest([pair, slot, "opaque-id"])[:24] for slot in range(2)]
            if pair in seen_pairs or shared["seed"] in seen_scene_seeds or any(eid in seen_ids for eid in ids):
                raise ValueError("Pair/episode/scene-seed collision; do not regenerate")
            seen_pairs.add(pair)
            seen_ids.update(ids)
            seen_scene_seeds.add(shared["seed"])
            for slot, eid in enumerate(ids):
                value = envs_v2.episode(shared, shared["label_order"][slot])
                observed, evaluation = value["observations"], value["evaluation_only"]
                if set(observed) != OBSERVATION_KEYS or set(evaluation) != EVALUATION_KEYS:
                    raise ValueError("Observation/evaluator schema changed")
                obs = root / "observations" / split / f"{eid}.npz"
                meta = root / "evaluation_only" / split / f"{eid}.npz"
                write_npz_new(obs, observed)
                write_npz_new(meta, evaluation)
                # Deliberately preserve only Phase 1's public record fields.
                # No timing landmarks, visibility, positions or labels are here.
                records.append({
                    "id": eid, "pair": pair, "sibling": ids[1-slot], "split": split,
                    "task": "motion", "seed": shared["seed"], "scene_index": index,
                    "query": shared["query"], "gap": shared["gap"], "frames": shared["length"],
                    "obs": str(obs.relative_to(root)), "eval": str(meta.relative_to(root)),
                    "obs_sha256": file_hash(obs), "eval_sha256": file_hash(meta),
                })
    manifest = {
        "signature": signature, "config_hash": envs_v2.digest(config["data"]),
        "code_hash": code_hash, "source_hashes": source_hashes,
        "generator_version": envs_v2.GENERATOR_VERSION,
        "mode": "train_validation_only", "generation_seconds": time.perf_counter() - began,
        "records": records,
    }
    write_json_new(root / "manifest.json", manifest)
    write_json_new(output / "generation_receipt.json", {
        "status": "generated_awaiting_registered_audit",
        "manifest_sha256": file_hash(root / "manifest.json"),
        "signature": signature, "source_hashes": source_hashes,
        "pairs": dict(config["data"]["pairs"]), "episodes": len(records),
        "test_generated_or_opened": False,
    })
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    with args.config.open(encoding="utf-8") as handle:
        config = json.load(handle)
    manifest = generate(config, args.output)
    print(json.dumps({
        "status": "generated_awaiting_registered_audit", "episodes": len(manifest["records"]),
        "manifest": str(args.output.absolute() / "dataset" / "manifest.json"),
        "manifest_sha256": file_hash(args.output / "dataset" / "manifest.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
