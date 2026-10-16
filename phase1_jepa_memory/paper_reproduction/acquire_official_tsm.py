"""Acquire and audit the author-prescribed TSM input, never train a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time
import zipfile

COURSE = "csci_ga_2271-2026fa"
PYTHON = "/scratch/xl4358/phase1_env/bin/python"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    tmp.replace(path)


def guard() -> dict:
    job = os.environ.get("SLURM_JOB_ID", "")
    if not job.isdigit() or sys.executable != PYTHON:
        raise RuntimeError("actual course batch/Python required")
    text = subprocess.check_output(
        ["/opt/slurm/bin/scontrol", "show", "job", "-o", job], text=True
    )
    fields = dict(x.split("=", 1) for x in text.split() if "=" in x)
    if fields.get("Account") != COURSE or fields.get("JobState") != "RUNNING":
        raise RuntimeError("unexpected course/state")
    if "gres/gpu" in fields.get("AllocTRES", ""):
        raise RuntimeError("data stage must have zero GPUs")
    return {"job_id": int(job), "account": fields["Account"],
            "partition": fields.get("Partition"),
            "allocated_tres": fields.get("AllocTRES")}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    resources = guard()
    lock = json.loads((args.root / "source_lock.json").read_text())
    for name, expected in lock["files"].items():
        path = args.root / name
        if path.is_symlink() or sha(path) != expected:
            raise RuntimeError("source hash mismatch")
    metadata = json.loads((args.root / "official_data_metadata.json").read_text())
    file = next(x for x in metadata["files"] if x["rfilename"] == "TSM_features/C10095_rgb.zip")
    archive_root = Path("/scratch/xl4358/phase1_space_official_data")
    archive_root.mkdir(exist_ok=True)
    if shutil.disk_usage(archive_root).free < int(file["size"] * 2.8):
        raise RuntimeError("insufficient scratch space for archive and LMDB")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_VERBOSITY"] = "error"
    from huggingface_hub import hf_hub_download
    archive = Path(hf_hub_download(
        repo_id=metadata["repo_id"], repo_type="dataset",
        filename=file["rfilename"], revision=metadata["revision"],
        local_dir=str(archive_root / "download"),
    ))
    digest = sha(archive)
    if archive.stat().st_size != file["size"] or digest != file["lfs"]["sha256"]:
        raise RuntimeError("official archive size/hash mismatch")
    extracted = archive_root / "lmdb"
    extracted.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        inventory = []
        for member in z.infolist():
            name = Path(member.filename)
            if name.is_absolute() or ".." in name.parts or stat.S_ISLNK(member.external_attr >> 16):
                raise RuntimeError("unsafe archive member")
            inventory.append({"name": member.filename, "bytes": member.file_size,
                              "compressed_bytes": member.compress_size, "crc32": member.CRC})
        data_members = [m for m in z.infolist() if Path(m.filename).name == "data.mdb"]
        if len(data_members) != 1:
            raise RuntimeError("expected exactly one official LMDB data file")
        needed = sum(m.file_size for m in z.infolist() if not m.is_dir())
        if shutil.disk_usage(extracted).free < needed + 4 * 1024**3:
            raise RuntimeError("insufficient space for official uncompressed LMDB")
        data_member = data_members[0]
        target = extracted / data_member.filename
        if target.exists():
            raise RuntimeError("refuse ambiguous preexisting LMDB; verify before reusing")
        z.extractall(extracted)
    write(args.output / "archive_integrity.json", {
        **resources, "revision": metadata["revision"], "sha256": digest,
        "bytes": archive.stat().st_size, "members": inventory,
        "archive_is_opaque_acquisition_not_test_feature_evaluation": True,
    })
    manifest_root = archive_root / "latents_train_val"
    if manifest_root.exists():
        raise RuntimeError("refuse preexisting prepared features")
    subprocess.run([
        PYTHON, str(args.root / "upstream/scripts/prepare_assembly101_tsm.py"),
        "--lmdb", str(target.parent), "--output-dir", str(manifest_root),
        "--splits", "train", "val",
    ], check=True)
    import torch
    from collections import Counter
    manifest = json.loads((manifest_root / "manifest.json").read_text())
    if manifest.get("test_features_accessed") or set(manifest["requested_splits"]) != {"train", "val"}:
        raise RuntimeError("test feature isolation failed")
    feature_proofs = []
    participants = {"train": set(), "val": set()}
    for row in manifest["videos"]:
        path = manifest_root / row["path"]
        before = sha(path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        features, indices = payload["latents"], payload["clip_indices"]
        if features.dtype != torch.float32 or tuple(features.shape) != (row["valid_length"], 2048):
            raise RuntimeError("official feature shape/dtype mismatch")
        if not torch.isfinite(features).all() or not torch.allclose(features.norm(dim=1), torch.ones(len(features)), atol=1e-5):
            raise RuntimeError("official normalized features invalid")
        if indices.dtype != torch.int64 or not bool((indices[1:] - indices[:-1] == 15).all()):
            raise RuntimeError("official indexed cadence is not contiguous stride fifteen")
        participant = payload["metadata"]["participant_id"]
        participants[row["split"]].add(participant)
        if sha(path) != before:
            raise RuntimeError("prepared feature changed during audit")
        feature_proofs.append({"video_id": row["video_id"], "split": row["split"],
                               "participant": participant, "frames": len(features),
                               "first_frame_index": int(indices[0]), "last_frame_index": int(indices[-1]),
                               "bytes": path.stat().st_size, "sha256": before})
    if len(participants["train"]) != 29 or len(participants["val"]) != 10 or participants["train"] & participants["val"]:
        raise RuntimeError("paper participant split mismatch")
    report = {**resources, "status": "OFFICIAL_TSM_TRAIN_VAL_PREPARED_AND_AUDITED",
              "model_inference_fit_training": False, "test_payload_deserialized": False,
              "manifest": str(manifest_root / "manifest.json"),
              "manifest_sha256": sha(manifest_root / "manifest.json"),
              "counts": dict(Counter(x["split"] for x in feature_proofs)),
              "participants": {k: sorted(v) for k, v in participants.items()},
              "features": feature_proofs, "elapsed_seconds": time.monotonic() - started,
              "annotation_join_proved": False}
    write(args.output / "data_report.json", report)
    print(json.dumps({"status": report["status"], "counts": report["counts"]}), flush=True)


if __name__ == "__main__":
    main()
