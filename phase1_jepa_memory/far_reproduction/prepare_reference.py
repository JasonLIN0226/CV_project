"""Prepare fixed public FAR releases on course Cloud; never run a model."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "upstream"
OUT = ROOT / "preparation" / ("job_" + os.environ["SLURM_JOB_ID"])
OUT.mkdir(parents=True, exist_ok=True)
START = time.monotonic()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(16 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def save(name, value):
    p = OUT / name
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(p)


def check_source():
    lock = json.loads((ROOT / "upstream_source_lock.json").read_text())
    for name, expected in lock["files"].items():
        if sha(UPSTREAM / name) != expected:
            raise RuntimeError("upstream source changed: " + name)
    return sha(ROOT / "upstream_source_lock.json")


def verified(path, entry):
    p = Path(path)
    if p.stat().st_size != entry["size_bytes"] or sha(p) != entry["sha256"]:
        raise RuntimeError("public release checksum mismatch: " + str(p))


def main():
    if os.environ.get("SLURM_JOB_ACCOUNT") != "csci_ga_2271-2026fa":
        raise RuntimeError("course allocation required")
    if Path(sys.executable).resolve() != Path("/scratch/xl4358/phase1_env/bin/python").resolve():
        raise RuntimeError("approved interpreter required")
    source_before = check_source()
    release = json.loads((ROOT / "release_identity.json").read_text())
    if shutil.disk_usage(ROOT).free < 220 * 1024**3:
        raise RuntimeError("need at least 220 GiB free before fixed test release")
    from huggingface_hub import hf_hub_download
    models = UPSTREAM / "models"
    models.mkdir(exist_ok=True)
    listing = json.loads((ROOT / "bundles.json").read_text())
    selected = [b for b in listing["bundles"] if b.get("corpus") == "loopnav"
                or b["path"] == "oasis_500m_vit_vae.pth"]
    progress = {"status": "DOWNLOADING_PUBLIC_RELEASES", "model_files": 0,
                "test_shards": 0, "elapsed_seconds": 0}
    save("progress.json", progress)
    for entry in selected:
        p = hf_hub_download(release["models"]["repo"], entry["path"],
                            revision=release["models"]["sha"], local_dir=models,
                            token=False)
        verified(p, entry)
        if entry.get("corpus"):
            hf_hub_download(release["models"]["repo"],
                            f"loopnav/{entry['arm']}/config.yaml",
                            revision=release["models"]["sha"], local_dir=models,
                            token=False)
        progress["model_files"] += 1
        progress["elapsed_seconds"] = time.monotonic() - START
        save("progress.json", progress)
    datasets = json.loads((ROOT / "datasets.json").read_text())
    for entry in datasets["files"]:
        dest = UPSTREAM / entry["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        src = ROOT / "release_metadata" / entry["path"]
        verified(src, entry)
        shutil.copyfile(src, dest)
    downloads = ROOT / "public_release_downloads"
    downloads.mkdir(exist_ok=True)
    shards = [s for s in datasets["shards"] if s["tier"] == "test"]
    extracted = []
    for entry in shards:
        p = Path(hf_hub_download(release["datasets"]["repo"], entry["path"],
                                revision=release["datasets"]["sha"],
                                repo_type="dataset", local_dir=downloads,
                                token=False))
        verified(p, entry)
        count = 0
        with tarfile.open(p) as tar:
            for member in tar:
                rel = Path(member.name)
                if rel.is_absolute() or ".." in rel.parts or member.issym() or member.islnk():
                    raise RuntimeError("unexpected public archive member")
                if member.isfile() and not str(rel).startswith("datasets/loopnav/latent/"):
                    raise RuntimeError("archive outside declared corpus")
                tar.extract(member, path=UPSTREAM, filter="data")
                count += int(member.isfile())
        extracted.append({"path": entry["path"], "sha256": entry["sha256"],
                          "file_members": count, "episodes": entry["episodes"]})
        progress["test_shards"] += 1
        progress["elapsed_seconds"] = time.monotonic() - START
        save("progress.json", progress)
    # Install into a new target. The shared phase1_env is never modified.
    vendor = ROOT / "vendor"
    if vendor.exists():
        raise RuntimeError("vendor already exists; inspect prior installation rather than overwrite")
    req = UPSTREAM / "requirements.txt"
    log = OUT / "dependency_install.log"
    with log.open("w") as f:
        subprocess.run([sys.executable, "-m", "pip", "install", "--target", str(vendor),
                        "--only-binary=:all:", "--no-cache-dir", "-r", str(req)],
                       stdout=f, stderr=subprocess.STDOUT, check=True)
    source_after = check_source()
    report = {"status": "PUBLIC_RELEASE_FILES_AND_DEPENDENCIES_PREPARED",
              "job_id": int(os.environ["SLURM_JOB_ID"]), "release": release,
              "upstream_source_sha256_before": source_before,
              "upstream_source_sha256_after": source_after,
              "reference_protocol_sha256": sha(ROOT / "reference_protocol.json"),
              "model_files_verified": [{"path": e["path"], "sha256": e["sha256"],
                                         "size_bytes": e["size_bytes"]} for e in selected],
              "test_shards_verified": extracted,
              "test_clips_declared": sum(e["episodes"] for e in shards),
              "vendor": str(vendor), "requirements_sha256": sha(req),
              "elapsed_seconds": time.monotonic() - START,
              "model_inference_or_training": False,
              "data_numeric_audit_and_GPU_compatibility_pending": True}
    save("preparation_report.json", report)
    save("progress.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        save("failure.json", {"status": "PREPARATION_INCOMPLETE", "error": str(e),
                              "type": type(e).__name__, "model_inference_or_training": False,
                              "elapsed_seconds": time.monotonic() - START})
        raise
