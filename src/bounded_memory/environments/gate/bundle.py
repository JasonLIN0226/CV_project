"""Fetch the team's frozen Gate bundles and lay them out as a Gate asset tree.

    python scripts/prepare_data/gate_team_bundle.py --work <dir>

Moved from experiments/memory_baselines/gate_bundle.py; only the names of the pinned checksums
changed.

Needs read access to the private org `cv-project-team5guys` (token in HF_HOME). Downloads the
data bundle (1,216 episodes, I-JEPA feature cache, original validation output) and the model
bundle (three predictors, the six frozen scorers, transforms, probes), verifies every published
checksum, and builds <work>/root (see environments/gate/assets.py) with the team's exact assets:

    root/dataset, root/features      from the data bundle
    root/features/normalization.*    from the model bundle
    root/predictor, root/probes      from the model bundle
    root/resolved_config.json        the team's config with only output/device/threads changed
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import tarfile
from pathlib import Path

ORG = "cv-project-team5guys"
DATA = (f"{ORG}/gate-clean-future-trainval-v1", "dataset", "gate_trainval.tar")
MODEL = (f"{ORG}/gate-retention-models-v1", "model", "gate_models.tar")
# Pinned in the team's tools/load_predictor.py; checked before any checkpoint is unpickled.
TEAM_PREDICTOR_SHA256 = {
    17: "449bb70e55161ff553ce4b2e75e9e093a58e1914f268bbe3748442f0315f9ed0",
    29: "f7301b06df9e62fede69268c53e1bc0f08de027659fe9f15f561756faba8d3c2",
    43: "96311a7cf010c527ae8f6c29a390a8275d0b402718dd57928395f832b07e7782",
}
TEAM_BASIS_SHA256 = "1828bd01d83a0c9b1e504739d3bd864bb9469ff3fb058292c895ca198dd0da8d"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(repo, repo_type, tar_name, dest):
    from huggingface_hub import hf_hub_download

    dest.mkdir(parents=True, exist_ok=True)
    for name in (tar_name, "MANIFEST.json", "SHA256SUMS"):
        hf_hub_download(repo, name, repo_type=repo_type, local_dir=dest)
    for line in (dest / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split()
        if sha256(dest / name) != expected:
            raise ValueError(f"checksum mismatch for {name}")
    return dest / tar_name


def link(target, name):
    if name.is_symlink() or name.exists():
        if name.is_symlink() and name.resolve() == target.resolve():
            return
        raise FileExistsError(f"{name} already exists and points elsewhere")
    name.symlink_to(target.resolve(), target_is_directory=target.is_dir())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--device", default="cpu"); ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    work = args.work.resolve()
    extracted, root = work / "extracted", work / "root"
    report = {}
    for repo, kind, tar_name in (DATA, MODEL):
        tar_path = fetch(repo, "dataset" if kind == "dataset" else "model", tar_name, work / "download" / kind)
        if not (extracted / kind).is_dir():
            with tarfile.open(tar_path) as tar:
                tar.extractall(extracted, filter="data")
        manifest = json.loads((work / "download" / kind / "MANIFEST.json").read_text())
        for entry in manifest["files"]:
            path = extracted / kind / entry["relative_path"]
            if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
                raise ValueError(f"extracted file differs from the bundle manifest: {path}")
        report[kind] = {"repository": repo, "tar_sha256": sha256(tar_path), "files_verified": len(manifest["files"])}

    data, model = extracted / "dataset", extracted / "model"
    spec = importlib.util.spec_from_file_location("team_verify_dataset", data / "tools/verify_dataset.py")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    report["dataset_verification"] = verifier.verify(data)
    for seed, expected in TEAM_PREDICTOR_SHA256.items():
        if sha256(model / f"predictor/gate/seed_{seed}/best.pt") != expected:
            raise ValueError(f"predictor seed {seed} differs from the pinned frozen checksum")
    if sha256(model / "predictor/gate/basis.npz") != TEAM_BASIS_SHA256:
        raise ValueError("basis differs from the pinned frozen checksum")

    (root / "features").mkdir(parents=True, exist_ok=True)
    link(data / "dataset", root / "dataset")
    for name in ("index.json", "pair_feature_audit.json", "train", "validation"):
        link(data / "features" / name, root / "features" / name)
    for name in ("normalization.npz", "normalization.json"):
        link(model / "features" / name, root / "features" / name)
    link(model / "predictor", root / "predictor")
    link(model / "probes", root / "probes")
    config = json.loads((model / "resolved_config.json").read_text())
    report["original_output_path"] = config.get("output")
    config.update(output=str(root), device=args.device, threads=args.threads)
    (root / "resolved_config.json").write_text(json.dumps(config, indent=2) + "\n")
    report.update(root=str(root), students=str(model), original_validation=str(data / "validation_results.json"))
    (work / "bundle_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "GATE_TEAM_BUNDLE_READY", **report}, indent=2))


if __name__ == "__main__":
    main()
