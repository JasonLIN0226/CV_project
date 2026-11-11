"""DINOv2 class-token features for every frame of a Gate asset tree.

    python scripts/encode/dinov2.py --config configs/gate/team_bundle.yaml --device cuda \\
        --out <dir>/features.npz [--revision <commit of facebook/dinov2-base>]

Writes one array [frames, 768] per episode id, plus metadata.json with the resolved checkpoint
revision. Without --revision the current head of the model repository is resolved and recorded.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bounded_memory import runs  # noqa: E402
from bounded_memory.data.io import read_json, require_sha256  # noqa: E402
from bounded_memory.encoders import dinov2  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--revision")
    ap.add_argument("--compare", type=Path, help="existing features.npz to compare the new features with")
    args = ap.parse_args()
    if args.out.exists() and not args.compare:
        print(f"{args.out} exists, nothing to do")
        return
    if args.out.exists():
        return compare(args.out, args.compare)
    root = Path(runs.load_config(args.config)["assets"]["root"])
    revision = args.revision
    if revision is None:
        from huggingface_hub import model_info
        revision = model_info(dinov2.REPO).sha
    model = dinov2.load_model(revision, args.device)
    features = {}
    records = read_json(root / "dataset/manifest.json")["records"]
    for n, rec in enumerate(records):
        path = require_sha256(root / "dataset" / rec["obs"], rec["obs_sha256"], "dataset episode")
        with np.load(path, allow_pickle=False) as f:
            features[rec["id"]] = dinov2.encode_frames(model, f["rgb"], args.device, args.batch)
        if n % 100 == 0:
            print(f"encoded {n}/{len(records)}", flush=True)
    if not all(np.isfinite(v).all() for v in features.values()):
        raise ValueError("nonfinite DINOv2 features")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".partial.npz")
    np.savez(tmp, **features)
    tmp.replace(args.out)
    (args.out.parent / "metadata.json").write_text(json.dumps(
        {"repository": dinov2.REPO, "revision": revision, "feature": "final class token", "dim": 768,
         "preprocessing": "resize to 224x224, no centre crop, ImageNet normalisation",
         "episodes": len(features), "frames": int(sum(len(v) for v in features.values()))}, indent=2) + "\n")
    print(f"wrote {args.out}")
    if args.compare:
        compare(args.out, args.compare)


def compare(path, other):
    with np.load(path) as a, np.load(other) as b:
        if sorted(a.files) != sorted(b.files):
            raise ValueError("the two feature files cover different episodes")
        worst = max(float(np.abs(a[k] - b[k]).max()) for k in a.files)
        identical = sum(np.array_equal(a[k], b[k]) for k in a.files)
        report = {"episodes": len(a.files), "identical_episodes": int(identical), "max_abs_diff": worst}
    (path.parent / "comparison.json").write_text(json.dumps({"against": str(other), **report}, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
