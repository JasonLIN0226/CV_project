"""DINOv2 ViT-B/14 class-token features for every Gate frame (FrameMorrow's visual encoder and teacher).

    python -m experiments.memory_baselines.encode_dinov2 --config <resolved_config.json> \
        --snapshot <clean_future_control/src> --out <dir>/features.npz

Writes one array [frames, 768] per episode id, plus metadata.json with the resolved checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import gate_data as gd

REPO = "facebook/dinov2-base"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)   # the checkpoint's own normalisation constants
STD = np.array([0.229, 0.224, 0.225], np.float32)


def preprocess(rgb):
    """Whole 96x96 frame -> 224x224, bicubic. The stock processor centre-crops, which would cut into the cue panel."""
    from PIL import Image
    x = np.stack([np.asarray(Image.fromarray(f).resize((224, 224), Image.BICUBIC), np.float32) / 255.0 for f in rgb])
    return torch.from_numpy(np.ascontiguousarray(((x - MEAN) / STD).transpose(0, 3, 1, 2)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True); ap.add_argument("--snapshot", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        print(f"{out} exists, nothing to do")
        return
    gd.add_paths(args.snapshot)
    from huggingface_hub import model_info
    from phase1.data import load_episode, require_audit
    from transformers import AutoModel

    c = json.loads(Path(args.config).read_text())
    revision = model_info(REPO).sha
    model = AutoModel.from_pretrained(REPO, revision=revision).to(args.device).eval()
    features = {}
    records = require_audit(c)["records"]
    with torch.inference_mode():
        for n, rec in enumerate(records):
            rgb = load_episode(c, rec)["rgb"]
            chunks = []
            for i in range(0, len(rgb), args.batch):
                pixels = preprocess(rgb[i:i + args.batch]).to(args.device)
                chunks.append(model(pixel_values=pixels).last_hidden_state[:, 0].float().cpu().numpy())
            features[rec["id"]] = np.concatenate(chunks).astype(np.float32)
            if n % 100 == 0:
                print(f"encoded {n}/{len(records)}", flush=True)
    if not all(np.isfinite(v).all() for v in features.values()):
        raise ValueError("nonfinite DINOv2 features")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".partial.npz")
    np.savez(tmp, **features)
    tmp.replace(out)
    (out.parent / "metadata.json").write_text(json.dumps(
        {"repository": REPO, "revision": revision, "feature": "final class token", "dim": 768,
         "preprocessing": "resize to 224x224, no centre crop, ImageNet normalisation",
         "episodes": len(features), "frames": int(sum(len(v) for v in features.values()))}, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
