"""Regenerate evidence-frame RGB with the frozen generator and compute the
appearance-only features exactly as run_terminal_structures.appearance_features.
Every regenerated frame is checked against the archived rgb_index (sha256 -> id)."""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from path_config import REPO, RECORDS, RGB_INDEX, APPEARANCE_OUTPUT
sys.path[:0] = [str(REPO), str(REPO / "experiments/predictor/control/src")]
from experiments.phase2 import terminal_structures as env  # noqa: E402

recs = json.load(open(RECORDS))
lookup = json.load(open(RGB_INDEX))
feat_by_frame, mism, t0 = {}, 0, time.time()
out = {}
for n, r in enumerate(recs):
    need = [r["rgb_indices"][t] for t in r["evidence"]]
    if not all(i in feat_by_frame for i in need):
        ep = env.episode(r)
        for t in r["evidence"]:
            im8 = np.asarray(ep["rgb"][t])
            if lookup.get(hashlib.sha256(im8.tobytes()).hexdigest()) != r["rgb_indices"][t]:
                mism += 1
            im = im8.astype(np.float32) / 255
            hist = np.concatenate([np.histogram(im[:, :, j], bins=8, range=(0, 1), density=False)[0] / (96 * 96) for j in range(3)])
            feat_by_frame[r["rgb_indices"][t]] = np.r_[hist, im.mean((0, 1)), im.std((0, 1))].astype(np.float32)
    out[r["id"]] = np.stack([feat_by_frame[i] for i in need])
    if n % 1000 == 0:
        print(json.dumps({"done": n, "total": len(recs), "sec": round(time.time() - t0, 1), "mismatch": mism}), flush=True)
ids = [r["id"] for r in recs]
np.savez_compressed(APPEARANCE_OUTPUT, ids=np.array(ids), items=np.stack([out[i] for i in ids]))
print(json.dumps({"saved": len(ids), "unique_frames": len(feat_by_frame), "hash_mismatches": mism, "sec": round(time.time() - t0, 1)}))
