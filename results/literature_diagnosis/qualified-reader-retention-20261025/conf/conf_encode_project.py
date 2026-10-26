"""One-shot CONF (seed 20261104, 48 families): generate, encode on A100, project to the
frozen TRAIN basis, and pack a small transfer bundle. Runs on NYU burst HPC inside the
existing terminal-structures bundle environment. Does not touch any reader, student or
threshold; evaluation happens afterwards on CPU with the frozen artifacts.

Usage (inside the sbatch allocation):
  $BASE/venv/bin/python conf_encode_project.py --base /scratch/xy3257/cv-phase2-20261020
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    a = ap.parse_args()
    base = a.base
    bundle = base / "terminal-structures-20261025/bundle"
    run = base / "terminal-structures-20261025/run"
    out = base / "qualified-reader-conf-20261026"
    if out.exists() and any(out.iterdir()):
        raise SystemExit("Refuse to overwrite: CONF is one-shot")
    out.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(bundle), str(bundle / "experiments/predictor/control/src")]
    import torch
    from experiments.phase2 import run_terminal_structures as rts

    receipt = {"utc_start": datetime.now(timezone.utc).isoformat(), "torch": torch.__version__,
               "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               "nvidia_smi": subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,memory.total", "--format=csv,noheader"],
                                            capture_output=True, text=True).stdout}
    if not receipt["cuda_device"] or "A100" not in receipt["cuda_device"]:
        raise SystemExit("Expected an allocated A100")
    c = rts.locked(run)                                    # verifies frozen bundle source hashes
    assert c["splits"]["confirmation"] == {"seed": 20261104, "families": 48}
    banks = dict(np.load(run / "data/action_banks.npz", allow_pickle=False))
    data = out / "confirmation_data"
    t0 = time.time()
    rts.generate(c, data, ["confirmation"], existing_banks=banks)
    receipt["generate_seconds"] = time.time() - t0
    t0 = time.time()
    rts.encode(c, data, base / "cache/ijepa", "cuda", run / "data/normalization.npz")
    receipt["encode_seconds"] = time.time() - t0

    b = np.load(run / "basis.npz")
    im, P, tm, B = b["input_mean"], b["projection"], b["target_mean"], b["output_basis"]
    F = np.load(data / "features.npy", mmap_mode="r")
    proj = np.empty((F.shape[0], 128), np.float32)
    fb = np.empty((F.shape[0], 128), np.float64)
    for s in range(0, F.shape[0], 2048):
        x = np.asarray(F[s:s + 2048], np.float32)
        proj[s:s + 2048] = ((x - im) @ P).astype(np.float32)
        fb[s:s + 2048] = x.astype(np.float64) @ B.astype(np.float64)
    tr = out / "transfer"
    tr.mkdir()
    np.save(tr / "projected_features.npy", proj)
    np.save(tr / "output_basis_coords.npy", fb.astype(np.float32))
    np.save(tr / "target_mean_coords.npy", tm.astype(np.float64) @ B.astype(np.float64))
    for name in ("records.json", "actions.npz", "rgb_index.json", "generation.json", "encoding.json"):
        shutil.copy2(data / name, tr / name)
    receipt.update({"utc_end": datetime.now(timezone.utc).isoformat(), "unique_frames": int(F.shape[0]),
                    "features_sha256": sha(data / "features.npy"), "basis_sha256": sha(run / "basis.npz"),
                    "normalization_sha256": sha(run / "data/normalization.npz"),
                    "transfer_sha256": {p.name: sha(p) for p in sorted(tr.iterdir())}})
    (tr / "receipt.json").write_text(json.dumps(receipt, indent=2))
    with zipfile.ZipFile(out / "conf_transfer.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(tr.iterdir()):
            z.write(p, p.name)
    print(json.dumps({"status": "completed", "zip": str(out / "conf_transfer.zip"),
                      "zip_sha256": sha(out / "conf_transfer.zip"), **{k: receipt[k] for k in ("unique_frames", "encode_seconds")}}))


if __name__ == "__main__":
    main()
