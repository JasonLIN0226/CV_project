"""Rebuild the Synergy (terminal R/D/S) train/validation/development features.

    python -m experiments.memory_baselines.synergy_assets --release <unzipped run/> --work <dir>

The frozen readers, basis, TRAIN normalisation and the encoded confirmation set are public
release assets of the team repo. The encoded TRAIN/VAL/DEV features were not published, so
this regenerates the frames with the repo generator (seeds in the frozen protocol), encodes
them with the pinned I-JEPA, applies the frozen TRAIN normalisation and projects onto the
frozen basis. Output layout matches what results/.../qualified-reader-retention-20261025/code/pipe.py reads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ["train", "validation", "development"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", type=Path, required=True, help="terminal-structures release, the extracted run/ folder")
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--hf-cache", type=Path, required=True, help="Hugging Face hub cache holding the pinned I-JEPA")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    sys.path[:0] = [str(ROOT), str(ROOT / "experiments/predictor/control/src")]
    from experiments.phase2 import run_terminal_structures as rts

    run, data, xfer = args.release, args.work / "data", args.work / "xfer"
    protocol = json.loads((run / "protocol.json").read_text())
    c = protocol["config"]
    changed = sorted(k for k, v in rts.source_files().items() if protocol["source_sha256"].get(k) != v)
    report = {"source_files_changed_since_freeze": changed}
    banks = dict(np.load(run / "data/action_banks.npz", allow_pickle=False))

    if not (data / "records.json").exists():
        rts.generate(c, data, SPLITS, existing_banks=banks)
    new = json.loads((data / "records.json").read_text())
    old = json.loads((run / "data/records.json").read_text())
    keys = ("id", "split", "family", "task_family", "condition", "query", "rgb_indices", "plan_targets")
    same = len(new) == len(old) and all(all(a.get(k) == b.get(k) for k in keys) for a, b in zip(new, old))
    report.update(records=len(new), records_match_release=bool(same),
                  unique_frames=json.loads((data / "generation.json").read_text())["unique_rgb"])
    if not same:
        raise SystemExit("regenerated records differ from the released records; refusing to mix them: "
                         + json.dumps(report))

    if not (data / "features.npy").exists():
        rts.encode(c, data, args.hf_cache, args.device, run / "data/normalization.npz")

    b = np.load(run / "basis.npz")
    im, P, tm, B = b["input_mean"], b["projection"], b["target_mean"], b["output_basis"]
    F = np.load(data / "features.npy", mmap_mode="r")
    proj = np.empty((F.shape[0], P.shape[1]), np.float32)
    fb = np.empty((F.shape[0], B.shape[1]), np.float64)
    for s in range(0, F.shape[0], 2048):
        x = np.asarray(F[s:s + 2048], np.float32)
        proj[s:s + 2048] = ((x - im) @ P).astype(np.float32)
        fb[s:s + 2048] = x.astype(np.float64) @ B.astype(np.float64)
    xfer.mkdir(parents=True, exist_ok=True)
    np.save(xfer / "projected_features.npy", proj)
    np.save(xfer / "output_basis_coords.npy", fb)
    np.save(xfer / "target_mean_coords.npy", tm.astype(np.float64) @ B.astype(np.float64))

    # The release keeps the reader's coefficients on DEV; reproducing them checks the re-encoded features.
    report["features_sha256"] = hashlib.sha256((data / "features.npy").read_bytes()).hexdigest()
    (args.work / "assets_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "SYNERGY_ASSETS_READY", **report}))


if __name__ == "__main__":
    main()
