"""Regenerate and re-encode the Synergy TRAIN/VAL/DEV frames, which were never released in encoded form.

    python scripts/prepare_data/synergy_trainvaldev.py --release <work>/terminal/run --work <work>/trainvaldev \
        --hf-cache $HF_HOME/hub --device cpu

The frozen readers, basis, TRAIN normalisation and the encoded confirmation set are public release
assets. This regenerates the frames with the package generator (environments/synergy/generate.py,
through dataset.generate, with the seeds of the frozen protocol and the released action banks),
requires the records to come out identical to the released ones, encodes the frames with the
pinned I-JEPA, applies the frozen TRAIN normalisation and projects onto the frozen basis.

Writes <work>/data (records, frames, features), <work>/xfer (projected features, the layout
environments/synergy/assets.py reads) and <work>/assets_report.json.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from bounded_memory.environments.synergy import dataset  # noqa: E402

SPLITS = ["train", "validation", "development"]
# file recorded in the frozen protocol -> where the same source now lives (None: same bytes; 1: one header line added)
FROZEN_SOURCES = {
    "experiments/phase2/terminal_structures.py": ("src/bounded_memory/environments/synergy/generate.py", 1),
    "experiments/phase2/fit_terminal_students.py": ("src/bounded_memory/methods/deletion_cost/terminal_students.py", None),
    **{f"experiments/predictor/control/src/phase1/{name}": (f"src/bounded_memory/predictors/phase1/{name}", None)
       for name in ("model.py", "inputs.py", "utils.py", "config.py", "encoders.py")},
}


def frozen_sources(protocol):
    """Which of the sources the frozen protocol recorded are the same bytes in the package."""
    out = {}
    for name, (path, skip) in FROZEN_SOURCES.items():
        data = (ROOT / path).read_bytes()
        if skip:
            data = data.split(b"\n", skip)[skip]
        out[path] = hashlib.sha256(data).hexdigest() == protocol["source_sha256"][name]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", type=Path, required=True, help="terminal-structures release, the extracted run/ folder")
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--hf-cache", type=Path, required=True, help="Hugging Face hub cache holding the pinned I-JEPA")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    run, data, xfer = args.release, args.work / "data", args.work / "xfer"
    protocol = json.loads((run / "protocol.json").read_text())
    c = protocol["config"]
    same_source = frozen_sources(protocol)
    report = {"frozen_sources_same_as_protocol": same_source}
    if not all(same_source.values()):
        raise SystemExit("package source differs from the frozen protocol: " + json.dumps(same_source))
    banks = dict(np.load(run / "data/action_banks.npz", allow_pickle=False))

    if not (data / "records.json").exists():
        dataset.generate(c, data, SPLITS, existing_banks=banks)
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
        dataset.encode(c, data, args.hf_cache, args.device, run / "data/normalization.npz")

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

    report["features_sha256"] = hashlib.sha256((data / "features.npy").read_bytes()).hexdigest()
    (args.work / "assets_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "SYNERGY_ASSETS_READY", **report}))


if __name__ == "__main__":
    main()
