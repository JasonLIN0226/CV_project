"""Train the PushT latent predictor on a marker corpus and evaluate fixed memory conditions.

    python scripts/train_predictor/pusht.py --version v5c --data-root <dir> --out <dir> [--threads 16]

Reads <data-root>/<corpus> and its I-JEPA features <corpus>-ijepa (scripts/encode/ijepa_pusht.py),
trains seeds 17/29/43 (300 epochs, CPU), saves each seed's model and bases to
<corpus>-ijepa/predictors/seed_<n>.pt and writes <out>/<corpus>_predictor.json: error at +6/+12 and
sibling identification for recent-only, FIFO, first frame, a random pair, the marker frames
(privileged reference) and the best pair per episode (upper bound). Rerunning skips finished seeds
and reuses saved models. Implementation: src/bounded_memory/environments/pusht/train_pusht_cog_predictor.py.
"""
import argparse
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PUSHT = ROOT / "src/bounded_memory/environments/pusht"
sys.path.insert(0, str(PUSHT))
from corpora import VERSIONS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", required=True, choices=sorted(k for k in VERSIONS if k.startswith("v5")))
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    args, extra = ap.parse_known_args()
    corpus = VERSIONS[args.version]["corpus"]
    data, features = args.data_root / corpus, args.data_root / f"{corpus}-ijepa"
    script = PUSHT / "train_pusht_cog_predictor.py"
    sys.argv = [str(script), "--data", str(data), "--features", str(features),
                "--output", str(args.out / f"{corpus}_predictor.json"), "--save-dir", str(features / "predictors"),
                "--threads", str(args.threads), *[e for e in extra if e != "--"]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
