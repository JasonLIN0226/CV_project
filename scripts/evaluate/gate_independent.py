"""Evaluate both frozen Gate students on validation or the independent cohort.

    python scripts/evaluate/gate_independent.py independent --config configs/gate/team_bundle.yaml \
        --test-data <encoded-data> --out <new-directory>
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bounded_memory.environments.gate.holdout_protocol import COHORT_DIR  # noqa: E402
from bounded_memory.evaluation.gate_independent import evaluate  # noqa: E402
from bounded_memory.runs import load_config  # noqa: E402


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    modes = cli.add_subparsers(dest="mode", required=True)
    for name in ("validation", "independent"):
        p = modes.add_parser(name)
        p.add_argument("--config", type=Path, required=True, help="team Gate asset config")
        p.add_argument("--out", "--output", type=Path, required=True, help="new run directory")
        p.add_argument("--device", default="cpu")
        p.add_argument("--threads", type=int, default=1)
        if name == "independent":
            p.add_argument("--test-data", type=Path, required=True)
            p.add_argument(
                "--prior-manifest", nargs="+", type=Path, default=[COHORT_DIR / "prior_ids.csv.gz"]
            )
            p.add_argument("--provenance", type=Path, default=COHORT_DIR / "cohort.json")
    args = cli.parse_args()
    if args.out.exists():
        cli.error("output directory already exists")
    torch.set_num_threads(args.threads)
    summary = evaluate(args, load_config(args.config))
    print(json.dumps(summary["comparison"] if args.mode == "validation" else summary["inference"], indent=2))
    if summary.get("passed") is False:
        raise SystemExit("Gate validation differs from the released rows")


if __name__ == "__main__":
    main()
