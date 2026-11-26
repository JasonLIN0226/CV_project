"""Prepare and verify the independent Gate dataset.

    python scripts/prepare_data/gate_independent.py generate --models <model-bundle> \
        --trainval <dataset-bundle> --output <new-data-directory>
    python scripts/prepare_data/gate_independent.py verify --data <data-directory>
    python scripts/prepare_data/gate_independent.py encode --data <data-directory> --device cpu
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bounded_memory.data.io import read_json  # noqa: E402
from bounded_memory.environments.gate.holdout_protocol import COHORT_DIR  # noqa: E402
from bounded_memory.environments.gate.prepare_holdout import (  # noqa: E402
    encode_holdout,
    generate,
    verify_videos,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    gen = stages.add_parser("generate", help="render the registered scene pairs")
    gen.add_argument("--models", type=Path, required=True)
    gen.add_argument("--trainval", type=Path, required=True)
    gen.add_argument("--output", type=Path, required=True)
    gen.add_argument("--prior-manifest", type=Path, nargs="+", default=[COHORT_DIR / "prior_ids.csv.gz"])
    gen.add_argument("--provenance", type=Path, default=COHORT_DIR / "cohort.json")
    verify = stages.add_parser("verify", help="check the rendered sibling videos")
    verify.add_argument("--data", type=Path, required=True)
    enc = stages.add_parser("encode", help="extract features or resume a partial cache")
    enc.add_argument("--data", type=Path, required=True)
    enc.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    if args.stage == "generate":
        generate(args.models, args.trainval, args.prior_manifest, read_json(args.provenance), args.output)
    elif args.stage == "verify":
        verify_videos(args.data, read_json(COHORT_DIR / "cohort.json")["manifest_sha256"])
    else:
        encode_holdout(args.data, args.device)


if __name__ == "__main__":
    main()
