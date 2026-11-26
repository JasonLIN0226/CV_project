"""Score our Motion v2 scorers (v2): on the fresh holdout, then, only after a holdout gain, once on validation.

    BM_MOTION_ROOT=<study folders> BM_SCRATCH=<scratch> BM_RUN=<run folder> \\
    python scripts/evaluate/motion_v2_deletion_cost_v2.py --config configs/motion_v2/fresh.yaml --seed 0 \\
        --device cuda --out <dir> --choice <variants.json> --split holdout|validation

holdout     fresh scene pairs 608-703 (never used for training or selection). Every variant's three scorers
            and the attention policy (step 0) are scored from the loss tables and paired per scene pair (mean of
            siblings, horizons and seeds; 95% interval over the scene pairs). Writes <out>/holdout.json, with
            the SHA256 of every checkpoint scored.
validation  the chosen variant's three scorers replayed on both validation intervals and scored by
            FrozenAssets.losses, like every row of the Motion v2 table. Runs only if <out>/holdout.json shows the
            chosen variant below attention with the whole interval under zero, and only on the checkpoints the
            holdout scored (same SHA256). Saves per-episode losses to <out>/raw/<interval>/ and appends the result
            to <out>/validation_ledger.jsonl, so every look stays on record.

--choice is the file of scripts/aggregate_results/motion_v2_scorer_variants.py; the scorers are read from the
config's scorer.checkpoints. --seed is unused (each call scores the three scorer seeds). The procedures are
methods/deletion_cost/motion_v2.holdout_comparison, validation_allowed and validation_scores.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch  # noqa: E402

from bounded_memory import runs  # noqa: E402
from bounded_memory.environments.motion_v2 import fresh, frozen  # noqa: E402
from bounded_memory.environments.motion_v2.step34_core import SEEDS  # noqa: E402
from bounded_memory.methods.deletion_cost import motion_v2 as scorer  # noqa: E402


def main():
    ap = runs.parser(__doc__)
    ap.add_argument("--choice", type=Path, required=True)
    ap.add_argument("--split", required=True, choices=("holdout", "validation"))
    args = ap.parse_args()
    os.environ.setdefault("BM_REPO", str(runs.REPO))
    out = Path(args.out)
    task = runs.Task(args, f"deletion_cost_v2_{args.split}")
    if task.done():
        return
    choice = json.loads(args.choice.read_text())
    config = runs.load_config(args.config)
    folder = Path(config["scorer"]["checkpoints"])
    if args.split == "validation":
        holdout_path = out / "holdout.json"
        holdout = json.loads(holdout_path.read_text()) if holdout_path.exists() else {}
        hashes = {s: frozen.sha256(folder / f"{choice['chosen']}_seed{s}.pt") for s in SEEDS}
        if not scorer.validation_allowed(holdout, choice["chosen"], hashes):
            raise SystemExit(f"validation not read: {choice['chosen']} shows no holdout gain over attention, or its "
                             f"checkpoints are not the ones the holdout scored ({holdout_path})")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    paths = frozen.Paths.from_config(config["assets"])
    frozen.verify(paths, list(frozen.INTERVALS) if args.split == "validation" else ["ordinary"])
    assets = frozen.load(paths, args.device)
    out.mkdir(parents=True, exist_ok=True)

    if args.split == "holdout":
        eps = fresh.load(assets, config["fresh"], parts=("holdout",))["holdout"]
        tables = scorer.load_tables(config["fresh"]["tables"], eps)
        result = scorer.holdout_comparison(assets, eps, tables, folder, list(choice["variants"]), SEEDS, args.device,
                                           log=task.log)
        result["chosen_on_dev"] = choice["chosen"]
        (out / "holdout.json").write_text(json.dumps(result, indent=1) + "\n")
        summary = {"attention": result["attention"], "chosen": result["variants"][choice["chosen"]]["paired_with_attention"]}
    else:
        result = scorer.validation_scores(assets, paths, folder, choice["chosen"], SEEDS, args.device, out / "raw",
                                          log=task.log)
        entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "source_sha256": runs.source_hash(),
                 "variant": choice["chosen"], "seeds": list(SEEDS), "results": result}
        with open(out / "validation_ledger.jsonl", "a") as ledger:
            ledger.write(json.dumps(entry) + "\n")
        summary = result
    task.write(environment="motion_v2", method="deletion_cost_v2", split=args.split, chosen=choice["chosen"],
               summary=summary)


if __name__ == "__main__":
    main()
