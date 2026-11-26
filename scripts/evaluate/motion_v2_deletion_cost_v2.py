"""Score our Motion v2 scorers (v2): on the fresh holdout, then, only after a holdout gain, once on validation.

    BM_MOTION_ROOT=<study folders> BM_SCRATCH=<scratch> python scripts/evaluate/motion_v2_deletion_cost_v2.py \\
        --config configs/motion_v2/fresh.yaml --seed 0 --device cuda --out <dir> --choice <variants.json> \\
        --split holdout|validation [--checkpoints <dir>]

holdout     fresh scene pairs 608-703 (never used for training or selection). Every variant's three scorers
            and the attention policy (step 0) are scored from the loss tables and paired per scene pair (mean of
            siblings, horizons and seeds; 95% interval over the scene pairs). The result, with the SHA256 of
            every checkpoint scored, is the task file <out>/deletion_cost_v2_holdout_seed<seed>.json.
validation  the chosen variant's three scorers replayed on both validation intervals and scored by
            FrozenAssets.losses, like every row of the Motion v2 table. Runs only if that holdout file (same
            --out and --seed) shows the chosen variant below attention with the whole interval under zero, and
            only on the checkpoints the holdout scored (same SHA256, checked again as each one is loaded). Saves per-episode losses to <out>/raw/<interval>/ and appends the result
            to <out>/validation_ledger.jsonl, so every look stays on record.

--choice is the file of scripts/aggregate_results/motion_v2_scorer_variants.py; the scorers are read from
--checkpoints (default <out>/checkpoints, as in the train script). --seed only names the task file (each call
scores the three scorer seeds). The procedures are
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
    ap.add_argument("--checkpoints", type=Path, help="folder of the scorers (default: <out>/checkpoints)")
    args = ap.parse_args()
    os.environ.setdefault("BM_REPO", str(runs.REPO))
    out = Path(args.out)
    task = runs.Task(args, f"deletion_cost_v2_{args.split}")
    if task.done():
        return
    choice = json.loads(args.choice.read_text())
    config = runs.load_config(args.config)
    folder = args.checkpoints or out / "checkpoints"
    hashes = None
    if args.split == "validation":
        holdout_path = out / f"deletion_cost_v2_holdout_seed{args.seed}.json"
        holdout = json.loads(holdout_path.read_text()).get("result", {}) if holdout_path.exists() else {}
        files = {s: folder / f"{choice['chosen']}_seed{s}.pt" for s in SEEDS}
        hashes = {s: frozen.sha256(f) for s, f in files.items() if f.exists()}
        same_choice = holdout.get("chosen_on_dev") == choice["chosen"]          # the variant chosen before the holdout
        complete = len(hashes) == len(files)
        if not (same_choice and complete and scorer.validation_allowed(holdout, choice["chosen"], hashes)):
            raise SystemExit(f"validation not read: {choice['chosen']} is not the variant the holdout recorded as chosen "
                             f"on dev, shows no holdout gain over attention, or its checkpoints in {folder} are missing "
                             f"or not the ones the holdout scored ({holdout_path})")
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
    else:
        result = scorer.validation_scores(assets, paths, folder, choice["chosen"], SEEDS, args.device, out / "raw",
                                          hashes=hashes, log=task.log)
        entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "source_sha256": runs.source_hash(),
                 "variant": choice["chosen"], "seeds": list(SEEDS), "results": result}
        with open(out / "validation_ledger.jsonl", "a") as ledger:
            ledger.write(json.dumps(entry) + "\n")
    task.write(environment="motion_v2", method="deletion_cost_v2", split=args.split, chosen=choice["chosen"],
               result=result)


if __name__ == "__main__":
    main()
