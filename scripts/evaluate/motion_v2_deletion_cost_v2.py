"""Score our Motion v2 scorers (v2): on the fresh holdout, then, only if that shows a gain, once on validation.

    BM_MOTION_ROOT=/scratch/$USER python scripts/evaluate/motion_v2_deletion_cost_v2.py \\
        --config configs/motion_v2/frozen.yaml --seed 0 --device cuda --out <dir> --choice <variants summary.json> \\
        --split holdout|validation

holdout     fresh scene pairs 608-703 (never used for training or selection). Every variant's three scorers
            and the attention policy (step 0) are scored from the loss tables, paired per scene pair (mean of
            siblings, horizons and seeds), 95% interval over the scene pairs (evaluation/metrics/motion_v2.py).
            Writes <out>/holdout.json.
validation  the chosen variant's three scorers replayed on both validation intervals and scored by
            FrozenAssets.losses, like every row of the Motion v2 table. Allowed only when <out>/holdout.json
            shows the chosen variant below attention with the whole interval under zero. Appends the result
            to <out>/validation_ledger.jsonl, so every look stays on record.

--choice is the file of scripts/aggregate_results/motion_v2_scorer_variants.py; checkpoints are read from the
config's scorer.checkpoints.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from bounded_memory import runs  # noqa: E402
from bounded_memory.environments.motion_v2 import episodes, frozen  # noqa: E402
from bounded_memory.evaluation.metrics import motion_v2 as metrics  # noqa: E402
from bounded_memory.methods.deletion_cost import motion_v2 as scorer  # noqa: E402

SEEDS = (17, 29, 43)


def load_scorer(folder, variant, seed, device):
    state = torch.load(Path(folder) / f"{variant}_seed{seed}.pt", map_location="cpu", weights_only=False)
    model = scorer.ScorerV2(state["width"]).to(device)
    model.load_state_dict(state["model"])
    return model, state["norm"]


def validation_allowed(holdout, variant):
    """The one validation look is spent only on a variant whose holdout interval lies wholly below attention."""
    entry = holdout["variants"].get(variant)
    return entry is not None and entry["paired_with_attention"]["ci95"][1] < 0


def holdout_check(assets, config, choice, device, task):
    eps = scorer.fresh_episodes(assets, config["fresh"], parts=("holdout",))["holdout"]
    tables = scorer.load_tables(config["fresh"]["tables"], eps)
    step_zero = scorer.ScorerV2().to(device)                    # untrained: the attention policy
    attention = metrics.final_pair_losses(tables, eps, scorer.rollout(assets, eps, step_zero, scorer.identity_norm(), device))
    out = {"chosen_on_dev": choice["chosen"], "episodes": len(eps),
           "attention": {"+3": float(attention[:, 0].mean()), "+6": float(attention[:, 1].mean()),
                         "mean": float(attention.mean())},
           "best_pair_mean": float(np.mean([tables[ep["rec"]["id"]].mean(0).min() for ep in eps])),
           "variants": {}}
    for name in choice["variants"]:
        per_seed = []
        for seed in SEEDS:
            model, norm = load_scorer(config["scorer"]["checkpoints"], name, seed, device)
            per_seed.append(metrics.final_pair_losses(tables, eps, scorer.rollout(assets, eps, model, norm, device)))
        mean = np.mean(per_seed, 0)
        out["variants"][name] = {"dev_mean": choice["variants"][name]["dev_mean"], "holdout_mean": float(mean.mean()),
                                 "holdout_+3": float(mean[:, 0].mean()), "holdout_+6": float(mean[:, 1].mean()),
                                 "holdout_mean_per_seed": {s: float(v.mean()) for s, v in zip(SEEDS, per_seed)},
                                 "paired_with_attention": metrics.paired(mean, attention, eps)}
        p = out["variants"][name]["paired_with_attention"]
        task.log(f"{name:12s} dev {choice['variants'][name]['dev_mean']:.6f} holdout {mean.mean():.6f} "
                 f"(attention {attention.mean():.6f}) diff {p['difference']:+.6f} "
                 f"95% CI [{p['ci95'][0]:+.6f}, {p['ci95'][1]:+.6f}], "
                 f"lower in {100 * p['share_of_pairs_lower']:.0f}% of pairs")
    return out


def validation_look(assets, paths, config, variant, device, task):
    results = {}
    for interval in frozen.INTERVALS:
        records, external = frozen.interval_inputs(assets, paths, interval)
        raw, _ = episodes.split_episodes(assets, records, "validation", external)
        scores = {}
        for seed in SEEDS:
            model, norm = load_scorer(config["scorer"]["checkpoints"], variant, seed, device)
            scores[seed] = episodes.score_frames(assets, raw, scorer.rollout(assets, raw, model, norm, device))
        results[interval] = episodes.summarize(scores, list(SEEDS))
        task.log(f"validation {interval} ({variant}): +3 {results[interval]['position_mse_+3']:.6f} "
                 f"+6 {results[interval]['position_mse_+6']:.6f}")
    return results


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
    if args.split == "validation":
        holdout_path = out / "holdout.json"
        holdout = json.loads(holdout_path.read_text()) if holdout_path.exists() else {"variants": {}}
        if not validation_allowed(holdout, choice["chosen"]):
            raise SystemExit(f"validation not read: {choice['chosen']} shows no holdout gain over attention ({holdout_path})")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    config = runs.load_config(args.config)
    paths = frozen.Paths.from_config(config["assets"])
    intervals = list(frozen.INTERVALS) if args.split == "validation" else ["ordinary"]
    frozen.verify(paths, intervals)
    assets = frozen.load(paths, args.device)
    out.mkdir(parents=True, exist_ok=True)
    if args.split == "holdout":
        result = holdout_check(assets, config, choice, args.device, task)
        (out / "holdout.json").write_text(json.dumps(result, indent=1) + "\n")
    else:
        result = validation_look(assets, paths, config, choice["chosen"], args.device, task)
        entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "source_sha256": runs.source_hash(),
                 "variant": choice["chosen"], "seeds": list(SEEDS), "results": result}
        with open(out / "validation_ledger.jsonl", "a") as ledger:
            ledger.write(json.dumps(entry) + "\n")
    task.write(environment="motion_v2", method="deletion_cost_v2", split=args.split, chosen=choice["chosen"], result=result)


if __name__ == "__main__":
    main()
