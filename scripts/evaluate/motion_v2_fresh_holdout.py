"""Motion v2: check the dev-chosen deletion-cost scorers on a second set of fresh scenes before reading validation.

    BM_MOTION_ROOT=/scratch/$USER python scripts/evaluate/motion_v2_fresh_holdout.py \\
        --config configs/motion_v2/frozen.yaml --seed 0 --device cuda --out <dir> \\
        --fresh <scenes dir> --tables <scorer dir>/tables --variants <variants dir>

Why: motion_v2_fresh_variants.py chose its variant and checkpoints on 96 fresh dev scene pairs, taking the best
of about 50 checkpoints per seed, and dev position MSE moves by about 0.002 between neighbouring checkpoints. A
dev gain of that size can be selection luck. Fresh scene pairs 608 to 703 (same generator and split name, never
used for training or selection) give an independent estimate without spending a validation look.

For each variant's saved scorers (three seeds) and the attention policy (step 0), the kept pair of every holdout
episode is scored from the all-pairs loss tables (reader-mean position MSE at +3 and +6, the dev metric).
Paired with attention per scene pair (mean of siblings, horizons and seeds); 95% interval over the 96 scene
pairs (10,000 draws, seed 20261126). Negative = the scorer has lower error than attention.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from bounded_memory import runs  # noqa: E402
from bounded_memory.environments.motion_v2 import deletion_cost_v2 as dc, fresh, frozen  # noqa: E402

SEEDS = (17, 29, 43)


def final_losses(tables, episodes, kept):
    """[episodes, 2 horizons] position loss of each episode's final pair."""
    return np.array([tables[ep["rec"]["id"]][:, a, b] for ep, (a, b) in zip(episodes, kept)])


def paired(ours, attention, episodes, resamples=10000, seed=20261126):
    """ours, attention: [episodes, 2]. Scene-pair means of (ours - attention), with a pair bootstrap."""
    pairs = sorted({ep["rec"]["pair"] for ep in episodes})
    index = {p: [i for i, ep in enumerate(episodes) if ep["rec"]["pair"] == p] for p in pairs}
    diff = np.array([(ours[index[p]] - attention[index[p]]).mean() for p in pairs])
    rng = np.random.default_rng(seed)
    boot = diff[rng.integers(0, len(pairs), size=(resamples, len(pairs)))].mean(1)
    return {"pairs": len(pairs), "difference": float(diff.mean()),
            "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "share_of_pairs_lower": float((diff < 0).mean())}


def main():
    ap = runs.parser(__doc__)
    ap.add_argument("--fresh", type=Path, required=True)
    ap.add_argument("--tables", type=Path, required=True)
    ap.add_argument("--variants", type=Path, required=True)
    ap.add_argument("--first-pair", type=int, default=608)
    ap.add_argument("--pairs", type=int, default=96)
    args = ap.parse_args()
    os.environ.setdefault("BM_REPO", str(runs.REPO))
    task = runs.Task(args, "fresh_holdout")
    if task.done():
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    config = runs.load_config(args.config)
    paths = frozen.Paths.from_config(config["assets"])
    frozen.verify(paths, ["ordinary"])
    assets = frozen.load(paths, args.device)
    summary = json.loads((args.variants / "summary.json").read_text())

    records = fresh.generate(args.fresh, args.first_pair + args.pairs, log=task.log)
    holdout = [r for r in records if r["scene_index"] >= args.first_pair]
    fresh.encode(assets, args.fresh, holdout, args.device, log=task.log)
    episodes = [fresh.episode(assets, args.fresh, r) for r in holdout]
    tables = {}
    for ep in episodes:
        path = args.tables / f"{ep['rec']['id']}.npy"
        if not path.exists():
            np.save(args.tables / f"{ep['rec']['id']}.tmp.npy", dc.pair_table(assets, ep))
            os.replace(args.tables / f"{ep['rec']['id']}.tmp.npy", path)
        tables[ep["rec"]["id"]] = np.load(path)
    task.log(f"holdout: {len(episodes)} episodes from fresh scene pairs {args.first_pair}..{args.first_pair + args.pairs - 1}")

    identity = {"items_mean": np.zeros(dc.ITEM_DIM), "items_std": np.ones(dc.ITEM_DIM),
                "context_mean": np.zeros(dc.CONTEXT_DIM), "context_std": np.ones(dc.CONTEXT_DIM)}
    attention = final_losses(tables, episodes, dc.rollout(assets, episodes, dc.ScorerV2().to(args.device), identity, args.device))
    best = np.array([tables[ep["rec"]["id"]].mean(0).min() for ep in episodes])
    out = {"holdout_pairs": [args.first_pair, args.first_pair + args.pairs - 1], "chosen_on_dev": summary["chosen"],
           "attention": {"+3": float(attention[:, 0].mean()), "+6": float(attention[:, 1].mean()),
                         "mean": float(attention.mean())},
           "best_pair_mean": float(best.mean()), "variants": {}}
    task.log(f"attention +3 {attention[:, 0].mean():.6f} +6 {attention[:, 1].mean():.6f}; best pair {best.mean():.6f}")
    for name in summary["variants"]:
        per_seed = []
        for seed in SEEDS:
            state = torch.load(args.variants / f"{name}_seed{seed}.pt", map_location="cpu", weights_only=False)
            model = dc.ScorerV2(state["width"]).to(args.device)
            model.load_state_dict(state["model"])
            per_seed.append(final_losses(tables, episodes, dc.rollout(assets, episodes, model, state["norm"], args.device)))
        ours = np.mean(per_seed, 0)
        out["variants"][name] = {
            "dev_mean": summary["variants"][name]["dev_mean"],
            "holdout_+3": float(ours[:, 0].mean()), "holdout_+6": float(ours[:, 1].mean()), "holdout_mean": float(ours.mean()),
            "holdout_mean_per_seed": {s: float(v.mean()) for s, v in zip(SEEDS, per_seed)},
            "paired_with_attention": paired(ours, attention, episodes)}
        p = out["variants"][name]["paired_with_attention"]
        task.log(f"{name:12s} dev {out['variants'][name]['dev_mean']:.6f} holdout {ours.mean():.6f} "
                 f"(attention {attention.mean():.6f}) diff {p['difference']:+.6f} 95% CI [{p['ci95'][0]:+.6f}, {p['ci95'][1]:+.6f}]"
                 f" lower in {100 * p['share_of_pairs_lower']:.0f}% of pairs")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "holdout.json").write_text(json.dumps(out, indent=1) + "\n")
    task.write(environment="motion_v2", method="deletion_cost_v2_fresh_holdout", holdout=out)


if __name__ == "__main__":
    main()
