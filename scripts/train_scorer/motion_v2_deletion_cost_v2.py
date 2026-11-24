"""Deletion-cost scorer v2 on Motion v2 (environments/motion_v2/deletion_cost_v2.py), on the frozen assets.

    BM_MOTION_ROOT=/scratch/xy3257 python scripts/train_scorer/motion_v2_deletion_cost_v2.py \\
        --config configs/motion_v2/frozen.yaml --seed 0 --seeds 17 29 43 --device cuda --out <dir> [--validate]

Stages, each cached under <out> so an interrupted job continues where it stopped:

    tables/<episode>.npy   all-pairs +3/+6 position loss of the three frozen readers, every training episode
    train_states.npz       behaviour-policy states of the non-dev training episodes: inputs, salience prior,
                           optimal-continuation deletion costs
    scorer_seed<s>.pt      one scorer per seed, checkpoint chosen on the dev split (10% of training scene pairs)
    raw/<interval>/deletion_cost_v2_seed<s>.npz   only with --validate: the validation replay, scored by
                           FrozenAssets.losses like every other row of the Motion v2 table

Without --validate nothing outside the training split is read. Every --validate run is appended to
<out>/validation_ledger.jsonl (time, code hash, seeds, results) so the number of looks stays on record.
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
from bounded_memory.environments.motion_v2 import deletion_cost_v2 as dc, frozen, literature  # noqa: E402

SEEDS = (17, 29, 43)


def stage_tables(assets, episodes, folder, log):
    folder.mkdir(parents=True, exist_ok=True)
    todo = [ep for ep in episodes if not (folder / f"{ep['rec']['id']}.npy").exists()]
    started = time.time()
    for i, ep in enumerate(todo):
        np.save(folder / f"{ep['rec']['id']}.tmp.npy", dc.pair_table(assets, ep))
        os.replace(folder / f"{ep['rec']['id']}.tmp.npy", folder / f"{ep['rec']['id']}.npy")
        if i % 50 == 0 or i == len(todo) - 1:
            log(f"tables {i + 1}/{len(todo)} new ({len(episodes) - len(todo)} cached), {time.time() - started:.0f}s")
    return {ep["rec"]["id"]: np.load(folder / f"{ep['rec']['id']}.npy") for ep in episodes}


def stage_states(assets, episodes, tables, path, log):
    if path.exists():
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    items, context, prior, cost = [], [], [], []
    for i, ep in enumerate(episodes):
        q = ep["rec"]["query"]
        for clock, pool in dc.behaviour_states(assets, ep, 20261124):
            x, c, p = dc.observe(assets, ep, clock, pool)
            items.append(x); context.append(c); prior.append(p)
            cost.append(dc.continuation_cost(tables[ep["rec"]["id"]], clock, pool, q))
        if i % 200 == 0:
            log(f"states from {i + 1}/{len(episodes)} episodes: {len(cost)}")
    data = {"items": np.stack(items), "context": np.stack(context), "prior": np.stack(prior), "cost": np.stack(cost)}
    np.savez(path, **data)
    return data


def main():
    ap = runs.parser(__doc__)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS), choices=SEEDS)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--validate", action="store_true", help="also replay the trained scorers on the validation intervals")
    args = ap.parse_args()
    os.environ.setdefault("BM_REPO", str(runs.REPO))
    task = runs.Task(args, "deletion_cost_v2" + ("_validated" if args.validate else ""))
    if task.done():
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = runs.load_config(args.config)
    paths = frozen.Paths.from_config(config["assets"])
    intervals = list(frozen.INTERVALS) if args.validate else ["ordinary"]
    task.info["verification"] = {k: v.get("checked", v.get("ok")) for k, v in frozen.verify(paths, intervals).items()}
    assets = frozen.load(paths, args.device)
    records = assets.store.manifest["records"]
    dev = dc.dev_pairs(records)
    train_raw = [assets.episode(r) for r in records if r["split"] == "train"]
    dev_eps = [ep for ep in train_raw if ep["rec"]["pair"] in dev]
    fit_eps = [ep for ep in train_raw if ep["rec"]["pair"] not in dev]
    task.log(f"training episodes {len(train_raw)}: fit {len(fit_eps)}, dev {len(dev_eps)} ({len(dev)} scene pairs)")

    tables = stage_tables(assets, train_raw, out / "tables", task.log)
    data = stage_states(assets, fit_eps, tables, out / "train_states.npz", task.log)
    task.info["train_states"] = int(len(data["cost"]))
    attention = dc.table_loss(tables, dev_eps, dc.rollout(assets, dev_eps, dc.ScorerV2().to(args.device),
                              dc.fit_normalization(data["items"], data["context"]), args.device))
    task.info["dev_attention_policy"] = {"+3": float(attention[0]), "+6": float(attention[1])}
    task.log(f"{len(data['cost'])} training states; dev attention policy +3 {attention[0]:.6f} +6 {attention[1]:.6f}")

    fits = {}
    for seed in args.seeds:
        path = out / f"scorer_seed{seed}.pt"
        if path.exists():
            state = torch.load(path, map_location="cpu", weights_only=False)
            fits[seed] = state
            continue
        model, norm, meta = dc.train(data, tables, dev_eps, assets, seed, args.device, steps=args.steps,
                                     log=lambda m, s=seed: task.log(f"seed {s} {m}"))
        state = {"model": {k: v.cpu() for k, v in model.state_dict().items()}, "norm": norm, "meta": meta}
        torch.save(state, path)
        fits[seed] = state
    task.info["fits"] = {s: f["meta"] for s, f in fits.items()}

    results = {}
    if args.validate:
        for interval in intervals:
            recs, external = frozen.interval_inputs(assets, paths, interval)
            raw, _ = literature.split_episodes(assets, recs, "validation", external)
            scores = {}
            for seed, state in fits.items():
                model = dc.ScorerV2(state["meta"]["width"]).to(args.device)
                model.load_state_dict(state["model"])
                kept = dc.rollout(assets, raw, model, state["norm"], args.device)
                scores[seed] = literature.score_frames(assets, raw, kept)
                target = out / "raw" / interval / f"deletion_cost_v2_seed{seed}.npz"
                target.parent.mkdir(parents=True, exist_ok=True)
                np.savez(target, kept=np.asarray(kept), episode_ids=np.array([r["rec"]["id"] for r in raw]), **scores[seed])
            results[interval] = literature.summarize(scores, list(fits))
            task.log(f"validation {interval}: +3 position MSE {results[interval]['position_mse_+3']:.6f}")
        with open(out / "validation_ledger.jsonl", "a") as ledger:
            ledger.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "source_sha256": runs.source_hash(),
                                     "seeds": list(fits), "steps": args.steps, "results": results}) + "\n")
    task.write(environment="motion_v2", method="deletion_cost_v2", dev_pairs=len(dev), validation=results or None)


if __name__ == "__main__":
    main()
