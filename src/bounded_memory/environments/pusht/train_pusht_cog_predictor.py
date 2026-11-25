"""PushT latent predictor on hidden-CoG marker data (v5/v5b), and retention evaluation.

Predicts frozen I-JEPA latents at query + 6 and + 12 from the 4-frame present (query-3..query),
up to B = 2 retained history frames and the absolute pusher targets. One model serves every
retention rule: each training sample keeps a random memory condition (none, a random pair, a
random single frame, or the marker frames), as the repo's Gate predictor was trained.

Bases (train only, no labels):
  input   StandardScaler + whitened PCA (rank 128) on train frames up to the query, plus the
          differences between siblings' frames at the same time step, so the marker's direction
          is kept even though it carries little variance (siblings differ only in marker frames)
  output  PCA (rank 128) of standardized future changes z[q+h] - z[q], plus sibling differences
          of those futures (as phase1 fit_basis does)

Evaluation on validation (evaluation split never read), per seed and per CoG class:
  target MSE      mean squared error over the 5120 standardized features at query + h
  sibling ID      fraction of episodes whose prediction is closest to their own future among the
                  three siblings' futures (siblings share the present: recent-only gives 1/3)
Retention conditions: recent-only, FIFO (last two history frames), first frame, a seeded random
pair, the marker frames (oracle reference: marker indices are privileged), and an exhaustive pair
oracle (the pair with the lowest error at +12 for that episode; an upper bound, uses the target).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

R, B, RANK = 4, 2, 128
HORIZONS = (6, 12)
SPLITS = ("train", "validation")
TRAIN_CONDITIONS = ("recent_only", "random_pair", "random_single", "marker")


def load(data: Path, features: Path):
    index = [r for r in json.loads((data / "observation_index.json").read_text())["records"] if r["split"] in SPLITS]
    eps = []
    for r in index:
        ev = json.loads((data / "evaluation_only" / f"{r['id']}.json").read_text())
        with np.load(data / r["observations"], allow_pickle=False) as a:
            actions = a["actions_normalized"].astype(np.float32)
        eps.append({**r, "z": np.load(features / "features" / f"{r['id']}.npy").astype(np.float32),
                    "actions": actions, "marker": ev["marker_frames"], "cog": ev["cog_class"]})
    return eps


def fit_bases(train, seed):
    rng = np.random.default_rng(seed)
    frames = np.concatenate([e["z"][:e["query_frame"] + 1] for e in train])
    mean, std = frames.mean(0), frames.std(0) + 1e-6
    standardize = lambda x: (x - mean) / std
    fam = {}
    for e in train:
        fam.setdefault(e["family_id"], []).append(e)
    sib_frames, sib_future, future = [], [], []
    for es in fam.values():
        q = es[0]["query_frame"]
        zs = [standardize(e["z"]) for e in es]
        for a, b in itertools.combinations(range(len(es)), 2):
            d = zs[a][:q + 1] - zs[b][:q + 1]
            sib_frames.append(d[np.abs(d).max(1) > 1e-6])          # nonzero only at marker frames
            sib_future.append(np.stack([(zs[a][q + h] - zs[a][q]) - (zs[b][q + h] - zs[b][q]) for h in HORIZONS]))
        future += [np.stack([z[q + h] - z[q] for h in HORIZONS]) for z in zs]
    x = standardize(frames)
    d = np.concatenate(sib_frames)
    # Scale sibling differences to the frames' total variance share so PCA cannot ignore them.
    d = d * np.sqrt(x.var(0).sum() / max(d.var(0).sum(), 1e-9)) * np.sqrt(len(x) / max(len(d), 1))
    pca_in = torch.pca_lowrank(torch.from_numpy(np.concatenate([x, d])), q=RANK, center=True, niter=4)
    _, s, v = pca_in
    scale = (s / np.sqrt(len(x) + len(d))).clamp_min(0.05)
    future = np.stack(future)                                     # [n, H, D]
    target_mean = future.mean(0)
    out = np.concatenate([(future - target_mean).reshape(-1, future.shape[-1]),
                          2 * np.concatenate(sib_future).reshape(-1, future.shape[-1])])
    _, _, vo = torch.pca_lowrank(torch.from_numpy(out), q=RANK, center=False, niter=4)
    del rng
    return {"mean": mean, "std": std, "proj": (v / scale).numpy().astype(np.float32),
            "proj_mean": x.mean(0).astype(np.float32), "target_mean": target_mean.astype(np.float32),
            "out_basis": vo.numpy().astype(np.float32)}


class Predictor(nn.Module):
    def __init__(self, hidden=512):
        super().__init__()
        self.slot = nn.Sequential(nn.Linear(RANK + 3, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.present = nn.Sequential(nn.Linear(R * (RANK + 2), hidden), nn.GELU())
        self.future = nn.Sequential(nn.Linear(max(HORIZONS) * 2, hidden), nn.GELU())
        self.trunk = nn.Sequential(nn.Linear(4 * hidden + 1, hidden), nn.GELU(), nn.Dropout(0.1),
                                   nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, len(HORIZONS) * RANK))

    def forward(self, present, slots, mask, future):
        h = self.slot(slots) * mask[..., None]
        n = mask.sum(1, keepdim=True)
        mean = h.sum(1) / n.clamp_min(1)
        mx = h.masked_fill(~mask.bool()[..., None], -1e4).max(1).values * (n > 0)
        x = torch.cat([self.present(present.flatten(1)), mean, mx, self.future(future.flatten(1)), n / B], -1)
        return self.trunk(x).view(-1, len(HORIZONS), RANK)


class Batcher:
    """Turns (episode, retained frames) into tensors; frames are projected once per episode."""

    def __init__(self, eps, bases):
        self.b = bases
        self.std = {e["id"]: (e["z"] - bases["mean"]) / bases["std"] for e in eps}
        self.p = {e["id"]: (self.std[e["id"]] - bases["proj_mean"]) @ bases["proj"] for e in eps}

    def tensors(self, items):
        present, slots, mask, future, current, target = [], [], [], [], [], []
        for e, kept in items:
            q, p, a = e["query_frame"], self.p[e["id"]], e["actions"]
            times = range(q - R + 1, q + 1)
            present.append(np.concatenate([np.concatenate([p[t], a[t - 1]]) for t in times]))
            s = np.zeros((B, RANK + 3), np.float32)
            m = np.zeros(B, np.float32)
            for i, t in enumerate(sorted(kept)[:B]):
                s[i] = np.concatenate([p[t], [(q - t) / 64.0], a[max(t - 1, 0)]])
                m[i] = 1
            slots.append(s)
            mask.append(m)
            future.append(a[q:q + max(HORIZONS)])
            current.append(self.std[e["id"]][q])
            target.append(np.stack([self.std[e["id"]][q + h] for h in HORIZONS]))
        f = lambda x: torch.from_numpy(np.stack(x).astype(np.float32))
        return f(present), f(slots), f(mask), f(future), f(current), f(target)


def decode(coeff, current, bases):
    basis = torch.from_numpy(bases["out_basis"])
    return current[:, None] + torch.from_numpy(bases["target_mean"])[None] + coeff @ basis.T


def condition_frames(e, condition, rng):
    q = e["query_frame"]
    history = list(range(0, q - R + 1))
    if condition == "recent_only":
        return []
    if condition == "random_pair":
        return list(rng.choice(history, 2, replace=False))
    if condition == "random_single":
        return [int(rng.choice(history))]
    if condition == "marker":
        m = e["marker"]
        return [m[0], m[-1]]
    if condition == "fifo":
        return history[-B:]
    if condition == "first_frame":
        return [0]
    raise ValueError(condition)


def train(train_eps, bases, seed, epochs, batch_size=64, lr=1e-3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Predictor()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    batcher = Batcher(train_eps, bases)
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(train_eps))
        for start in range(0, len(order), batch_size):
            items = [(train_eps[i], condition_frames(train_eps[i], TRAIN_CONDITIONS[rng.integers(len(TRAIN_CONDITIONS))], rng))
                     for i in order[start:start + batch_size]]
            present, slots, mask, future, current, target = batcher.tensors(items)
            loss = (decode(model(present, slots, mask, future), current, bases) - target).square().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    return model.eval()


@torch.inference_mode()
def predict(model, batcher, items, bases):
    present, slots, mask, future, current, target = batcher.tensors(items)
    return decode(model(present, slots, mask, future), current, bases).numpy(), target.numpy()


@torch.inference_mode()
def evaluate(model, val_eps, bases, seed):
    batcher = Batcher(val_eps, bases)
    rng = np.random.default_rng(seed + 1000)
    by_id = {e["id"]: e for e in val_eps}
    fam = {}
    for e in val_eps:
        fam.setdefault(e["family_id"], []).append(e["id"])
    preds = {}
    for cond in ("recent_only", "fifo", "first_frame", "random_pair", "marker"):
        items = [(e, condition_frames(e, cond, rng)) for e in val_eps]
        p, t = predict(model, batcher, items, bases)
        preds[cond] = {e["id"]: (p[i], t[i], items[i][1]) for i, e in enumerate(val_eps)}
    # Exhaustive pair oracle at +12 (upper bound; selects with the target).
    oracle = {}
    for e in val_eps:
        q = e["query_frame"]
        pairs = list(itertools.combinations(range(0, q - R + 1), 2))
        p, t = predict(model, batcher, [(e, list(pr)) for pr in pairs], bases)
        err = ((p[:, -1] - t[:, -1]) ** 2).mean(1)
        k = int(err.argmin())
        oracle[e["id"]] = (p[k], t[k], list(pairs[k]))
    preds["pair_oracle"] = oracle

    out = {}
    for cond, res in preds.items():
        row = {}
        for hi, h in enumerate(HORIZONS):
            mse = {i: float(((v[0][hi] - v[1][hi]) ** 2).mean()) for i, v in res.items()}
            ident = {}
            for ids in fam.values():
                truths = np.stack([res[i][1][hi] for i in ids])
                for i in ids:
                    d = ((truths - res[i][0][hi][None]) ** 2).mean(1)
                    ident[i] = float(ids[int(d.argmin())] == i)
            per_class = {c: {"mse": float(np.mean([mse[i] for i in res if by_id[i]["cog"] == c])),
                             "sibling_id": float(np.mean([ident[i] for i in res if by_id[i]["cog"] == c]))}
                         for c in sorted({e["cog"] for e in val_eps})}
            row[f"h{h}"] = {"mse": float(np.mean(list(mse.values()))), "sibling_id": float(np.mean(list(ident.values()))),
                            "per_class": per_class}
        if cond == "pair_oracle":
            hits = [any(t in by_id[i]["marker"] for t in res[i][2]) for i in res]
            row["oracle_pair_contains_marker_frame"] = float(np.mean(hits))
        out[cond] = row
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--save-dir", type=Path, help="save each seed's model and bases here (seed_<n>.pt)")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    eps = load(args.data, args.features)
    train_eps = [e for e in eps if e["split"] == "train"]
    val_eps = [e for e in eps if e["split"] == "validation"]
    # The output is rewritten after every seed ("complete": false until the end); a resubmission skips
    # seeds already in it and reuses saved models in --save-dir instead of retraining them.
    results = {}
    if args.output.exists():
        previous = json.loads(args.output.read_text())
        results = {s: v for s, v in previous.get("per_seed", {}).items() if int(s) in args.seeds}

    def write_output(complete):
        done = [s for s in args.seeds if str(s) in results]
        conds = list(results[str(done[0])]) if done else []
        summary = {c: {f"h{h}": {k: float(np.mean([results[str(s)][c][f"h{h}"][k] for s in done]))
                                 for k in ("mse", "sibling_id")} for h in HORIZONS} for c in conds}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "completed_utc": datetime.now(timezone.utc).isoformat(), "complete": complete, "seeds_done": done,
            "data": args.data.name, "features": str(args.features), "seeds": args.seeds, "epochs": args.epochs,
            "counts": {"train_episodes": len(train_eps), "validation_episodes": len(val_eps)},
            "train_conditions": TRAIN_CONDITIONS, "horizons": HORIZONS, "rank": RANK,
            "notes": {"marker": "marker frame indices are privileged (evaluation_only); used for the marker training "
                                "condition and the marker reference only",
                      "pair_oracle": "selects with the validation target; an upper bound, not a policy",
                      "mse": "mean over 5120 standardized I-JEPA features"},
            "summary_mean_over_seeds": summary, "per_seed": results,
        }, indent=2) + "\n")
        tmp.replace(args.output)
        return conds, summary

    for seed in args.seeds:
        if str(seed) in results:
            print(f"seed {seed}: already evaluated, skipping", flush=True)
            continue
        saved = args.save_dir / f"seed_{seed}.pt" if args.save_dir is not None else None
        if saved is not None and saved.exists():
            ck = torch.load(saved, map_location="cpu", weights_only=False)
            bases, model = ck["bases"], Predictor()
            model.load_state_dict(ck["model"])
            model.eval()
            print(f"seed {seed}: reusing saved model {saved}", flush=True)
        else:
            bases = fit_bases(train_eps, seed)
            model = train(train_eps, bases, seed, args.epochs)
            if saved is not None:
                args.save_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(), "bases": bases, "seed": seed, "epochs": args.epochs,
                            "data": args.data.name}, saved)
        results[str(seed)] = evaluate(model, val_eps, bases, seed)
        write_output(complete=False)
        r = results[str(seed)]
        print(f"seed {seed}: " + "  ".join(f"{c} mse {v['h12']['mse']:.3f} id {v['h12']['sibling_id']:.2f}"
                                          for c, v in r.items()), flush=True)
    conds, summary = write_output(complete=True)
    for c in conds:
        print(f"{c:12s} " + "  ".join(f"h{h}: mse {summary[c][f'h{h}']['mse']:.4f} sibling_id {summary[c][f'h{h}']['sibling_id']:.3f}"
                                      for h in HORIZONS))


if __name__ == "__main__":
    main()
