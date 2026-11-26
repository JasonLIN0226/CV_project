"""Deletion-cost retention on hidden-CoG PushT (marker data), with a frozen latent predictor.

Online memory: the model always sees the present (R = 4 frames). When a frame leaves the present
it competes with the B = 2 stored frames and one of the three is deleted for good.

Policies (all past-only at run time):
  fifo, reservoir, temporal_buckets, visual_diversity   the repo's audited rules (evaluation/heuristics/gate.py update,
                                                          extracted unchanged from gate_retention_comparison_v1/online.py)
  random                                                  delete one of the three uniformly
  deletion_cost                                           the learned student (below)
References: recent_only (no history) and marker (keeps the first and last marker frames; privileged).

Deletion-cost student (Gate v2 design, two changes agreed for PushT):
  * target-time teacher: for a decision state (clock, three candidates) each possible deletion is
    scored by the frozen predictor's error at the episode's real targets (query + 6, + 12), as if
    the two kept frames survived to the query; the three errors become a softmax target
    (lower error after deleting a candidate -> more likely to delete it)
  * training states along whole episodes: decision states sampled at many clocks from rollouts of
    the behaviour policies, not only at the query; then DAgger rounds roll out the student on the
    training episodes, relabel the states it visits and retrain on the aggregate
  Student: set-conditioned scorer over the three candidates (per-candidate MLP + mean-pooled
  context), cross-entropy to the teacher distribution. Inputs per candidate: projected latent, its
  difference to the current frame, log age, the pusher target at that time, the current frame, the
  mean of the present and the last action. No labels, targets or marker indices at run time.

Evaluation on validation (evaluation split never read): each policy's memory at the query goes to
the frozen predictor; error at +6/+12, sibling identification and marker hit rate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_pusht_cog_predictor as P  # noqa: E402

R, B = P.R, P.B
BEHAVIOURS = ("fifo", "reservoir", "temporal_buckets", "visual_diversity", "random")
EVAL_POLICIES = BEHAVIOURS + ("deletion_cost",)
FEAT = P.RANK * 4 + 1 + 2 + 2  # p[t], p[t]-current, current, mean of the present, log age, action at t, last action


def load_online_rules():
    path = HERE.parents[1] / "evaluation" / "heuristics" / "gate.py"
    spec = importlib.util.spec_from_file_location("gate_online_rules", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.update


class Scorer(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(FEAT, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x):                      # x: [batch, 3, FEAT + RANK] -> deletion logits [batch, 3]
        h = self.phi(x)
        pooled = h.mean(1, keepdim=True).expand_as(h)
        return self.head(torch.cat([h, pooled], -1)).squeeze(-1)


def state_features(p, actions, clock, candidates):
    current = p[clock]
    recent = p[clock - R + 1:clock + 1].mean(0)
    last = actions[clock - 1]
    rows = []
    for t in candidates:
        rows.append(np.concatenate([p[t], p[t] - current, current, recent, [math.log1p(clock - t) / math.log(65)],
                                    actions[max(t - 1, 0)], last]))
    return np.stack(rows).astype(np.float32)


def rollout(e, p, policy, rng, update, scorer=None, record=None):
    """Run one online episode to the query; returns the final memory (<= 2 frames).

    `record`, if a list, receives (clock, candidates) for every three-way decision."""
    q, actions = e["query_frame"], e["actions"]
    mem = []
    for clock in range(R, q + 1):
        arrival = clock - R
        pool = mem + [arrival]
        if len(pool) <= B:
            mem = sorted(pool)
            continue
        if record is not None:
            record.append((clock, tuple(pool)))
        if policy == "random":
            drop = pool[int(rng.integers(3))]
            mem = sorted(t for t in pool if t != drop)
        elif policy == "deletion_cost":
            with torch.inference_mode():
                logits = scorer(torch.from_numpy(state_features(p, actions, clock, pool))[None])[0].numpy()
            drop = pool[int(np.argmax(logits))]
            mem = sorted(t for t in pool if t != drop)
        else:
            vectors = {t: p[t] for t in pool} if policy == "visual_diversity" else {}
            mem = update(policy, list(mem), arrival, clock, R, vectors, p[clock], {}, rng)
    return mem


@torch.inference_mode()
def teacher_energy(model, batcher, bases, e, states):
    """[n_states, 3] errors at the query targets after deleting each candidate."""
    items = [(e, [t for t in cand if t != drop]) for _, cand in states for drop in cand]
    out = []
    for start in range(0, len(items), 4096):
        pred, target = P.predict(model, batcher, items[start:start + 4096], bases)
        out.append(((pred - target) ** 2).mean((1, 2)))
    return np.concatenate(out).reshape(len(states), 3)


def soft_target(energy):
    logits = -energy / np.maximum(energy.std(1, keepdims=True), 1e-6)
    w = np.exp(logits - logits.max(1, keepdims=True))
    return (w / w.sum(1, keepdims=True)).astype(np.float32)


def collect_states(eps, batcher, policy, rng, update, per_episode, scorer=None, repeats=1):
    """Sampled decision states (episode index, clock, candidates) from rollouts of `policy`."""
    out = []
    for i, e in enumerate(eps):
        for _ in range(repeats):
            rec = []
            rollout(e, batcher.p[e["id"]], policy, rng, update, scorer, rec)
            if not rec:
                continue
            keep = sorted(set(rng.choice(len(rec), min(per_episode, len(rec)), replace=False)) | {len(rec) - 1})
            out += [(i, rec[k][0], rec[k][1]) for k in keep]
    return out


def label(states, eps, model, batcher, bases):
    by_ep = {}
    for i, clock, cand in states:
        by_ep.setdefault(i, []).append((clock, cand))
    x, y = [], []
    for i, rows in by_ep.items():
        e = eps[i]
        energy = teacher_energy(model, batcher, bases, e, rows)
        y.append(soft_target(energy))
        p = batcher.p[e["id"]]
        x += [state_features(p, e["actions"], c, cand) for c, cand in rows]
    return np.stack(x), np.concatenate(y)


def fit_scorer(x, y, seed, epochs=30, batch=1024, lr=1e-3):
    torch.manual_seed(seed)
    scorer = Scorer()
    opt = torch.optim.AdamW(scorer.parameters(), lr=lr, weight_decay=1e-4)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for idx in torch.randperm(len(xt), generator=g).split(batch):
            loss = -(yt[idx] * torch.log_softmax(scorer(xt[idx]), -1)).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return scorer.eval()


@torch.inference_mode()
def evaluate(policies, val_eps, model, batcher, bases, update, seed, scorer):
    by_id = {e["id"]: e for e in val_eps}
    fam = {}
    for e in val_eps:
        fam.setdefault(e["family_id"], []).append(e["id"])
    memories = {"recent_only": {e["id"]: [] for e in val_eps},
                "marker": {e["id"]: [e["marker"][0], e["marker"][-1]] for e in val_eps}}
    for policy in policies:
        rng = np.random.default_rng(seed + 7)
        memories[policy] = {e["id"]: rollout(e, batcher.p[e["id"]], policy, rng, update, scorer) for e in val_eps}
    out = {}
    for name, mem in memories.items():
        items = [(by_id[i], m) for i, m in mem.items()]
        pred, target = P.predict(model, batcher, items, bases)
        res = {i: (pred[k], target[k]) for k, i in enumerate(mem)}
        row = {"marker_hit": float(np.mean([any(t in by_id[i]["marker"] for t in m) for i, m in mem.items()]))}
        for hi, h in enumerate(P.HORIZONS):
            mse = {i: float(((v[0][hi] - v[1][hi]) ** 2).mean()) for i, v in res.items()}
            ident = {}
            for ids in fam.values():
                truths = np.stack([res[i][1][hi] for i in ids])
                for i in ids:
                    ident[i] = float(ids[int(((truths - res[i][0][hi][None]) ** 2).mean(1).argmin())] == i)
            row[f"h{h}"] = {"mse": float(np.mean(list(mse.values()))),
                            "sibling_id": float(np.mean(list(ident.values()))),
                            "per_class_sibling_id": {c: float(np.mean([ident[i] for i in ident if by_id[i]["cog"] == c]))
                                                     for c in sorted({e["cog"] for e in val_eps})}}
        out[name] = row
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--predictors", type=Path, required=True, help="folder with seed_<n>.pt from the predictor script")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--states-per-episode", type=int, default=12)
    parser.add_argument("--dagger-rounds", type=int, default=2)
    parser.add_argument("--max-train-episodes", type=int, default=0, help="0 = all (for smoke tests)")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--checkpoint-dir", type=Path,
                        help="where to keep per-round states and scorers (default: <predictors>/../<output stem>_ckpt)")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    update = load_online_rules()
    eps = P.load(args.data, args.features)
    train_eps = [e for e in eps if e["split"] == "train"]
    if args.max_train_episodes:
        train_eps = train_eps[:args.max_train_episodes]
    val_eps = [e for e in eps if e["split"] == "validation"]
    # Intermediate state, so a killed job keeps its work and a resubmission resumes:
    #   <output stem>_ckpt/seed<n>_round<r>.npz   labelled states (x, y) after each training round
    #   <output stem>_ckpt/seed<n>_round<r>.pt    the scorer trained on them
    #   <output>                                  rewritten after every seed ("complete": false until the end)
    # Kept under build/ (next to the predictor checkpoints), never in the repo: states are ~GB per seed.
    ckpt = args.checkpoint_dir or args.predictors.parent / f"{args.output.stem}_ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)
    results, log = {}, {}
    if args.output.exists():
        previous = json.loads(args.output.read_text())
        results = {s: v for s, v in previous.get("per_seed", {}).items() if int(s) in args.seeds}
        log = {s: v for s, v in previous.get("training_log", {}).items() if s in results}

    def write_output(complete):
        done = [s for s in args.seeds if str(s) in results]
        names = list(results[str(done[0])]) if done else []
        summary = {k: {"marker_hit": float(np.mean([results[str(s)][k]["marker_hit"] for s in done])),
                       **{f"h{h}": {m: float(np.mean([results[str(s)][k][f"h{h}"][m] for s in done]))
                                    for m in ("mse", "sibling_id")} for h in P.HORIZONS}} for k in names}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "completed_utc": datetime.now(timezone.utc).isoformat(), "complete": complete,
            "seeds_done": done, "data": args.data.name,
            "predictors": str(args.predictors), "seeds": args.seeds,
            "counts": {"train_episodes": len(train_eps), "validation_episodes": len(val_eps)},
            "behaviours": BEHAVIOURS, "states_per_episode": args.states_per_episode, "dagger_rounds": args.dagger_rounds,
            "teacher": "target-time deletion cost: frozen predictor error at query+6/+12 with the two kept frames",
            "notes": {"marker": "privileged reference (marker indices from evaluation_only)",
                      "prediction_surprise": "not included (needs a running forecast stream)",
                      "resume": "a resumed seed reuses saved rounds; its later sampling differs from an uninterrupted run"},
            "summary_mean_over_seeds": summary, "per_seed": results, "training_log": log,
        }, indent=2) + "\n")
        tmp.replace(args.output)
        return names, summary

    for seed in args.seeds:
        if str(seed) in results:
            print(f"seed {seed}: already evaluated, skipping", flush=True)
            continue
        t0 = time.perf_counter()
        ck = torch.load(args.predictors / f"seed_{seed}.pt", map_location="cpu", weights_only=False)
        model = P.Predictor()
        model.load_state_dict(ck["model"])
        model.eval()
        bases = ck["bases"]
        batcher = P.Batcher(train_eps + val_eps, bases)
        rng = np.random.default_rng(seed)
        rounds = []
        x = y = scorer = None
        paths = lambda r: (ckpt / f"seed{seed}_round{r}.npz", ckpt / f"seed{seed}_round{r}.pt")
        # Only the latest round is kept (its states include every earlier round's); resume from it.
        saved = [r for r in range(args.dagger_rounds + 1) if all(p.exists() for p in paths(r))]
        start = 0
        if saved:
            last = max(saved)
            with np.load(paths(last)[0]) as f:
                x, y = f["x"].astype(np.float32), f["y"]
            scorer = Scorer()
            scorer.load_state_dict(torch.load(paths(last)[1], map_location="cpu"))
            scorer.eval()
            rounds.append({"round": last, "states": len(x), "seconds": time.perf_counter() - t0, "resumed": True})
            print(f"seed {seed} round {last}: resumed from checkpoint ({len(x)} states)", flush=True)
            start = last + 1
        for r in range(start, args.dagger_rounds + 1):
            data_path, model_path = paths(r)
            if r == 0:
                states = []
                for policy in BEHAVIOURS:
                    states += collect_states(train_eps, batcher, policy, rng, update, args.states_per_episode)
                x, y = label(states, train_eps, model, batcher, bases)
            else:
                new = collect_states(train_eps, batcher, "deletion_cost", rng, update, args.states_per_episode, scorer)
                xn, yn = label(new, train_eps, model, batcher, bases)
                x, y = np.concatenate([x, xn]), np.concatenate([y, yn])
            scorer = fit_scorer(x, y, seed)
            # Write the new round fully before removing the previous one; float16 halves the disk use.
            with open(data_path.with_suffix(".tmp"), "wb") as f:
                np.savez(f, x=x.astype(np.float16), y=y)
            data_path.with_suffix(".tmp").replace(data_path)
            torch.save(scorer.state_dict(), model_path)
            for old in range(r):
                for p in paths(old):
                    p.unlink(missing_ok=True)
            rounds.append({"round": r, "states": len(x), "seconds": time.perf_counter() - t0})
            print(f"seed {seed} round {r}: {len(x)} states, {time.perf_counter() - t0:.0f}s", flush=True)
        results[str(seed)] = evaluate(EVAL_POLICIES, val_eps, model, batcher, bases, update, seed, scorer)
        log[str(seed)] = rounds
        write_output(complete=False)
        r = results[str(seed)]
        print(f"seed {seed} ({time.perf_counter() - t0:.0f}s): " + "  ".join(
            f"{k} mse {v['h12']['mse']:.3f} id {v['h12']['sibling_id']:.2f} hit {v['marker_hit']:.2f}" for k, v in r.items()), flush=True)
    names, summary = write_output(complete=True)
    for k in names:
        s = summary[k]
        print(f"{k:18s} h12 mse {s['h12']['mse']:.4f} sibling_id {s['h12']['sibling_id']:.3f} | "
              f"h6 mse {s['h6']['mse']:.4f} | marker hit {s['marker_hit']:.2f}")


if __name__ == "__main__":
    main()
