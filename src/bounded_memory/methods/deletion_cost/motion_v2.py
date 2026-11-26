"""Our deletion-cost scorer on Motion v2, development version 2 (not the registered step-3/4 recipe).

What stays from the proposal: a causal scorer reads only the three resident candidates and the four
recent frames, and is trained offline to tell which deletion costs the frozen predictor least; at
deployment the candidate with the highest deletion logit is dropped, irrevocably (B = 2, R = 4).

What changes from the registered recipe (environments/motion_v2/step34_students.py), and why:

    teacher    optimal-continuation deletion cost. After deleting candidate j at clock t, every pair of
               (the two residents left) U (frames still to arrive before the query) remains reachable,
               so cost_j = the lowest +3/+6 position loss of the frozen readers over those pairs (or a soft
               minimum, `temperature`). The registered labels assumed the attention policy decides
               afterwards. One all-pairs loss table per episode gives the label of any state and scores
               any policy's final pair.
    prior      the readers' own pooling salience of each candidate (what the attention baseline keeps)
               enters the scorer, which is a zero-initialised residual on it: step 0 is exactly the
               attention policy (a zero-initialised head adds nothing until training moves it).
    student    width-256 MLP with LayerNorm/GELU and set pooling; pairwise loss weighted by the cost gap;
               AdamW (0.9, 0.95), warmup-stable-decay schedule.
    states     the behaviour policies' states, and with `rounds` > 0 the states the trained scorer visits
               itself, labelled from the tables and added before retraining (DAgger-style).
    selection  the checkpoint with the lowest online position loss on a dev split, step 0 included
               (training loss is never used to select: it rewards fitting the labels, not choosing well). Validation is
               never read while training or
               selecting.

The training episodes are either the registered ones (which the frozen readers memorised) or fresh scenes
from the same generator (environments/motion_v2/fresh.py). Labels use the episodes' future positions
(evaluation-only metadata) through the readers' losses, as the registered labels did; the scorer itself
only ever sees observable inputs.

    VARIANTS       the four label/state settings compared on fresh dev (label temperature, on-policy rounds)
    pair_table     all-pairs loss table of one episode (the teacher); load_tables reads them back
    fit_variant    train one variant for one seed, returning every round and the best one on dev
    rollout        online replay of a scorer (step 0 is the attention policy)
    holdout_comparison / validation_allowed / validation_scores
                   the check of the chosen scorers on the fresh holdout, the rule that guards the one
                   validation look, and that look
"""
from __future__ import annotations

import itertools
import math
import os
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch import nn

from ...environments.motion_v2 import episodes as motion_episodes
from ...environments.motion_v2 import frozen
from ...environments.motion_v2 import step34_core as core
from ...environments.motion_v2.episodes import HORIZONS
from ...evaluation.metrics import motion_v2 as metrics
ITEM_DIM, CONTEXT_DIM = 259, 524          # step34_core.causal_features
BEHAVIOURS = ("fifo", "reservoir", "diversity", "temporal_buckets", "attention")
VARIANTS = {                     # name: (label temperature in mean +3/+6 position MSE, on-policy rounds after the first)
    "min": (0.0, 0),
    "min_dagger": (0.0, 3),
    "soft": (0.002, 0),
    "soft_dagger": (0.002, 3),
}
BEHAVIOUR_SEED = 20261124        # random choices of the behaviour policies (reservoir, ties)


def identity_norm():
    """Input normalisation that leaves features unchanged: with it, an untrained ScorerV2 is the attention policy."""
    return {"items_mean": np.zeros(ITEM_DIM), "items_std": np.ones(ITEM_DIM),
            "context_mean": np.zeros(CONTEXT_DIM), "context_std": np.ones(CONTEXT_DIM)}


def load_tables(folder, episodes):
    """{episode id: all-pairs table} written by scripts/generate_teacher_labels/motion_v2.py."""
    missing = [ep["rec"]["id"] for ep in episodes if not (Path(folder) / f"{ep['rec']['id']}.npy").exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} loss tables missing in {folder} (first: {missing[0]}); "
                                "run scripts/generate_teacher_labels/motion_v2.py first")
    return {ep["rec"]["id"]: np.load(Path(folder) / f"{ep['rec']['id']}.npy") for ep in episodes}


@torch.inference_mode()
def pair_table(assets, ep, chunk=512):
    """[2 horizons, n, n] reader-mean position loss of every legal memory pair (n = q - 3); inf on the diagonal."""
    q = ep["rec"]["query"]
    n = q - 3
    pairs = list(itertools.combinations(range(n), 2))
    out = np.full((2, n, n), np.inf)
    truth = ep["meta"]["states"][q + HORIZONS].astype(np.float64)
    for at in range(0, len(pairs), chunk):
        block = pairs[at:at + chunk]
        batch = assets.collate([assets.sample(ep, q, list(p)) for p in block], assets.device)
        loss = 0.0
        for model in assets.models:
            predicted = model(batch).cpu().numpy()
            decoded = assets.decoder.predict(predicted.reshape(-1, predicted.shape[-1])).reshape(-1, 2, 2)
            loss = loss + np.square(decoded - truth).mean(-1) / len(assets.models)
        for (a, b), value in zip(block, loss):
            out[:, a, b] = out[:, b, a] = value
    return out


def continuation_cost(table, clock, pool, q, temperature=0.0):
    """cost[j]: best reachable mean(+3, +6) loss after deleting pool[j] at `clock`.

    temperature > 0 replaces the minimum over the reachable pairs by a soft minimum,
    low - T log mean exp(-(loss - low) / T): between the minimum (T -> 0) and the mean (T -> inf). The plain
    minimum rewards whichever of ~200 pairs happens to give the readers their luckiest prediction; the soft
    minimum credits a frame that is part of many good pairs.
    """
    future = list(range(clock - 3, q - 3))
    mean = table.mean(0)
    cost = np.empty(3)
    for j in range(3):
        reach = np.array([t for i, t in enumerate(pool) if i != j] + future)
        sub = mean[np.ix_(reach, reach)]
        if temperature <= 0:
            cost[j] = sub.min()
        else:
            values = sub[np.triu_indices(len(reach), 1)]
            low = values.min()
            cost[j] = low - temperature * np.log(np.mean(np.exp(-(values - low) / temperature)))
    return cost


def salience(assets, ep, clock, pool):
    """Mean over the three readers of their pooling softmax over the candidates (the attention policy's score)."""
    return core.mean_attention(assets.attention_tables(ep)[:, clock, list(pool)])


def observe(assets, ep, clock, pool):
    """Observable scorer inputs at one state: items [3, 259], context [524], log-salience prior [3]."""
    items, context = core.causal_features(ep["p"], ep["prefix"], clock, list(pool))
    prior = np.log(np.clip(salience(assets, ep, clock, pool), 1e-12, None)).astype(np.float32)
    return items, context, prior


def behaviour_states(assets, ep, seed):
    """Every full-pool state (clock, pool) visited by the behaviour policies on one episode."""
    q, states = ep["rec"]["query"], []
    for policy in BEHAVIOURS:
        rng = np.random.default_rng(core.seed_for(seed, ep["rec"]["id"], policy))
        kept = []
        for clock in range(4, q + 1):
            pool = kept + [clock - 4]
            scores, vectors = {}, {}
            if len(pool) == 3:
                states.append((clock, tuple(pool)))
                if policy == "attention":
                    scores = dict(zip(pool, salience(assets, ep, clock, pool)))
                elif policy == "diversity":
                    vectors = {t: ep["p"][t].copy() for t in pool}
            kept = core.observable_update(policy, kept, clock - 4, clock, vectors, ep["p"][clock], scores, rng)
    return sorted(set(states))


class ScorerV2(nn.Module):
    """Deletion logit = -alpha * log salience + zero-initialised residual MLP (higher = delete)."""

    def __init__(self, width=256):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(ITEM_DIM + CONTEXT_DIM, width), nn.LayerNorm(width), nn.GELU(),
                                 nn.Linear(width, width), nn.GELU())
        self.head = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.alpha = nn.Parameter(torch.ones(()))

    def forward(self, items, context, prior):
        h = self.phi(torch.cat([items, context[:, None].expand(-1, 3, -1)], -1))
        h = torch.cat([h, h.mean(1, keepdim=True).expand(-1, 3, -1)], -1)
        return -self.alpha * prior + self.head(h).squeeze(-1)


def fit_normalization(items, context):
    flat = items.reshape(-1, items.shape[-1])
    return {"items_mean": flat.mean(0), "items_std": flat.std(0) + 1e-4,
            "context_mean": context.mean(0), "context_std": context.std(0) + 1e-4}


def normalize(norm, items, context):
    return ((items - norm["items_mean"]) / norm["items_std"]).astype(np.float32), \
           ((context - norm["context_mean"]) / norm["context_std"]).astype(np.float32)


def pairwise_loss(logits, cost):
    """Delete i before j when cost_i < cost_j, weighted by the cost gap (costs in train-scale units)."""
    gap = torch.relu(cost[:, None, :] - cost[:, :, None])            # [b, i, j] = cost_j - cost_i if positive
    margin = logits[:, :, None] - logits[:, None, :]                  # logit_i - logit_j
    return (gap * nn.functional.softplus(-margin)).sum() / gap.sum().clamp_min(1e-8)


def schedule(step, steps, warmup=0.1, stable=0.6, floor=5e-3):
    """Learning-rate multiplier: linear warmup, a constant stretch, then cosine decay to `floor`."""
    if step < warmup * steps:
        return (step + 1) / (warmup * steps)
    if step < (warmup + stable) * steps:
        return 1.0
    frac = (step - (warmup + stable) * steps) / max(1.0, (1 - warmup - stable) * steps)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, frac)))


@torch.inference_mode()
def rollout(assets, episodes, model, norm, device, record=False):
    """Online replay of the scorer on episodes; returns the final kept pair of each.

    record=True also returns the full-pool states (clock, pool) the scorer itself visited, per episode
    (on-policy states for DAgger-style aggregation)."""
    model.eval()
    kept = {ep["rec"]["id"]: [] for ep in episodes}
    visited = {ep["rec"]["id"]: [] for ep in episodes}
    for clock in range(4, max(ep["rec"]["query"] for ep in episodes) + 1):
        live = [ep for ep in episodes if ep["rec"]["query"] >= clock]
        full, rows = [], []
        for ep in live:
            pool = kept[ep["rec"]["id"]] + [clock - 4]
            if len(pool) < 3:
                kept[ep["rec"]["id"]] = pool
            else:
                full.append((ep, pool))
                rows.append(observe(assets, ep, clock, pool))
                visited[ep["rec"]["id"]].append((clock, tuple(pool)))
        if not rows:
            continue
        items, context = normalize(norm, np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows]))
        logits = model(torch.as_tensor(items, device=device), torch.as_tensor(context, device=device),
                       torch.as_tensor(np.stack([r[2] for r in rows]), device=device)).cpu().numpy()
        for (ep, pool), logit in zip(full, logits):
            drop = int(np.argmax(logit))                                 # ties: the first (oldest) candidate
            kept[ep["rec"]["id"]] = sorted(t for i, t in enumerate(pool) if i != drop)
    final = [kept[ep["rec"]["id"]] for ep in episodes]
    return (final, visited) if record else final


def state_rows(assets, ep, table, states, temperature=0.0):
    """Scorer inputs and deletion-cost labels of the given (clock, pool) states of one episode."""
    q, out = ep["rec"]["query"], {"items": [], "context": [], "prior": [], "cost": []}
    for clock, pool in states:
        x, c, p = observe(assets, ep, clock, pool)
        out["items"].append(x); out["context"].append(c); out["prior"].append(p)
        out["cost"].append(continuation_cost(table, clock, pool, q, temperature))
    return out


@torch.inference_mode()
def state_metrics(model, norm, data, device, informative=1e-5):
    """Per-state quality of a scorer on labelled states: mean regret of its deletion against the cheapest one,
    and how often it picks the cheapest, over all states and over states whose costs differ."""
    model.eval()
    items, context = normalize(norm, data["items"], data["context"])
    logits = model(torch.as_tensor(items, device=device), torch.as_tensor(context, device=device),
                   torch.as_tensor(data["prior"].astype(np.float32), device=device)).cpu().numpy()
    cost = data["cost"]
    chosen = cost[np.arange(len(cost)), logits.argmax(1)]
    regret = chosen - cost.min(1)
    hit = regret <= 1e-12
    useful = (cost.max(1) - cost.min(1)) > informative
    return {"states": int(len(cost)), "informative_share": float(useful.mean()), "regret": float(regret.mean()),
            "top1": float(hit.mean()), "regret_informative": float(regret[useful].mean()) if useful.any() else 0.0,
            "top1_informative": float(hit[useful].mean()) if useful.any() else 1.0}


def train(data, tables, dev_episodes, assets, seed, device, steps=3000, batch=256, lr=1e-3, width=256,
          eval_every=250, log=print):
    """Train one scorer; data: items [N,3,259], context [N,524], prior [N,3], cost [N,3] (training states only).

    Returns (model, norm, meta, final): the checkpoint with the lowest dev loss (step 0 included), its input
    normalisation, the training record, and the weights at the last step (for rolling out on-policy states)."""
    norm = fit_normalization(data["items"], data["context"])
    items, context = normalize(norm, data["items"], data["context"])
    centered = data["cost"] - data["cost"].mean(1, keepdims=True)
    scale = float(np.sqrt(np.mean(centered ** 2))) or 1.0
    tensors = {k: torch.as_tensor(v, device=device) for k, v in
               {"items": items, "context": context, "prior": data["prior"].astype(np.float32),
                "cost": (data["cost"] / scale).astype(np.float32)}.items()}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ScorerV2(width)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: schedule(s, steps))
    rng = np.random.default_rng(seed)
    history, best = [], None
    for step in range(steps + 1):
        if step % eval_every == 0 or step == steps:
            dev = metrics.final_pair_losses(tables, dev_episodes, rollout(assets, dev_episodes, model, norm, device)).mean(0)
            history.append({"step": step, "dev_position_+3": float(dev[0]), "dev_position_+6": float(dev[1]),
                            "alpha": float(model.alpha)})
            log(f"step {step:5d} dev +3 {dev[0]:.6f} +6 {dev[1]:.6f} alpha {float(model.alpha):.3f}")
            if best is None or dev.mean() < best[0]:
                best = (float(dev.mean()), step, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            model.train()
        if step == steps:
            break
        idx = torch.as_tensor(rng.integers(0, len(items), batch), device=device)
        loss = pairwise_loss(model(tensors["items"][idx], tensors["context"][idx], tensors["prior"][idx]),
                             tensors["cost"][idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    final = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best[2])
    return model.eval(), norm, {"selected_step": best[1], "selected_dev_mean": best[0], "history": history,
                                "label_scale": scale, "states": int(len(items)), "steps": steps, "batch": batch,
                                "lr": lr, "width": width, "dev_episodes": len(dev_episodes)}, final


def concat(parts):
    """Stack labelled state rows (dicts of items, context, prior, cost); {} when there are none."""
    parts = [p for p in parts if len(p["cost"])]
    if not parts:
        return {}
    return {k: np.concatenate([np.asarray(p[k], np.float64 if k == "cost" else np.float32) for p in parts])
            for k in ("items", "context", "prior", "cost")}


def labelled(assets, episodes, tables, states, temperature=0.0):
    """Labelled rows of the given states ({episode id: [(clock, pool), ...]}) of several episodes."""
    return concat([state_rows(assets, ep, tables[ep["rec"]["id"]], states[ep["rec"]["id"]], temperature)
                   for ep in episodes])


class Best(NamedTuple):
    """The round of fit_variant whose selected checkpoint has the lowest dev loss."""
    dev_mean: float
    round: int
    state: dict
    norm: dict
    width: int


def fit_variant(assets, fit, dev, tables, base_rows, dev_rows, behaviour, temperature, rounds, seed, device,
                steps=3000, log=print):
    """Train on the behaviour-state rows, then `rounds` times add the trained scorer's own states and retrain.

    Returns (history, best): one entry per round (states, selected step and dev loss, the last step's dev loss,
    per-state metrics on the dev rows) and the Best round."""
    seen = {ep["rec"]["id"]: set(behaviour[ep["rec"]["id"]]) for ep in fit}
    data, history, best = base_rows, [], None
    for r in range(rounds + 1):
        model, norm, meta, final_state = train(data, tables, dev, assets, seed, device, steps=steps,
                                               log=lambda m, r=r: log(f"round {r} {m}"))
        final = ScorerV2(meta["width"]).to(device)
        final.load_state_dict(final_state)
        metrics_by_checkpoint = {"selected": state_metrics(model, norm, dev_rows, device),
                                 "final": state_metrics(final, norm, dev_rows, device)}
        last = meta["history"][-1]
        entry = {"round": r, "states": int(len(data["cost"])), "selected_step": meta["selected_step"],
                 "selected_dev_mean": meta["selected_dev_mean"],
                 "final_dev_mean": float(np.mean([last["dev_position_+3"], last["dev_position_+6"]])),
                 "dev_state_metrics": metrics_by_checkpoint}
        history.append(entry)
        log(f"round {r}: selected step {meta['selected_step']} dev {meta['selected_dev_mean']:.6f} final dev "
            f"{entry['final_dev_mean']:.6f} | dev-state top1 (informative) selected "
            f"{metrics_by_checkpoint['selected']['top1_informative']:.3f} "
            f"final {metrics_by_checkpoint['final']['top1_informative']:.3f}")
        if best is None or meta["selected_dev_mean"] < best.dev_mean:
            best = Best(meta["selected_dev_mean"], r, {k: v.cpu() for k, v in model.state_dict().items()}, norm,
                        meta["width"])
        if r == rounds:
            break
        _, visited = rollout(assets, fit, final, norm, device, record=True)
        new = {k: [s for s in v if s not in seen[k]] for k, v in visited.items()}
        for k, v in new.items():
            seen[k].update(v)
        added = labelled(assets, fit, tables, new, temperature)
        log(f"round {r}: +{len(added['cost']) if added else 0} on-policy states")
        if added:
            data = concat([data, added])
    return history, best


def save_scorer(path, best, **info):
    """Write a Best scorer atomically (temporary file, then rename); returns the file's SHA256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save({"model": best.state, "norm": best.norm, "round": best.round, "width": best.width, **info}, tmp)
    os.replace(tmp, path)
    return frozen.sha256(path)


def load_scorer(path, device):
    """(model, norm, sha256) of a scorer written by save_scorer."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = ScorerV2(state["width"]).to(device)
    model.load_state_dict(state["model"])
    return model, state["norm"], frozen.sha256(path)


def attention_policy_losses(assets, episodes, tables, device):
    """[episodes, horizons] final-pair losses of the attention policy (an untrained ScorerV2)."""
    kept = rollout(assets, episodes, ScorerV2().to(device), identity_norm(), device)
    return metrics.final_pair_losses(tables, episodes, kept)


def holdout_comparison(assets, episodes, tables, folder, variants, seeds, device, log=print):
    """Every variant's scorers (one per seed) and the attention policy on the holdout episodes, paired per scene pair.

    Records the SHA256 of every checkpoint it scored, so validation_allowed can require the same files."""
    attention = attention_policy_losses(assets, episodes, tables, device)
    out = {"episodes": len(episodes),
           "attention": {"+3": float(attention[:, 0].mean()), "+6": float(attention[:, 1].mean()),
                         "mean": float(attention.mean())},
           "best_pair_mean": float(np.mean([tables[ep["rec"]["id"]].mean(0).min() for ep in episodes])),
           "variants": {}}
    for name in variants:
        per_seed, hashes = [], {}
        for seed in seeds:
            model, norm, hashes[seed] = load_scorer(Path(folder) / f"{name}_seed{seed}.pt", device)
            per_seed.append(metrics.final_pair_losses(tables, episodes, rollout(assets, episodes, model, norm, device)))
        mean = np.mean(per_seed, 0)
        paired = metrics.paired(mean, attention, episodes)
        out["variants"][name] = {"holdout_mean": float(mean.mean()), "holdout_+3": float(mean[:, 0].mean()),
                                 "holdout_+6": float(mean[:, 1].mean()),
                                 "holdout_mean_per_seed": {s: float(v.mean()) for s, v in zip(seeds, per_seed)},
                                 "paired_with_attention": paired, "checkpoint_sha256": hashes}
        log(f"{name:12s} holdout {mean.mean():.6f} (attention {attention.mean():.6f}) diff {paired['difference']:+.6f} "
            f"95% CI [{paired['ci95'][0]:+.6f}, {paired['ci95'][1]:+.6f}], "
            f"lower in {100 * paired['share_of_pairs_lower']:.0f}% of pairs")
    return out


def validation_allowed(holdout, variant, hashes=None):
    """The one validation look is spent only on a variant whose holdout interval lies wholly below attention, and
    (given `hashes`, seed -> SHA256) only on the very checkpoints the holdout scored."""
    entry = holdout.get("variants", {}).get(variant)
    if entry is None or entry["paired_with_attention"]["ci95"][1] >= 0:
        return False
    recorded = {int(k): v for k, v in entry.get("checkpoint_sha256", {}).items()}
    return hashes is None or (bool(recorded) and recorded == {int(k): v for k, v in hashes.items()})


def validation_scores(assets, paths, folder, variant, seeds, device, raw_dir, log=print):
    """The chosen variant's scorers replayed on both validation intervals and scored by FrozenAssets.losses.

    Saves each seed's per-episode losses and kept pairs to raw_dir/<interval>/<variant>_seed<seed>.npz, so the
    single look can be analysed again without replaying."""
    results = {}
    for interval in frozen.INTERVALS:
        records, external = frozen.interval_inputs(assets, paths, interval)
        raw, _ = motion_episodes.split_episodes(assets, records, "validation", external)
        scores = {}
        for seed in seeds:
            model, norm, _ = load_scorer(Path(folder) / f"{variant}_seed{seed}.pt", device)
            kept = rollout(assets, raw, model, norm, device)
            scores[seed] = motion_episodes.score_frames(assets, raw, kept)
            target = Path(raw_dir) / interval / f"{variant}_seed{seed}.npz"
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez(target, kept=np.asarray(kept), episode_ids=np.array([r["rec"]["id"] for r in raw]), **scores[seed])
        results[interval] = motion_episodes.summarize(scores, list(seeds), matched=False)
        log(f"validation {interval} ({variant}): +3 {results[interval]['position_mse_+3']:.6f} "
            f"+6 {results[interval]['position_mse_+6']:.6f}")
    return results
