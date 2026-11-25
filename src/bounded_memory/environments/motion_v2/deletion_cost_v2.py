"""Deletion-cost scorer, development version 2, on Motion v2 (not the registered step-3/4 recipe).

What stays from the proposal: a causal scorer reads only the three resident candidates and the four
recent frames, and is trained offline to tell which deletion costs the frozen predictor least; at
deployment the candidate with the highest deletion logit is dropped, irrevocably (B = 2, R = 4).

What changes from the registered recipe (step34_students.py), and why:

    teacher    optimal-continuation deletion cost. After deleting candidate j at clock t, every pair of
               (the two residents left) U (frames still to arrive before the query) remains reachable,
               so cost_j = the lowest +3/+6 position loss of the frozen readers over those pairs. The
               registered labels assumed the attention policy decides afterwards. One all-pairs loss table
               per training episode gives the label of any state, and scores any policy's final pair.
    prior      the readers' own pooling salience of each candidate (what the attention baseline keeps)
               enters the scorer, which is a zero-initialised residual on it: step 0 is exactly the
               attention policy (IndexMem: zero-initialised new branches start at the base policy).
    student    width-256 MLP with LayerNorm/GELU and set pooling; pairwise loss weighted by the cost gap;
               AdamW (0.9, 0.95), warmup-stable-decay schedule.
    selection  the checkpoint with the lowest online position loss on a dev split of training scene
               pairs, step 0 included (IndexMem: never select by training loss). Validation and the
               confirmation set are never read while training or selecting.

Labels use the training episodes' future positions (evaluation-only metadata) through the readers'
losses, as the registered labels did; the scorer itself only ever sees observable inputs.
"""
from __future__ import annotations

import hashlib
import itertools
import math

import numpy as np
import torch
from torch import nn

from . import step34_core as core

HORIZONS = np.array([3, 6])
ITEM_DIM, CONTEXT_DIM = 259, 524          # step34_core.causal_features
BEHAVIORS = ("fifo", "reservoir", "diversity", "temporal_buckets", "attention")


def dev_pairs(records, modulus=10):
    """Training scene pairs held out for checkpoint selection (sha256 of the pair id, 1 in `modulus`)."""
    return {r["pair"] for r in records if r["split"] == "train"
            and int(hashlib.sha256(r["pair"].encode()).hexdigest(), 16) % modulus == 0}


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
    for policy in BEHAVIORS:
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
    """Warmup-stable-decay multiplier (IndexMem++ recipe shape)."""
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


def table_loss(tables, episodes, pairs):
    """Mean +3 and +6 position loss of final pairs, looked up in the all-pairs tables."""
    values = np.array([tables[ep["rec"]["id"]][:, a, b] for ep, (a, b) in zip(episodes, pairs)])
    return values.mean(0)


def train(data, tables, dev_episodes, assets, seed, device, steps=3000, batch=256, lr=1e-3, width=256,
          eval_every=250, log=print):
    """data: items [N,3,259], context [N,524], prior [N,3], cost [N,3] (training pairs only)."""
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
            dev = table_loss(tables, dev_episodes, rollout(assets, dev_episodes, model, norm, device))
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
    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best[2])
    return model.eval(), norm, {"selected_step": best[1], "selected_dev_mean": best[0], "history": history,
                                "label_scale": scale, "states": int(len(items)), "steps": steps, "batch": batch,
                                "lr": lr, "width": width, "dev_episodes": len(dev_episodes),
                                "final_state": final_state}
