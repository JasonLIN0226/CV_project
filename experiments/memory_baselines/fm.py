"""FrameMorrow (Yin et al., arXiv 2609.38839) on Gate.

The selector, the future-similarity teacher and the ranking losses are the official release
(framemorrow_ref/, MIT), used unchanged. This file only builds the training tuples and
applies the selector to the project's episodes.

  kept     Selector: causal transformer, 4 layers / 512 wide / 8 heads, 4 prospective tokens,
           smooth-max scoring (Eq. 3-4, Table 7). Teacher: cosine similarity between each
           history frame and the realised continuation, smooth-max over the continuation
           (Eq. 5). Loss: listwise + 0.5 pairwise with the paper's temperatures (Eq. 6-8,
           Table 8). AdamW 1e-4, weight decay 0.01, cosine schedule with 5% warmup, batch 64,
           gradient clip 1.0, checkpoint chosen by held-out listwise loss (Table 9, train.py).
  changed  Condition tokens are the raw 2-d actions with a learned projection. The paper
           serialises actions as text and encodes them with CLIP.
  changed  Continuation = the next 6 frames (the predictor's longest horizon), not 8 frames
           over 4 seconds.
  option   `visual`: "dinov2" uses DINOv2 ViT-B/14 class tokens for selector and teacher, as
           published. "ijepa" uses the project's frozen I-JEPA features instead.
  note     As published, the selector ranks the WHOLE eligible history at the query
           (`published`). That needs every past frame to still exist. `online` is the same
           selector applied at each eviction to the three resident candidates, with only
           already-executed actions as its condition; this is not in the paper.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from .framemorrow_ref import FrameMorrow, SelectorConfig, ranking_loss, teacher_scores
from .gate_data import B, R

FUTURE = 6      # continuation frames seen only by the teacher
CONDITION = 6   # action tokens


def _condition(e, t, variant):
    if variant == "published":
        return e.actions[t:t + CONDITION]              # the known next-rollout actions
    past = e.actions[max(0, t - CONDITION):t]          # online: executed actions only
    return np.concatenate([np.zeros((CONDITION - len(past), 2), np.float32), past])


def _tuple(e, v, t, variant, rng=None, subset=False):
    eligible = np.arange(t - R + 1)
    if subset:  # the pool an online policy would hold: the arriving frame and two older ones
        older = rng.choice(eligible[:-1], size=B, replace=False)
        eligible = np.sort(np.concatenate([older, eligible[-1:]]))
    return {"history": v[eligible], "times": eligible, "recent": v[t - R + 1:t + 1],
            "condition": _condition(e, t, variant), "future": v[t + 1:t + 1 + FUTURE]}


def _collate(rows, device, with_future=True):
    out = {}
    for key in ("history", "recent", "condition") + (("future",) if with_future else ()):
        seqs = [torch.as_tensor(np.asarray(r[key], np.float32)) for r in rows]
        out[key] = pad_sequence(seqs, batch_first=True).to(device)
        lengths = torch.tensor([len(s) for s in seqs])
        out[key + "_mask"] = (torch.arange(out[key].shape[1])[None] < lengths[:, None]).to(device)
    return out


def _student_kwargs(batch):
    return {k: batch[k] for k in ("history", "recent", "condition", "history_mask", "recent_mask", "condition_mask")}


def _loss(selector, batch):
    target = teacher_scores(batch["history"], batch["future"], batch["history_mask"], batch["future_mask"])
    return ranking_loss(selector(**_student_kwargs(batch)), target, batch["history_mask"])


def train(train_episodes, visual, seed, device, variant="published", epochs=10, batch=64, lr=1e-4,
          hidden=512, layers=4, heads=8):
    """`visual` maps episode id -> [T, Dv] frozen visual features. The predictor is never used."""
    if variant not in ("published", "online"):
        raise ValueError(variant)
    rng = np.random.default_rng(seed)
    pairs = sorted({e.pair for e in train_episodes})
    held = set(rng.permutation(pairs)[:max(1, len(pairs) // 10)].tolist())  # 90:10 by scene pair
    fit = [e for e in train_episodes if e.pair not in held]
    val = [e for e in train_episodes if e.pair in held]
    index = [(e, t) for e in fit for t in range(R + B, e.q + 1)]
    val_rng = np.random.default_rng(seed + 1)
    val_rows = [_tuple(e, visual[e.id], t, variant, val_rng, subset=(variant == "online" and i % 2 == 1))
                for i, (e, t) in enumerate((e, t) for e in val for t in range(R + B, e.q + 1))]
    dim = visual[train_episodes[0].id].shape[1]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        selector = FrameMorrow(SelectorConfig(visual_dim=dim, condition_dim=2, hidden_dim=hidden, layers=layers,
                                              heads=heads, ffn_dim=4 * hidden, queries=4, modality="action"))
    selector = selector.to(device)
    opt = torch.optim.AdamW(selector.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.999))
    per_epoch = math.ceil(len(index) / batch)
    updates = per_epoch * epochs
    warmup = max(1, math.ceil(updates * 0.05))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warmup if s < warmup else
                                              0.5 * (1 + math.cos(math.pi * min(1, (s - warmup) / max(1, updates - warmup)))))

    @torch.inference_mode()
    def validate():
        selector.eval()
        total = 0.0
        for i in range(0, len(val_rows), 256):
            rows = val_rows[i:i + 256]
            total += float(_loss(selector, _collate(rows, device))["listwise"]) * len(rows)
        return total / len(val_rows)

    best, best_state, history = math.inf, None, []
    for epoch in range(epochs):
        selector.train()
        order = rng.permutation(len(index))
        for i in range(per_epoch):
            rows = []
            for j in order[i * batch:(i + 1) * batch]:
                e, t = index[j]
                rows.append(_tuple(e, visual[e.id], t, variant, rng, subset=(variant == "online" and rng.random() < 0.5)))
            opt.zero_grad(set_to_none=True)
            loss = _loss(selector, _collate(rows, device))["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0, error_if_nonfinite=True)
            opt.step(); sched.step()
        listwise = validate()
        history.append(listwise)
        if listwise < best:
            best, best_state = listwise, copy.deepcopy(selector.state_dict())
    selector.load_state_dict(best_state)
    return selector.eval(), {"variant": variant, "tuples_per_epoch": len(index), "updates": updates,
                             "held_out_pairs": len(held), "held_out_listwise": history,
                             "best_epoch": int(np.argmin(history)) + 1,
                             "parameters": sum(p.numel() for p in selector.parameters())}


@torch.inference_mode()
def select_published(selector, episodes, visual, device, chunk=128):
    """Top-B over the whole eligible history at the query, as in the paper."""
    out = []
    for i in range(0, len(episodes), chunk):
        eps = episodes[i:i + chunk]
        rows = [_tuple(e, visual[e.id], e.q, "published") for e in eps]
        batch = _collate(rows, device, with_future=False)
        times = pad_sequence([torch.as_tensor(r["times"], dtype=torch.float32) for r in rows], batch_first=True).to(device)
        chosen, _ = selector.select(batch["history"], batch["recent"], batch["condition"], times, k=B,
                                    history_mask=batch["history_mask"], recent_mask=batch["recent_mask"],
                                    condition_mask=batch["condition_mask"])
        out.extend(sorted(int(r["times"][j]) for j in idx.tolist()) for idx, r in zip(chosen, rows))
    return out


@torch.inference_mode()
def select_online(selector, episodes, visual, device):
    """Irrevocable replay: at each arrival the selector ranks the three residents and the lowest is dropped."""
    kept = {e.id: [] for e in episodes}
    for t in range(R, max(e.q for e in episodes) + 1):
        live = [e for e in episodes if e.q >= t]
        full, rows = [], []
        for e in live:
            pool = kept[e.id] + [t - R]
            if len(pool) <= B:
                kept[e.id] = pool
                continue
            v = visual[e.id]
            rows.append({"history": v[pool], "recent": v[t - R + 1:t + 1], "condition": _condition(e, t, "online")})
            full.append((e, pool))
        if not rows:
            continue
        scores = selector(**_student_kwargs(_collate(rows, device, with_future=False))).cpu().numpy()
        for (e, pool), s in zip(full, scores):
            evict = min(range(len(pool)), key=lambda j: (s[j], pool[j]))
            kept[e.id] = [f for j, f in enumerate(pool) if j != evict]
    return [sorted(kept[e.id]) for e in episodes]


@torch.inference_mode()
def select_teacher(episodes, visual, device):
    """Diagnostic: top-B by the teacher's own future-similarity score (reads the realised future)."""
    out = []
    for e in episodes:
        row = _tuple(e, visual[e.id], e.q, "published")
        batch = _collate([row], device)
        s = teacher_scores(batch["history"], batch["future"]).cpu().numpy()[0]
        out.append(sorted(np.argsort(-s, kind="stable")[:B].tolist()))
    return out
