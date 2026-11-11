"""FrameMorrow on the terminal (Synergy) tasks: the released selector ranks the three cards and
the lowest is deleted. Moved unchanged from experiments/memory_baselines/synergy_run.py.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from ...environments.synergy.assets import frame
from .reference import FrameMorrow, SelectorConfig, ranking_loss, teacher_scores


def fm_rows(store, records):
    rows = []
    for r in records:
        q, cap = r["query"], store.capsule(r)
        rows.append({"history": cap["pz"][4:], "recent": cap["pz"][:4],
                     "condition": store.actions[r["id"]][q:q + 6].astype(np.float32),
                     "future": np.stack([frame(store, r, t) for t in range(q + 1, q + 7)])})
    return rows


def fm_collate(rows, keys=("history", "recent", "condition", "future")):
    out = {}
    for key in keys:
        out[key] = pad_sequence([torch.as_tensor(r[key]) for r in rows], batch_first=True)
        out[key + "_mask"] = torch.ones(out[key].shape[:2], dtype=torch.bool)
    return out


def fm_student(batch):
    return {k: batch[k] for k in ("history", "recent", "condition", "history_mask", "recent_mask", "condition_mask")}


def framemorrow(train_store, train, conf_store, conf, seed, epochs=30, batch=64, hidden=512, layers=4):
    rng = np.random.default_rng(seed)
    families = sorted({r["family"] for r in train})
    held = set(rng.permutation(families)[:max(1, len(families) // 10)].tolist())
    fit = fm_rows(train_store, [r for r in train if r["family"] not in held])
    val = fm_collate(fm_rows(train_store, [r for r in train if r["family"] in held]))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        selector = FrameMorrow(SelectorConfig(visual_dim=128, condition_dim=2, hidden_dim=hidden, layers=layers,
                                              heads=8, ffn_dim=4 * hidden, queries=4, modality="action"))
    opt = torch.optim.AdamW(selector.parameters(), lr=1e-4, weight_decay=0.01, betas=(0.9, 0.999))
    per_epoch = math.ceil(len(fit) / batch)
    updates = per_epoch * epochs
    warmup = max(1, math.ceil(updates * 0.05))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warmup if s < warmup else
                                              0.5 * (1 + math.cos(math.pi * min(1, (s - warmup) / max(1, updates - warmup)))))

    def loss(b):
        target = teacher_scores(b["history"], b["future"], b["history_mask"], b["future_mask"])
        return ranking_loss(selector(**fm_student(b)), target, b["history_mask"])

    best, best_state = math.inf, None
    for _ in range(epochs):
        selector.train()
        order = rng.permutation(len(fit))
        for i in range(per_epoch):
            opt.zero_grad(set_to_none=True)
            loss(fm_collate([fit[j] for j in order[i * batch:(i + 1) * batch]]))["loss"].backward()
            torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0)
            opt.step(); sched.step()
        selector.eval()
        with torch.inference_mode():
            listwise = float(loss(val)["listwise"])
        if listwise < best:
            best, best_state = listwise, copy.deepcopy(selector.state_dict())
    selector.load_state_dict(best_state)
    selector.eval()
    drops = []
    with torch.inference_mode():
        rows = fm_rows(conf_store, conf)
        for a in range(0, len(rows), 512):
            b = fm_collate(rows[a:a + 512], keys=("history", "recent", "condition"))
            drops.append(selector(**fm_student(b)).numpy().argmin(1))
    return np.concatenate(drops)


def framemorrow_teacher(conf_store, conf):
    b = fm_collate(fm_rows(conf_store, conf))
    return teacher_scores(b["history"], b["future"]).numpy().argmin(1)
