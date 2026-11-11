"""Synergy (terminal R/D/S): literature memory baselines on the locked confirmation set.

    python -m experiments.memory_baselines.synergy_run --terminal <run/> --agg <reader run/> \
        --train-xfer <trainvaldev/xfer> --conf-xfer <conf_xfer> --released <unzipped confirmation release> \
        --out results/memory_baselines/synergy

Reference rungs and the team's frozen deletion-cost students are scored from the released
confirmation arrays (no retraining). The three literature baselines are trained on the TRAIN
split and scored on the same 3,456 confirmation records with the same two frozen readers.

Each record is one decision over three candidates, so "ranks the whole history" and "online"
coincide here: the pool is the whole history.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from . import synergy_data as sd
from .framemorrow_ref import FrameMorrow, SelectorConfig, ranking_loss, teacher_scores
from .pacc import Compressor, _schedule
from .pafu import SalienceHead

pipe = sd.pipe
SEEDS = (17, 29, 43)
RELEASED = "results/literature_diagnosis/qualified-reader-retention-20261025"


def frame(store, r, t):
    return store.P[r["rgb_indices"][t]]


def reader_batch(caps, plans, memory=None):
    """Reader inputs for recent frames + either all three candidates or synthetic memory items."""
    n = len(caps)
    recent = {k: torch.from_numpy(np.stack([c[k][:4] for c in caps])) for k in ("pz", "age", "past")}
    if memory is None:
        extra = {k: torch.from_numpy(np.stack([c[k][4:] for c in caps])) for k in ("pz", "age", "past")}
    else:
        extra = dict(zip(("pz", "past", "age"), memory))
    m = 4 + extra["pz"].shape[1]
    return {"z": torch.zeros(n, m, 1), "projected_z": torch.cat([recent["pz"], extra["pz"]], 1),
            "age": torch.cat([recent["age"], extra["age"]], 1),
            "past_actions": torch.cat([recent["past"], extra["past"]], 1),
            "type": torch.tensor([[0] * 4 + [1] * (m - 4)] * n), "mask": torch.ones(n, m, dtype=torch.bool),
            "actions": torch.as_tensor(np.asarray(plans, np.float32)), "current": torch.zeros(n, 1),
            "projected_current": torch.from_numpy(np.stack([c["pcur"] for c in caps]))}


# ---------------------------------------------------------------- PaFu-KV

@torch.inference_mode()
def pafu_teacher(model, store, records, bank="reference", chunk=64):
    """Salience of the three candidates: per attention path, the max over queries placed before / at /
    after each candidate, averaged over paths and over the groups that exist (PaFu Eq. 4-5)."""
    out = np.zeros((len(records), 3), np.float32)
    for a in range(0, len(records), chunk):
        rows, owner, times = [], [], []
        for i, r in enumerate(records[a:a + chunk]):
            q = r["query"]
            cap = store.capsule(r)
            ids = list(r["rgb_indices"][:q + 1]) + [x for plan in r["plan_targets"][bank] for x in plan]
            tt = list(range(q + 1)) + [q + h for _ in r["plan_targets"][bank] for h in pipe.H]
            seen = set()
            for frame_id, t in zip(ids, tt):
                if (frame_id, t) in seen:      # identical frame at the same time gives the same row
                    continue
                seen.add((frame_id, t))
                rows.append((cap, store.P[frame_id])); owner.append(i); times.append(t)
        batch = reader_batch([c for c, _ in rows], np.zeros((len(rows), 6, 2), np.float32))
        batch["projected_current"] = torch.from_numpy(np.stack([f for _, f in rows]).astype(np.float32))
        w = sd.head_attention(model, batch).numpy()            # [rows, paths, 3]
        owner, times = np.asarray(owner), np.asarray(times)
        for i, r in enumerate(records[a:a + chunk]):
            mine = owner == i
            for j in range(3):
                k = r["query"] - 6 + j
                groups = (times[mine] < k, times[mine] == k, times[mine] > k)
                out[a + i, j] = np.mean([w[mine][g][:, :, j].max(0).mean() for g in groups if g.any()])
    return out


def pafu_features(store, r):
    """Each candidate when it leaves the recent window (clock = its time + 4)."""
    out = []
    for j in range(3):
        k = r["query"] - 6 + j
        t = k + 4
        recent = np.stack([frame(store, r, s) for s in range(t - 3, t + 1)])
        out.append(np.concatenate([frame(store, r, k), frame(store, r, k) - frame(store, r, t), recent.mean(0)]))
    return np.stack(out).astype(np.float32)


def pafu(model, train_store, train, conf_store, conf, seed, steps=1000, batch=256):
    y = torch.from_numpy(pafu_teacher(model, train_store, train).reshape(-1))
    x = torch.from_numpy(np.concatenate([pafu_features(train_store, r) for r in train]))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        head = SalienceHead(x.shape[1])
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        idx = torch.as_tensor(rng.integers(0, len(x), size=batch))
        opt.zero_grad(set_to_none=True)
        F.smooth_l1_loss(head(x[idx]), y[idx]).backward()
        opt.step()
    with torch.inference_mode():
        scores = head(torch.from_numpy(np.concatenate([pafu_features(conf_store, r) for r in conf]))).numpy()
    return scores.reshape(-1, 3).argmin(1)  # ties: argmin returns the oldest candidate


# ---------------------------------------------------------------- Compress to Remember

def pacc(model, train_store, train, conf_store, conf, seed, steps=3000, batch=64):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        compressor = Compressor(128)
    opt = torch.optim.AdamW(compressor.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _schedule(s, steps))
    rng = np.random.default_rng(seed)
    caps = [train_store.capsule(r) for r in train]
    reference = train_store.banks["reference"]

    def memory(cs):
        p, age, past = (torch.from_numpy(np.stack([c[k][4:] for c in cs])) for k in ("pz", "age", "past"))
        return compressor.items(compressor.compress(p, past, age, torch.ones(p.shape[:2], dtype=torch.bool)))

    for _ in range(steps):
        cs = [caps[i] for i in rng.integers(0, len(caps), size=batch)]
        plans = reference[rng.integers(0, len(reference), size=batch)]
        with torch.no_grad():
            target = sd.reader_coefficients(model, reader_batch(cs, plans))        # reader with all three frames
        loss = (sd.reader_coefficients(model, reader_batch(cs, plans, memory(cs))) - target).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(compressor.parameters(), 5.0)
        opt.step(); sched.step()
    compressor.eval()
    evaluation = conf_store.banks["evaluation"]
    losses = np.empty(len(conf))
    with torch.inference_mode():
        for a in range(0, len(conf), 64):
            rr = conf[a:a + 64]
            cs = [conf_store.capsule(r) for r in rr]
            rep = [c for c in cs for _ in evaluation]
            plans = np.tile(evaluation, (len(cs), 1, 1))
            coeff = sd.reader_coefficients(model, reader_batch(rep, plans, memory(rep))).numpy().astype(np.float64)
            resid = np.stack([conf_store.target_residual(r, "evaluation") for r in rr])
            losses[a:a + len(rr)] = pipe.qform(coeff.reshape(len(cs), len(evaluation), 2, -1) - resid, conf_store.G).mean(1)
    return losses


# ---------------------------------------------------------------- FrameMorrow

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


def feature_check(train_store, train, released, agg, limit=1024):
    """Do the re-encoded TRAIN features reproduce the team's released reader quantities?

    Recomputes the frozen readers' pooled attention and 16-plan sensitivity inputs on TRAIN records and
    compares with the released stage-A arrays, which were computed from the team's own feature cache.
    """
    out = {}
    records = [r for r in train if r["split"] == "train"][:limit]
    index = {r["id"]: i for i, r in enumerate(train_store.split("train"))}
    rows = [index[r["id"]] for r in records]
    caps = [train_store.capsule(r) for r in records]
    for reader in sd.READERS:
        path = released / f"stageA/{reader}_train.npz"
        if not path.exists():
            continue
        a = np.load(path)
        model = sd.load_reader(reader, train_store, agg)
        mine = pipe.pooled_attention(model, caps)
        theirs = a["attention"][rows]
        out[reader] = {"records": len(records), "attention_max_abs_diff": float(np.abs(mine - theirs).max()),
                       "attention_choice_agreement": float((mine.argmin(1) == theirs.argmin(1)).mean())}
    return out


# ---------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--terminal", type=Path, required=True); ap.add_argument("--agg", type=Path, required=True)
    ap.add_argument("--train-xfer", type=Path, required=True); ap.add_argument("--conf-xfer", type=Path, required=True)
    ap.add_argument("--released", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-split", default="train"); ap.add_argument("--limit", type=int)
    ap.add_argument("--pafu-steps", type=int, default=1000); ap.add_argument("--pacc-steps", type=int, default=3000)
    ap.add_argument("--fm-epochs", type=int, default=30); ap.add_argument("--fm-hidden", type=int, default=512)
    ap.add_argument("--fm-layers", type=int, default=4); ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--check-only", action="store_true", help="only compare the re-encoded TRAIN features with the release")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    started = time.time()

    def log(msg):
        print(f"[+{time.time() - started:6.0f}s] {msg}", flush=True)

    conf_store = sd.Store(args.conf_xfer / "records.json", args.conf_xfer / "actions.npz", args.conf_xfer, args.terminal)
    conf = conf_store.records
    if args.train_xfer == args.conf_xfer:   # mechanics-only local check, never a result
        train_store, train = conf_store, conf
    else:
        train_store = sd.Store(args.terminal / "data/records.json", args.terminal / "data/actions.npz",
                               args.train_xfer, args.terminal)
        train = train_store.split(args.train_split)
    if args.limit:
        train = train[:args.limit]
    released = args.released / RELEASED
    check = {} if train_store is conf_store else feature_check(train_store, train, released, args.agg)
    for reader, value in check.items():
        log(f"train feature check {reader}: {value}")
    if args.check_only:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "train_feature_check.json").write_text(json.dumps(check, indent=2) + "\n")
        return

    fm_drops = {}
    for seed in SEEDS:
        fm_drops[seed] = framemorrow(train_store, train, conf_store, conf, seed, epochs=args.fm_epochs,
                                     hidden=args.fm_hidden, layers=args.fm_layers)
        log(f"framemorrow seed {seed} trained")
    fm_teacher = framemorrow_teacher(conf_store, conf)

    summary = {"task": "synergy_terminal_RDS", "split": "confirmation (seed 20261104, 48 families)",
               "records": len(conf), "train_records": len(train), "train_feature_check": check, "readers": {},
               "note": "ours and the reference rungs are the team's frozen confirmation outputs; the literature "
                       "baselines are trained here on re-encoded TRAIN features"}
    for reader in sd.READERS:
        a = np.load(released / f"conf/stageA/{reader}_confirmation.npz")
        if a["ids"].tolist() != [r["id"] for r in conf]:
            raise ValueError("released confirmation arrays do not align with the confirmation records")
        cost = a["eval_cost"]
        z = np.load(released / f"conf/predictions/{reader}_standard/confirmation_predictions.npz")
        model = sd.load_reader(reader, conf_store, args.agg)
        methods = {
            "fifo": sd.score(conf, cost, drops=[np.zeros(len(conf), int)]),
            "oracle": sd.score(conf, cost, drops=[cost.argmin(1)]),
            "attention": sd.score(conf, cost, drops=[a["attention"].argmin(1)]),
            "full_sensitivity": sd.score(conf, cost, drops=[a["sensitivity"].argmin(1)]),
            "deletion_cost_student": sd.score(conf, cost, drops=[z[f"set_cost_seed{s}__drop"] for s in SEEDS]),
            "future_attention_student": sd.score(conf, cost, drops=[z[f"set_future_attention_seed{s}__drop"] for s in SEEDS]),
            "fm_published": sd.score(conf, cost, drops=[fm_drops[s] for s in SEEDS]),
            "fm_teacher_top2": sd.score(conf, cost, drops=[fm_teacher]),
        }
        drops, losses = [], []
        for seed in SEEDS:
            drops.append(pafu(model, train_store, train, conf_store, conf, seed, steps=args.pafu_steps))
            losses.append(pacc(model, train_store, train, conf_store, conf, seed, steps=args.pacc_steps))
            log(f"{reader} seed {seed}: pafu and pacc trained")
        methods["pafu_kv"] = sd.score(conf, cost, drops=drops)
        methods["pafu_teacher_top2"] = sd.score(conf, cost, drops=[pafu_teacher(model, conf_store, conf, "evaluation").argmin(1)])
        methods["pacc"] = sd.score(conf, cost, losses=losses)
        summary["readers"][reader] = methods
        for name, m in methods.items():
            log(f"{reader:15s} {name:26s} " + "  ".join(
                f"{t}: lost {'  n/a ' if m[t]['needed_evidence_lost'] is None else format(100 * m[t]['needed_evidence_lost'], '6.2f')}%"
                f" regret {m[t]['regret']:.4f}" for t in sd.TASKS))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    log(f"wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
