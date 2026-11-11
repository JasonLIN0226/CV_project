"""Gate assets as plain arrays, and the frozen predictor as a function of memory items.

Everything here is read-only with respect to the frozen assets: the dataset, I-JEPA
features, train-only transforms, state decoder and the three predictors. Metrics follow
the definitions in retention_online_matrix/run_online_matrix.py::append_metric_rows.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
R, B, HORIZONS = 4, 2, (3, 6)
SEEDS = (17, 29, 43)


def add_paths(snapshot_src=None):
    """Import path for the frozen predictor source; the snapshot wins when given."""
    paths = [ROOT, ROOT / "experiments/predictor/control/src",
             ROOT / "experiments/retention_baseline_matrix", ROOT / "experiments/retention_online_matrix"]
    for p in paths:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    if snapshot_src is not None:
        snapshot_src = Path(snapshot_src).resolve()
        sys.path.insert(0, str(snapshot_src))
        import phase1
        if Path(phase1.__file__).resolve().parent != snapshot_src / "phase1":
            raise RuntimeError("phase1 was imported before the frozen snapshot was put on the path")


@dataclass
class Episode:
    id: str
    pair: str
    sibling: str
    split: str
    q: int
    cue: tuple            # inclusive (start, end) of the cue interval
    evidence: np.ndarray  # privileged timestamps, used only by the oracle reference
    z: np.ndarray         # [T, D] train-normalised I-JEPA features; frames after q are clean targets
    p: np.ndarray         # [T, d] projected features (the predictor's token space)
    actions: np.ndarray   # [T-1, 2]
    classes: np.ndarray   # [T] route label per frame (evaluation only)
    prefix: np.ndarray    # [T, 2] cumulative executed actions

    @property
    def candidates(self):
        return self.q - R + 1  # frames 0 .. q-R are eligible history at the query


@dataclass
class GateAssets:
    c: dict
    store: object
    episodes: list
    decoder: object
    device: torch.device

    def split(self, name):
        return [e for e in self.episodes if e.split == name]

    def predictor(self, seed):
        from phase1.training import load_predictor
        model, _ = load_predictor(self.c, seed, self.store, self.device, "gate")
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        return model


def load_gate(config_path, device="cpu"):
    from phase1.data import load_episode
    from phase1.inputs import FeatureStore, selected_evidence
    from phase1.model import fit_basis
    from phase1.probes import load_decoder

    c = json.loads(Path(config_path).read_text())
    if (c["data"]["recent"], c["data"]["budget"], tuple(c["data"]["horizons"])) != (R, B, HORIZONS):
        raise ValueError("expected R=4, B=2, horizons (3, 6)")
    store = FeatureStore(c, max_cached=len(json.loads(
        (Path(c["output"]) / "dataset/manifest.json").read_text())["records"]) + 8)
    store.bases = {"gate": fit_basis(c, store, "gate")}
    episodes = []
    for rec in store.manifest["records"]:
        if rec["task"] != "gate" or rec["split"] not in ("train", "validation"):
            raise ValueError("Gate train/validation records only")
        meta = load_episode(c, rec, True, keys=("classes", "evidence"))
        actions = load_episode(c, rec, keys=("actions",))["actions"].astype(np.float32)
        z = store.features(rec)
        q = int(rec["query"])
        if len(z) < q + max(HORIZONS) + 1 or len(actions) < q + max(HORIZONS):
            raise ValueError("episode is shorter than query + horizon")
        episodes.append(Episode(
            id=rec["id"], pair=rec["pair"], sibling=rec["sibling"], split=rec["split"], q=q,
            cue=(int(rec["cue_start"]), int(rec["cue_end"])),
            evidence=selected_evidence(c, rec, meta["evidence"]),
            z=z, p=store.projected_features(rec).copy(), actions=actions,
            classes=np.asarray(meta["classes"]),
            prefix=np.concatenate([np.zeros((1, 2), np.float32), np.cumsum(actions, axis=0)]).astype(np.float32)))
    return GateAssets(c=c, store=store, episodes=episodes, decoder=load_decoder(c, "gate"),
                      device=torch.device(device))


# ---------------------------------------------------------------- predictor as a function

def item_batch(episodes, clocks, memory_times, device):
    """Predictor inputs for [R recent frames + the given memory frames] at each clock."""
    n = R + max((len(m) for m in memory_times), default=0)
    rows = len(episodes)
    d, dim = episodes[0].p.shape[1], episodes[0].z.shape[1]
    out = {"projected_z": np.zeros((rows, n, d), np.float32), "age": np.zeros((rows, n), np.float32),
           "type": np.zeros((rows, n), np.int64), "past_actions": np.zeros((rows, n, 2), np.float32),
           "mask": np.zeros((rows, n), bool), "actions": np.zeros((rows, max(HORIZONS), 2), np.float32),
           "current": np.zeros((rows, dim), np.float32), "projected_current": np.zeros((rows, d), np.float32)}
    for i, (e, t, mem) in enumerate(zip(episodes, clocks, memory_times)):
        mem = sorted(int(m) for m in mem)
        if len(set(mem)) != len(mem) or any(m < 0 or m > t - R for m in mem):
            raise ValueError("memory frame outside the legal history")
        times = np.array(list(range(t - R + 1, t + 1)) + mem, np.int64)
        k = len(times)
        out["projected_z"][i, :k] = e.p[times]
        out["age"][i, :k] = t - times
        out["type"][i, R:k] = 1
        out["past_actions"][i, :k] = e.prefix[t] - e.prefix[times]
        out["mask"][i, :k] = True
        out["actions"][i] = e.actions[t:t + max(HORIZONS)]
        out["current"][i] = e.z[t]
        out["projected_current"][i] = e.p[t]
    batch = {k: torch.from_numpy(v).to(device) for k, v in out.items()}
    batch["z"] = torch.zeros(rows, n, 1, device=device)  # unused once projected_z is supplied
    return batch


def with_memory_tokens(batch, token_p, token_past, token_age):
    """Replace the memory part of a recent-only batch by synthetic memory items (differentiable)."""
    rows, m = token_p.shape[:2]
    recent = {k: batch[k][:, :R] for k in ("projected_z", "age", "type", "past_actions", "mask")}
    new = dict(batch)
    new["projected_z"] = torch.cat([recent["projected_z"], token_p], 1)
    new["past_actions"] = torch.cat([recent["past_actions"], token_past], 1)
    new["age"] = torch.cat([recent["age"], token_age], 1)
    new["type"] = torch.cat([recent["type"], torch.ones(rows, m, dtype=torch.long, device=token_p.device)], 1)
    new["mask"] = torch.cat([recent["mask"], torch.ones(rows, m, dtype=torch.bool, device=token_p.device)], 1)
    new["z"] = torch.zeros(rows, R + m, 1, device=token_p.device)
    return new


def pooling_attention(model, keys_p, current_p):
    """The frozen predictor's own pooling weights over `keys_p` for a query at `current_p`.

    Same operations as DynamicsPredictor.forward: learned score plus the novelty term, softmax over keys.
    keys_p [rows, K, d], current_p [rows, d] -> [rows, K].
    """
    delta = keys_p - current_p[:, None]
    score = model.score(torch.cat([keys_p, delta], -1)).squeeze(-1) + torch.log1p(delta.square().mean(-1))
    return torch.softmax(score, 1)


# ---------------------------------------------------------------- metrics

def metric_rows(assets, episodes, prediction, selected, policy, seed):
    """One row per episode and horizon. `selected` is a list of kept frame lists, or None for token memories."""
    prediction = np.asarray(prediction, np.float32)
    if prediction.shape[:2] != (len(episodes), len(HORIZONS)) or not np.isfinite(prediction).all():
        raise ValueError("invalid predictor output")
    rows = []
    for i, e in enumerate(episodes):
        state = assets.decoder.predict(prediction[i])
        kept = None if selected is None else sorted(int(t) for t in selected[i])
        for h_index, h in enumerate(HORIZONS):
            target = e.z[e.q + h]
            rows.append({
                "policy": policy, "seed": seed, "episode": e.id, "pair": e.pair, "horizon": h,
                "route_accuracy": float((state[h_index, 0] > 0) == (e.classes[e.q + h] > 0)),
                "normalized_mse": float(np.mean((prediction[i, h_index] - target) ** 2)),
                "cue_kept": None if kept is None else float(any(e.cue[0] <= t <= e.cue[1] for t in kept)),
                "kept": None if kept is None else kept})
    return rows


def summarise(rows):
    """Pooled means per policy over episodes, seeds and horizons (the README's aggregation)."""
    out = {}
    for policy in sorted({r["policy"] for r in rows}):
        group = [r for r in rows if r["policy"] == policy]
        cue = [r["cue_kept"] for r in group if r["cue_kept"] is not None]
        per_seed = {}
        for seed in sorted({r["seed"] for r in group}):
            per_seed[str(seed)] = float(np.mean([r["route_accuracy"] for r in group if r["seed"] == seed]))
        out[policy] = {"route_accuracy": float(np.mean([r["route_accuracy"] for r in group])),
                       "normalized_mse": float(np.mean([r["normalized_mse"] for r in group])),
                       "cue_kept": float(np.mean(cue)) if cue else None,
                       "rows": len(group), "route_accuracy_by_seed": per_seed}
    return out


@torch.inference_mode()
def predict_selection(model, episodes, selected, device, chunk=256):
    preds = []
    for i in range(0, len(episodes), chunk):
        eps = episodes[i:i + chunk]
        batch = item_batch(eps, [e.q for e in eps], selected[i:i + chunk], device)
        preds.append(model(batch).cpu().numpy())
    return np.concatenate(preds)
