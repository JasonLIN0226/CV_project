"""Stage A: reader-derived quantities for TRAIN/VAL/DEV (labels, selectors, oracle).

Usage: python -I run_main.py <reader> <outdir>
Writes <outdir>/<reader>_<split>.npz with per-record arrays aligned to record order.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipe  # noqa: E402
from path_config import SPLITS

reader, out = sys.argv[1], Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
st = pipe.Store()
model = pipe.load_reader(reader, st)
G = st.G
REF, EVAL = st.banks["reference"], st.banks["evaluation"]


def bank_coeffs(caps, plans):
    """[N,P,4,2,128] for every keep code and every plan."""
    n, p = len(caps), len(plans)
    rep = [c for c in caps for _ in range(p)]
    pl = np.tile(plans, (n, 1, 1))
    res = np.empty((n, p, 4, 2, 128), np.float64)
    for d in range(4):
        res[:, :, d] = pipe.coefficients(model, rep, pipe.KEEPS[d], pl).reshape(n, p, 2, 128)
    return res


def selectors(C):
    """C [N,P,4,2,128] from the reference bank -> medoid pairwise sums & full-set sensitivity."""
    del_ = C[:, :, :3]
    pair = np.zeros((len(C), 3, 3))
    for i in range(3):
        for j in range(i + 1, 3):
            pair[:, i, j] = pair[:, j, i] = pipe.qform(del_[:, :, i] - del_[:, :, j], G).mean(1)
    sens = pipe.qform(del_ - C[:, :, 3:4], G).mean(1)                 # [N,3]
    return pair.sum(2), sens


def bank_losses(C, resid):
    """resid [N,P,2,128]; loss per deletion & plan inside span -> [N,P,3]."""
    return pipe.qform(C[:, :, :3] - resid[:, :, None], G)


for split in SPLITS:
    started = time.time()
    recs = [r for r in st.records if r["split"] == split]
    ids = [r["id"] for r in recs]
    acc = {k: [] for k in ("medoid_sum", "sensitivity", "ref_cost", "eval_cost", "eval_losses",
                           "teacher_ref", "teacher_eval", "attention")}
    for a in range(0, len(recs), 64):
        rr = recs[a:a + 64]
        caps = [st.capsule(r) for r in rr]
        Cr = bank_coeffs(caps, REF)
        msum, sens = selectors(Cr)
        acc["medoid_sum"].append(msum); acc["sensitivity"].append(sens)
        acc["attention"].append(pipe.pooled_attention(model, caps))
        if split == "train":
            resid = np.stack([st.target_residual(r, "reference") for r in rr])
            acc["ref_cost"].append(bank_losses(Cr, resid).mean(1))
            q = np.stack([st.P[np.asarray(r["plan_targets"]["reference"]).reshape(-1)] for r in rr])
            acc["teacher_ref"].append(pipe.teacher_attention(model, caps, q))
        else:
            Ce = bank_coeffs(caps, EVAL)
            resid = np.stack([st.target_residual(r, "evaluation") for r in rr])
            L = bank_losses(Ce, resid)
            acc["eval_losses"].append(L); acc["eval_cost"].append(L.mean(1))
            q = np.stack([st.P[np.asarray(r["plan_targets"]["evaluation"]).reshape(-1)] for r in rr])
            acc["teacher_eval"].append(pipe.teacher_attention(model, caps, q))
        if (a // 64) % 8 == 0:
            print(json.dumps({"reader": reader, "split": split, "done": a + len(rr), "total": len(recs),
                              "sec": round(time.time() - started, 1)}), flush=True)
    arrays = {k: np.concatenate(v) for k, v in acc.items() if v}
    np.savez_compressed(out / f"{reader}_{split}.npz", ids=np.array(ids), **arrays)
    print(json.dumps({"reader": reader, "split": split, "saved": sorted(arrays), "sec": round(time.time() - started, 1)}), flush=True)
