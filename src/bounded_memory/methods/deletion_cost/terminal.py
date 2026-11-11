"""Our method on the terminal (Synergy) tasks: reader-derived teacher labels and student inputs.

Stage A runs the frozen reader on every possible deletion of each three-card decision. The mean
in-span loss per deletion over the action plans is the deletion-cost label (`ref_cost` on TRAIN
with the 16 reference plans, `eval_cost` on held-out splits with the 32 evaluation plans). The
same pass yields the quantities the ablations and baselines use: the reader's pooled attention,
the future-attention teacher, the full-set sensitivity and the prediction medoid.

`bank_coeffs`, `selectors`, `bank_losses` and the loop in `stage_a` are extracted from
results/literature_diagnosis/qualified-reader-retention-20261025/code/run_main.py (the functions
now take the model and Gram matrix as arguments instead of reading module globals). `student_export`
and `ARM_NAMES` come from that folder's run_students.py and evaluate_dev.py. The student itself
(set-conditioned or independent scorer, three label kinds) is terminal_students.py.
"""
from __future__ import annotations

import numpy as np

from ...environments.synergy.assets import KEEPS, coefficients, pooled_attention, qform, teacher_attention
from . import terminal_students as fts


def bank_coeffs(model, caps, plans):
    """[N,P,4,2,128] for every keep code and every plan."""
    n, p = len(caps), len(plans)
    rep = [c for c in caps for _ in range(p)]
    pl = np.tile(plans, (n, 1, 1))
    res = np.empty((n, p, 4, 2, 128), np.float64)
    for d in range(4):
        res[:, :, d] = coefficients(model, rep, KEEPS[d], pl).reshape(n, p, 2, 128)
    return res


def selectors(C, G):
    """C [N,P,4,2,128] from the reference bank -> medoid pairwise sums & full-set sensitivity."""
    del_ = C[:, :, :3]
    pair = np.zeros((len(C), 3, 3))
    for i in range(3):
        for j in range(i + 1, 3):
            pair[:, i, j] = pair[:, j, i] = qform(del_[:, :, i] - del_[:, :, j], G).mean(1)
    sens = qform(del_ - C[:, :, 3:4], G).mean(1)                 # [N,3]
    return pair.sum(2), sens


def bank_losses(C, resid, G):
    """resid [N,P,2,128]; loss per deletion & plan inside span -> [N,P,3]."""
    return qform(C[:, :, :3] - resid[:, :, None], G)


def stage_a(store, model, records, train, log=None):
    """Per-record arrays for one split, aligned to `records`. `train` selects the reference plan bank."""
    G = store.G
    REF, EVAL = store.banks["reference"], store.banks["evaluation"]
    acc = {k: [] for k in ("medoid_sum", "sensitivity", "ref_cost", "eval_cost", "eval_losses",
                           "teacher_ref", "teacher_eval", "attention")}
    for a in range(0, len(records), 64):
        rr = records[a:a + 64]
        caps = [store.capsule(r) for r in rr]
        Cr = bank_coeffs(model, caps, REF)
        msum, sens = selectors(Cr, G)
        acc["medoid_sum"].append(msum); acc["sensitivity"].append(sens)
        acc["attention"].append(pooled_attention(model, caps))
        if train:
            resid = np.stack([store.target_residual(r, "reference") for r in rr])
            acc["ref_cost"].append(bank_losses(Cr, resid, G).mean(1))
            q = np.stack([store.P[np.asarray(r["plan_targets"]["reference"]).reshape(-1)] for r in rr])
            acc["teacher_ref"].append(teacher_attention(model, caps, q))
        else:
            Ce = bank_coeffs(model, caps, EVAL)
            resid = np.stack([store.target_residual(r, "evaluation") for r in rr])
            L = bank_losses(Ce, resid, G)
            acc["eval_losses"].append(L); acc["eval_cost"].append(L.mean(1))
            q = np.stack([store.P[np.asarray(r["plan_targets"]["evaluation"]).reshape(-1)] for r in rr])
            acc["teacher_eval"].append(teacher_attention(model, caps, q))
        if log and (a // 64) % 8 == 0:
            log(f"stage A {a + len(rr)}/{len(records)}")
    arrays = {k: np.concatenate(v) for k, v in acc.items() if v}
    arrays["ids"] = np.array([r["id"] for r in records])
    return arrays


def student_inputs(store, records):
    feats = [store.student_features(r) for r in records]
    return {"items": np.stack([f[0] for f in feats]), "context": np.stack([f[1] for f in feats])}


def student_export(store, records, arrays, mode, train, appearance=None):
    """What the students are fitted on. mode "standard": deletion cost, future attention and shuffled
    labels plus the appearance-only control. mode "distilled": the label `cost` is replaced by the
    16-plan full-set sensitivity (small -> delete); only its cost arms are reported."""
    if arrays["ids"].tolist() != [r["id"] for r in records]:
        raise ValueError("stage-A arrays are not aligned with the records")
    counts = {t: sum(r["task_family"] == t for r in records) for t in "RDS"}
    w = np.asarray([1 / counts[r["task_family"]] for r in records], np.float64); w /= w.sum()
    att = arrays["teacher_ref"] if train else arrays["teacher_eval"]
    ids = [r["id"] for r in records]
    if mode == "standard":
        cost = arrays["ref_cost"] if train else arrays["eval_cost"]
        app = None if appearance is None else {"items": appearance, "context": np.empty((len(records), 0), np.float32)}
        return fts.StudentExport(student_inputs(store, records), cost, att, ids, app, w)
    if mode == "distilled":
        return fts.StudentExport(student_inputs(store, records), arrays["sensitivity"], att, ids, None, w)
    raise ValueError(mode)


def arm_name(mode, key):
    """Released prediction key (e.g. set_cost_seed17) -> (policy name, seed), or None when not reported."""
    name = key[:-len("__drop")] if key.endswith("__drop") else key
    arch, rest = name.split("_", 1)
    label, seed = rest.rsplit("_seed", 1)
    if mode == "distilled":
        if label != "cost":
            return None
        return f"distilled_sensitivity_student_{arch}", int(seed)
    if arch == "appearance":
        return "appearance_only_student", int(seed)
    kind = {"cost": "deletion_cost_student", "future_attention": "future_attention_student",
            "shuffle": "shuffled_label_student"}[label]
    return f"{kind}_{arch}", int(seed)
