"""Score every released Synergy policy on the locked confirmation set, from the released arrays.

    python -m experiments.memory_baselines.synergy_released --released <conf_full> --conf-xfer <conf_xfer> \
        --out results/memory_baselines/synergy/released_scores.json [--terminal <run> --agg <reader run>]

Nothing is trained. The reference rules, the team's standard students and its distilled-sensitivity
students are scored with the same metric as synergy_run (needed-evidence loss and prediction regret
per task). With --terminal and --agg two reader checks are added:

  deletion_path_check   a plain deletion passed through the path synergy_run.pacc evaluates with must
                        reproduce the released cost of that deletion (same scale as the cost table);
  three_cards           the frozen reader given all three cards, scored like a memory that deletes nothing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import synergy_data as sd

RELEASED = "results/literature_diagnosis/qualified-reader-retention-20261025"
STUDENTS = {  # name here -> (prediction run, key prefix in confirmation_predictions.npz)
    "deletion_cost_student_set": ("standard", "set_cost"),
    "deletion_cost_student_independent": ("standard", "independent_cost"),
    "future_attention_student_set": ("standard", "set_future_attention"),
    "future_attention_student_independent": ("standard", "independent_future_attention"),
    "shuffled_label_student_set": ("standard", "set_shuffle"),
    "shuffled_label_student_independent": ("standard", "independent_shuffle"),
    "appearance_only_student": ("standard", "appearance_cost"),
    "distilled_sensitivity_student_set": ("distilled", "set_cost"),
    "distilled_sensitivity_student_independent": ("distilled", "independent_cost"),
}


def released_scores(conf, released, reader):
    a = np.load(released / f"conf/stageA/{reader}_confirmation.npz")
    if a["ids"].tolist() != [r["id"] for r in conf]:
        raise ValueError("released confirmation arrays do not align with the confirmation records")
    cost = a["eval_cost"]
    out = {"fifo": sd.score(conf, cost, drops=[np.zeros(len(conf), int)]),
           "oracle": sd.score(conf, cost, drops=[cost.argmin(1)]),
           "attention": sd.score(conf, cost, drops=[a["attention"].argmin(1)]),
           "full_set_sensitivity": sd.score(conf, cost, drops=[a["sensitivity"].argmin(1)])}
    for name, (run, prefix) in STUDENTS.items():
        z = np.load(released / f"conf/predictions/{reader}_{run}/confirmation_predictions.npz")
        keys = sorted(k for k in z.files if k.startswith(prefix + "_seed") and k.endswith("__drop"))
        if len(keys) != 3:
            raise ValueError(f"{reader} {run} {prefix}: expected three student seeds, found {len(keys)}")
        out[name] = sd.score(conf, cost, drops=[z[k] for k in keys])
    return out, cost


def reader_checks(store, conf, cost, reader, agg, chunk=64):
    import torch
    import pipe
    from .synergy_run import reader_batch

    model = sd.load_reader(reader, store, agg)
    evaluation = store.banks["evaluation"]
    three, worst = np.empty(len(conf)), 0.0
    with torch.inference_mode():
        for b in range(0, len(conf), chunk):
            rr = conf[b:b + chunk]
            cs = [store.capsule(r) for r in rr]
            rep = [c for c in cs for _ in evaluation]
            plans = np.tile(evaluation, (len(cs), 1, 1))
            resid = np.stack([store.target_residual(r, "evaluation") for r in rr])

            def loss(memory=None):
                coeff = sd.reader_coefficients(model, reader_batch(rep, plans, memory)).numpy().astype(np.float64)
                return pipe.qform(coeff.reshape(len(cs), len(evaluation), 2, -1) - resid, store.G).mean(1)

            three[b:b + len(rr)] = loss()
            for d in range(3):
                keep = [j for j in range(3) if j != d]
                kept = tuple(torch.from_numpy(np.stack([c[k][4:][keep] for c in rep])) for k in ("pz", "past", "age"))
                worst = max(worst, float(np.abs(loss(kept) - cost[b:b + len(rr), d]).max()))
    return {"deletion_path_check": {"records": len(conf), "deletions": 3, "max_abs_diff_from_released_cost": worst},
            "three_cards": sd.score(conf, cost, losses=[three])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--released", type=Path, required=True); ap.add_argument("--conf-xfer", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--terminal", type=Path); ap.add_argument("--agg", type=Path)
    args = ap.parse_args()
    conf = json.loads((args.conf_xfer / "records.json").read_text())
    store = None
    if args.terminal and args.agg:
        store = sd.Store(args.conf_xfer / "records.json", args.conf_xfer / "actions.npz", args.conf_xfer, args.terminal)
    summary = {"task": "synergy_terminal_RDS", "split": "confirmation (seed 20261104, 48 families)", "records": len(conf),
               "source": f"released arrays under {RELEASED}/conf (stageA and predictions); nothing trained here",
               "readers": {}}
    for reader in sd.READERS:
        methods, cost = released_scores(conf, args.released / RELEASED, reader)
        if store is not None:
            methods.update(reader_checks(store, conf, cost, reader, args.agg))
        summary["readers"][reader] = methods
        for name, m in methods.items():
            if "S" in m:
                lost = m["S"]["needed_evidence_lost"]
                print(f"{reader:15s} {name:42s} S: lost {'  n/a ' if lost is None else f'{100 * lost:5.2f}%'}  regret {m['S']['regret']:+.4f}")
            else:
                print(f"{reader:15s} {name:42s} {m}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
