"""Stage B: train preregistered students with Codex's fit_terminal_students.py, then predict DEV.

Usage: python -I run_students.py <reader> <stageA_dir> <out_dir>
Run 1 ("standard"): cost / future_attention / shuffle labels + appearance-only control.
Run 2 ("distilled"): label 'cost' replaced by the 16-plan full-set sensitivity score (small -> delete);
only its independent/set 'cost' arms are reported as distilled-sensitivity students.
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipe  # noqa: E402
from path_config import APPEARANCE_OUTPUT
from experiments.phase2 import fit_terminal_students as fts  # noqa: E402

reader, A, OUT = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
st = pipe.Store()
app = np.load(APPEARANCE_OUTPUT)
app_by_id = dict(zip(app["ids"].tolist(), app["items"]))
cfg = json.load(open(pipe.REPO / "experiments/phase2/configs/terminal_structures_20261025.json"))["students"]
config = fts.StudentConfig(**{**cfg, "seeds": tuple(cfg["seeds"]), "cpu_threads": 2})


def split_arrays(split):
    recs = [r for r in st.records if r["split"] == split]
    a = np.load(A / f"{reader}_{split}.npz")
    assert a["ids"].tolist() == [r["id"] for r in recs]
    feats = [st.student_features(r) for r in recs]
    inputs = {"items": np.stack([f[0] for f in feats]), "context": np.stack([f[1] for f in feats])}
    counts = {t: sum(r["task_family"] == t for r in recs) for t in "RDS"}
    w = np.asarray([1 / counts[r["task_family"]] for r in recs], np.float64); w /= w.sum()
    appearance = {"items": np.stack([app_by_id[r["id"]] for r in recs]), "context": np.empty((len(recs), 0), np.float32)}
    return recs, a, inputs, w, appearance


def export(split, label_mode):
    recs, a, inputs, w, appearance = split_arrays(split)
    if label_mode == "standard":
        cost = a["ref_cost"] if split == "train" else a["eval_cost"]
        att = a["teacher_ref"] if split == "train" else a["teacher_eval"]
        return fts.StudentExport(inputs, cost, att, [r["id"] for r in recs], appearance, w)
    att = a["teacher_ref"] if split == "train" else a["teacher_eval"]
    return fts.StudentExport(inputs, a["sensitivity"], att, [r["id"] for r in recs], None, w)


dev_recs, _, dev_inputs, _, dev_app = split_arrays("development")
results = {}
for mode in ("standard", "distilled"):
    out = OUT / f"{reader}_{mode}"
    manifest = fts.fit_students(export("train", mode), export("validation", mode), out, config)
    preds = fts.predict_students(out, dev_inputs, dev_app if mode == "standard" else None)
    np.savez_compressed(out / "dev_predictions.npz", ids=np.array([r["id"] for r in dev_recs]),
                        **{k + "__drop": v["drop"] for k, v in preds.items()},
                        **{k + "__logits": v["logits"] for k, v in preds.items()})
    results[mode] = {k: {"selected_step": v["selected_step"], "parameters": v["parameters"]} for k, v in manifest["arms"].items()}
    print(json.dumps({"reader": reader, "mode": mode, "arms": len(manifest["arms"])}), flush=True)
json.dump(results, open(OUT / f"{reader}_student_summary.json", "w"), indent=1)
