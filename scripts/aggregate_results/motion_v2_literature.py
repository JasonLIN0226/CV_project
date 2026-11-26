"""Pool the Motion v2 literature baselines next to the tracked online table.

    BM_MOTION_ROOT=/scratch/$USER python scripts/aggregate_results/motion_v2_literature.py \\
        --config configs/motion_v2/frozen.yaml --baselines <dir of scripts/train_baselines/motion_v2.py> --out <file.json>

Reads <baselines>/raw/<interval>/<label>_seed<seed>.npz for whatever reader seeds exist (17, 29, 43) and puts
the pooled means (environments/motion_v2/episodes.summarize) next to the project baselines, students and
references of the tracked step-3/4 evidence. Prints each interval sorted by +3 position MSE.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402

from bounded_memory import runs  # noqa: E402
from bounded_memory.environments.motion_v2 import episodes, evidence, frozen  # noqa: E402

SEEDS = (17, 29, 43)
MEMORY = {"pafu_kv": "bounded", "pafu_teacher_top2": "privileged (reads the future)", "pacc_full": "full history",
          "pacc_recurrent": "bounded", "fm_ijepa_published": "full history",
          "fm_ijepa_teacher_top2": "privileged (reads the future)", "fm_ijepa_online": "bounded"}
TRACKED_MEMORY = {"reference/posthoc_best": "privileged (best pair in hindsight)",
                  "reference/b": "privileged (the two cue frames)",
                  "reference/last_visible": "privileged (reads visibility)"}


def pool(folder):
    """{label: pooled summary} of the seeds present in one interval's raw folder."""
    rows = {}
    for label in sorted({p.stem.rsplit("_seed", 1)[0] for p in folder.glob("*_seed*.npz")}):
        scores = {}
        for s in SEEDS:
            f = folder / f"{label}_seed{s}.npz"
            if f.exists():
                with np.load(f) as z:
                    scores[s] = {k: z[k] for k in ("position", "feature", "success")}
        rows[f"literature/{label}"] = {"memory": MEMORY.get(label), **episodes.summarize(scores, list(scores))}
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--baselines", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    os.environ.setdefault("BM_REPO", str(runs.REPO))
    paths = frozen.Paths.from_config(runs.load_config(args.config)["assets"])
    table = {}
    for interval in frozen.INTERVALS:
        rows = pool(args.baselines / "raw" / interval)
        try:
            tracked = evidence.online_table(paths.evidence, interval)
        except FileNotFoundError:
            tracked = {}
        for name, r in tracked.items():
            rows[name] = {"memory": TRACKED_MEMORY.get(name, "bounded"),
                          "position_mse_+3": r["position_mse"]["3"], "position_mse_+6": r["position_mse"]["6"],
                          "success_rate": r["success_rate"], "source": "tracked step-3/4 evidence"}
        table[interval] = rows
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, indent=2) + "\n")
    for interval, rows in table.items():
        print(f"\n{interval}: +3 position MSE (lower is better)")
        for name, r in sorted(rows.items(), key=lambda kv: kv[1]["position_mse_+3"]):
            print(f"  {name:48s} {r['position_mse_+3']:.6f}  {r['memory']}")


if __name__ == "__main__":
    main()
