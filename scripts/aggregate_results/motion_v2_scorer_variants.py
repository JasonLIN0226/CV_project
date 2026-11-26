"""Choose our Motion v2 scorer's variant on fresh dev, before any holdout or validation look.

    python scripts/aggregate_results/motion_v2_scorer_variants.py --runs <dir> --out <file.json>

Reads <dir>/<variant>_seed<seed>.json written by scripts/train_scorer/motion_v2_deletion_cost_v2.py (fresh
episodes). A variant's dev score is the mean over its three seeds of the selected checkpoint's dev position MSE
(mean of +3 and +6; step 0, the attention policy, is always a candidate). The variant with the lowest score is
chosen among the variants present; a variant present must have all three seeds, and the variants with no
results are listed as absent. Writes the per-variant scores, the dev attention policy, the absent variants and
the choice.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402

from bounded_memory.environments.motion_v2.step34_core import SEEDS  # noqa: E402
from bounded_memory.methods.deletion_cost.motion_v2 import VARIANTS  # noqa: E402


def summarize(folder):
    variants, reference, absent = {}, None, []
    for name in VARIANTS:
        records = {s: folder / f"{name}_seed{s}.json" for s in SEEDS}
        if not any(p.exists() for p in records.values()):
            absent.append(name)
            continue
        missing = [s for s, p in records.items() if not p.exists()]
        if missing:
            raise SystemExit(f"variant {name} lacks seeds {missing}")
        per = {s: json.loads(p.read_text()) for s, p in records.items()}
        reference = reference or next(iter(per.values()))["dev_attention"]
        variants[name] = {
            "dev_mean": float(np.mean([r["best_dev_mean"] for r in per.values()])),
            "dev_mean_per_seed": {s: r["best_dev_mean"] for s, r in per.items()},
            "best_round_per_seed": {s: r["best_round"] for s, r in per.items()},
            "selected_step_per_seed": {s: r["history"][r["best_round"]]["selected_step"] for s, r in per.items()},
            "final_dev_mean_per_round": {s: [h["final_dev_mean"] for h in r["history"]] for s, r in per.items()}}
    if not variants:
        raise SystemExit(f"no scorer results in {folder}")
    chosen = min(variants, key=lambda n: variants[n]["dev_mean"])
    return {"dev_attention": reference, "variants": variants, "absent": absent, "chosen": chosen}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    summary = summarize(args.runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"dev attention policy {summary['dev_attention']['mean']:.6f}")
    for name, v in summary["variants"].items():
        print(f"{name:12s} dev {v['dev_mean']:.6f}  rounds {v['best_round_per_seed']}  steps {v['selected_step_per_seed']}")
    if summary["absent"]:
        print(f"absent (no results): {', '.join(summary['absent'])}")
    print(f"chosen on fresh dev: {summary['chosen']}")


if __name__ == "__main__":
    main()
