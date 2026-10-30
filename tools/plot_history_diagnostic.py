"""Plot paired history benefit from a saved read-only diagnostic summary."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.summary.read_text())
    rows = report["aggregate_metrics"]
    versions = ("initial", "best", "latest")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    colors = {"train": "#146b8c", "validation": "#c24b38"}
    for ax, task in zip(axes, ("gate", "motion")):
        for split in ("train", "validation"):
            values = []
            for version in versions:
                means = {}
                for condition in ("recent_only", "privileged_small_budget"):
                    selected = [r for r in rows if r["task"] == task and r["split"] == split
                                and r["version"] == version and r["condition"] == condition]
                    assert sorted(r["horizon"] for r in selected) == [3, 6]
                    means[condition] = np.mean([r["mse"]["mean"] for r in selected])
                values.append(100 * (1-means["privileged_small_budget"]/means["recent_only"]))
            ax.plot(range(3), values, marker="o", lw=2, color=colors[split], label=split.title())
            for x, y in enumerate(values):
                ax.annotate(f"{y:+.1f}%", (x, y), xytext=(0, 8 if split == "train" else -15),
                            textcoords="offset points", ha="center", fontsize=9, color=colors[split])
        ax.axhline(0, color="#777777", linewidth=.8, linestyle="--")
        ax.set_ylim(-7, 13.5)
        ax.set_xticks(range(3), ["Before training", "Epoch 1\n(selected)", "Epoch 4"])
        ax.set_title(task.title(), fontsize=13)
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("MSE reduction from privileged K=2 history (%)\nPositive = better than recent-only")
    axes[0].legend(frameon=False)
    fig.suptitle("Key histories do not improve this pilot's validation MSE", fontsize=14)
    fig.text(.5, .02, "Same saved predictors; mean of horizons 3 and 6. Each task: 16 train pairs, 4 reused validation pairs.\n"
             "Evaluator-selected history is a diagnostic, not a deployable retention policy. No significance claim.",
             ha="center", fontsize=8.5, color="#555555")
    fig.tight_layout(rect=(0, .11, 1, .94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
