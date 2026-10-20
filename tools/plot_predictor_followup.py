"""Plot saved development controls without executing or selecting a model."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.summary.read_text())
    conditions = ("recent_only", "privileged_small_budget", "wrong_evidence", "full_history")
    labels = ("Recent\nonly", "Correct key\nhistory, K=2", "Wrong key\nhistory, K=2", "Full\nhistory*")
    colors = ("#7b8794", "#167f76", "#bf5b47", "#4883a5")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.9), sharey=True)
    for ax, task in zip(axes, ("gate", "motion")):
        arm = next(a for a in report["arms"] if a["task"] == task and a["arm"] == "unit_columns")
        values = arm["diagnostics"]["selected"]["validation"]["metrics"]
        bars = ax.bar(range(4), [values[c]["mse"] for c in conditions], color=colors, width=.7)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.004,
                    f"{bar.get_height():.6f}", ha="center", va="bottom", fontsize=9)
        ax.axhline(values["current_plus_train_target_mean"]["mse"], color="#555555",
                   ls="--", lw=1, label="Current + train-mean residual")
        ax.set_xticks(range(4), labels, fontsize=9)
        ax.set_ylim(0, .226)
        ax.set_title(f"{task.title()} — selected epoch {arm['selected_epoch']}", fontsize=12)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Validation feature MSE (lower is better)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Gate benefits from history after removing input whitening", fontsize=14)
    fig.text(.5, .025, "Fixed three-arm, 40-epoch development comparison; no-whitening arms shown. Four reused validation pairs per task.\n"
             "Key and wrong histories are evaluator controls. *Full history exceeds K=2. No learned-retention or generalization claim.",
             ha="center", fontsize=8.5, color="#555555")
    fig.tight_layout(rect=(0, .13, 1, .94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
