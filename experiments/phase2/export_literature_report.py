"""Export existing literature-baseline evidence and two matplotlib figures.

This script never imports a model/runner, trains, encodes, or runs an experiment.
The portable archive is sufficient to redraw the figures with --render-only.
Binary inputs/checkpoints stay outside the archive; their hashes are recorded.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ORDER = ("fifo", "reservoir", "visual_diversity", "attention",
         "prediction_surprise", "wr", "pafu")
LABELS = ("FIFO", "Reservoir", "Visual diversity", "Reader attention",
          "Prediction surprise", "WR-inspired", "PaFu-inspired")
COLORS = ["#9ba5ac"] * 5 + ["#c55a35", "#237aa5"]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def relative(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_info(path):
    return {"path_relative_to_repository": relative(path), "sha256": sha(path),
            "bytes": Path(path).stat().st_size}


def scrub(value):
    """Preserve numeric values while removing machine-specific repo prefixes."""
    if isinstance(value, dict):
        return {str(k).replace(str(ROOT), "$REPOSITORY"): scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    return value.replace(str(ROOT), "$REPOSITORY") if isinstance(value, str) else value


def wr_trace(rows, events):
    output = []
    for row in rows:
        if row["policy"] != "wr":
            continue
        selected = [e for e in events if e["policy"] == "wr"
                    and e["episode_id"] == row["episode_id"] and e["seed"] == row["seed"]]
        scalars, timeline = {}, []
        for event in selected:
            timeline.append({key: event[key] for key in
                             ("time", "retained_timestamps", "recent_timestamps", "deleted_timestamp")})
            for record in event["wr_scalar_records"]:
                t = record["time"]
                base = {k: record[k] for k in ("time", "score", "threshold", "valid")}
                if t in scalars and any(scalars[t][k] != v for k, v in base.items()):
                    raise ValueError("A causal WR score or threshold was revised")
                scalars.setdefault(t, {**base, "confirmed_at": None})
                if record["event"] and scalars[t]["confirmed_at"] is None:
                    scalars[t]["confirmed_at"] = event["time"]
        if [x["time"] for x in timeline] != list(range(row["query"] + 1)):
            raise ValueError("WR trace does not cover the complete online prefix")
        output.append({k: row[k] for k in ("episode_id", "pair", "seed", "query", "kept_times",
            "evidence_times", "evidence_equivalent_feature_times", "evidence_survived",
            "first_evidence_information_loss")} | {
                "scalar_records": [scalars[t] for t in sorted(scalars)], "timeline": timeline})
    return output


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "semibold", "axes.grid": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white"})


def save(fig, destination, stem):
    fig.savefig(destination / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(destination / f"{stem}.pdf", bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def metrics_figure(summary, rows, destination):
    aggregate = {x["policy"]: x for x in summary["aggregate"]}
    means = [aggregate[p]["mean_mse"] for p in ORDER]
    survival = [100 * aggregate[p]["evidence_survived"] for p in ORDER]
    reference = [r for r in rows if r["policy"] == "fifo"]
    recent = np.mean([r["recent_only_mse"] for r in reference])
    hindsight = np.mean([r["hindsight_oracle"]["mean_mse"] for r in reference])
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.2, 5.7),
                                     gridspec_kw={"width_ratios": [1.5, 1]})
    y = np.arange(len(ORDER))
    left.barh(y, means, color=COLORS, alpha=.75, height=.58)
    for i, policy in enumerate(ORDER):
        part = [r for r in rows if r["policy"] == policy]
        pairs = sorted({r["pair"] for r in part})
        values = [np.mean([r["mean_mse"] for r in part if r["pair"] == p]) for p in pairs]
        left.scatter(values, i + np.linspace(-.14, .14, len(values)), color="#27343b",
                     s=17, zorder=4, linewidths=.4, edgecolors="white")
    left.axvline(recent, ls="--", lw=1.2, color="#a44b4b", label=f"Recent only: {recent:.4f}")
    left.axvline(hindsight, ls=":", lw=1.6, color="#35664c", label=f"Hindsight K2*: {hindsight:.4f}")
    left.set(yticks=y, yticklabels=LABELS, xlabel="Mean feature MSE (h3 and h6; lower is better)",
             title="Prediction error", xlim=(0, max(max(means), .14) * 1.06))
    left.invert_yaxis()
    left.legend(frameon=False, loc="lower right", fontsize=9)
    left.grid(axis="x", alpha=.17)
    right.barh(y, survival, color=COLORS, alpha=.85, height=.58)
    for i, value in enumerate(survival):
        right.text(value + 2, i, f"{value:.0f}%", va="center", fontsize=9)
    right.set(yticks=y, yticklabels=[], xlabel="Cue feature-copy survival (%)",
              title="Cue retained at the query", xlim=(0, 116), xticks=[0, 25, 50, 75, 100])
    right.invert_yaxis()
    right.grid(axis="x", alpha=.17)
    fig.suptitle("Gate development: frozen predictor, R=4, K=2", fontsize=14, y=.98)
    fig.text(.5, .025, "Four reused validation pairs (8 episodes). Dots: pair means; PaFu averages 3 seeds.\n"
             "*Privileged evaluation-only exhaustive K2 subset reference. Equal K does not mean equal total bytes.",
             ha="center", va="bottom", fontsize=9, color="#4a5055")
    fig.subplots_adjust(left=.17, right=.98, top=.86, bottom=.19, wspace=.17)
    save(fig, destination, "metrics")


def choose_timeline_cases(traces):
    def key(case):
        return min(case["evidence_times"]), case["episode_id"]
    failed = sorted((c for c in traces if not c["evidence_survived"]), key=key)
    survived = sorted((c for c in traces if c["evidence_survived"]), key=key)
    if not failed or not survived:
        raise ValueError("Timeline requires both a cue-loss and a cue-retained WR case")
    return [failed[0], survived[0]]


def timeline_figure(traces, destination):
    cases = choose_timeline_cases(traces)
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 6.9),
                             gridspec_kw={"height_ratios": [2.2, 1]}, sharex="col")
    for col, case in enumerate(cases):
        top, bottom = axes[0, col], axes[1, col]
        records, q = case["scalar_records"], case["query"]
        t = [r["time"] for r in records]
        score = [r["score"] if r["valid"] else np.nan for r in records]
        threshold = [r["threshold"] if r["threshold"] is not None else np.nan for r in records]
        top.plot(t, score, "o-", ms=3.5, lw=1.2, color="#237aa5", label="Causal surprise")
        top.plot(t, threshold, "s--", ms=3, lw=1.1, color="#c55a35", label="Past-only MAD threshold")
        available = [r["time"] for r in records if r["threshold"] is not None]
        if available:
            top.axvspan(-.4, min(available)-.5, color="#d9d9d9", alpha=.45)
            top.text(.02, .97, "Threshold unavailable at startup", transform=top.transAxes,
                     ha="left", va="top", fontsize=9, color="#5c5c5c")
        cue = set(sum(case["evidence_equivalent_feature_times"], []))
        for ax in (top, bottom):
            for tick in sorted(cue):
                ax.axvspan(tick-.42, tick+.42, color="#8ab697", alpha=.2, lw=0)
            ax.axvline(q, color="#333333", ls=":", lw=1)
            ax.set_xlim(-.4, q+.6)
        for event in case["timeline"]:
            tick = event["time"]
            for lane, name, color in ((1, "recent_timestamps", "#789b7e"),
                                      (0, "retained_timestamps", "#237aa5")):
                if cue.intersection(event[name]):
                    bottom.broken_barh([(tick-.4, .8)], (lane-.22, .44), facecolors=color)
        first_loss = [x for x in case["first_evidence_information_loss"] if x is not None]
        if first_loss:
            loss = min(first_loss)
            bottom.axvline(loss, ls="--", color="#b63f32", lw=1)
            bottom.text(loss+.15, 1.5, f"Last copy lost: t={loss}", color="#a43c32", fontsize=9)
        else:
            bottom.text(.03, .91, "Cue copy still retained at query", transform=bottom.transAxes,
                        color="#35664c", fontsize=9)
        top.set_yscale("symlog", linthresh=1)
        top.set_ylim(0, 16000)
        top.set_title(("Cue-loss example" if col == 0 else "Cue-retained example") +
                      f"\n{case['episode_id'][:10]}  |  cue starts t={min(cue)}, query t={q}", fontsize=11)
        top.grid(axis="y", alpha=.2)
        bottom.set(yticks=[0, 1], yticklabels=["Historical K", "Recent R"], ylim=(-.5, 1.85),
                   xlabel="Observed frame time")
        bottom.set_xticks(range(0, q+1, 2))
        bottom.text(q-.08, .5, "query", ha="right", va="center", fontsize=8)
        if col == 0:
            top.set_ylabel("Surprise / threshold (symlog scale)")
            top.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, .87), fontsize=8)
    fig.suptitle("WR online traces: score availability and cue-copy retention", fontsize=14, y=.98)
    fig.text(.5, .025, "Green bands: frames with an exact copy of the cue feature. No future values enter online WR scores.\n"
             "Examples selected by earliest cue time, then episode ID, separately within cue-loss and cue-retained groups.",
             ha="center", va="bottom", fontsize=9, color="#4a5055")
    fig.subplots_adjust(left=.085, right=.98, top=.84, bottom=.16, wspace=.24, hspace=.17)
    save(fig, destination, "wr_badcase_timeline")
    return [c["episode_id"] for c in cases]


def render(output):
    style()
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    metrics_figure(read(output / "summary.json"), read(output / "episode_results.json"), figures)
    return timeline_figure(read(output / "wr_warmup_trace.json"), figures)


def verify_archive(output, manifest):
    for name, record in manifest["published_files"].items():
        if sha(output / name) != record["published_sha256"]:
            raise ValueError("Published artifact checksum mismatch: " + name)
    if any(p.suffix in (".npz", ".npy", ".pt", ".pth", ".ckpt") for p in output.rglob("*")):
        raise ValueError("Binary input or checkpoint unexpectedly entered the portable archive")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "build/literature-baselines-20261022/dev-v3")
    parser.add_argument("--output", type=Path, default=ROOT / "results/literature_baselines/dev-20261022")
    parser.add_argument("--render-only", action="store_true", help="Redraw using only portable JSON files")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    transformations = {}
    if args.render_only:
        manifest = read(output / "manifest.json")
        verify_archive(output, manifest)
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Refuse to overwrite an existing archive; use --render-only for figures")
        summary, protocol = read(source / "summary.json"), read(source / "protocol.json")
        rows, training = read(source / "episode_results.json"), read(source / "training.json")
        if (summary["status"] != "completed_development_diagnostic" or summary["test_arrays_read"] != 0
                or summary["fresh_generalization_claim"] or summary["protocol_sha256"] != sha(source / "protocol.json")
                or summary["source_sha256"] != protocol["source_sha256"]):
            raise ValueError("Source is not the declared completed development-only artifact")
        events = [json.loads(line) for line in (source / "online_trace.jsonl").read_text().splitlines()]
        output.mkdir(parents=True, exist_ok=True)
        direct = ("summary.json", "protocol.json", "episode_results.json", "oracle_subsets.json")
        raw_names = (*direct, "training.json", "online_trace.jsonl", "attention_training_metadata.json")
        raw_sources = {name: source_info(source / name) for name in raw_names}
        for name in direct:
            write(output / name, scrub(read(source / name)))
            transformations[name] = {"raw_sources": [raw_sources[name]],
                "transformation": "JSON values preserved; machine-specific repository prefix replaced by $REPOSITORY."}
        write(output / "wr_warmup_trace.json", wr_trace(rows, events))
        transformations["wr_warmup_trace.json"] = {"raw_sources": [raw_sources["online_trace.jsonl"], raw_sources["episode_results.json"]],
            "transformation": "WR prefixes only; immutable score/threshold per frame and first observed event-confirmation time; resident-ID timelines retained."}
        diagnostics = [{k: r[k] for k in ("episode_id", "pair", "seed", "query", "kept_times", "mean_mse", "local_regret", "teacher_diagnostic")}
                       for r in rows if r["policy"] == "pafu"]
        write(output / "pafu_teacher_diagnostic.json", diagnostics)
        transformations["pafu_teacher_diagnostic.json"] = {"raw_sources": [raw_sources["episode_results.json"]],
            "transformation": "Select PaFu query rows and existing offline teacher diagnostics; no model evaluation."}
        receipt = {"records": training["records"], "training_pairs": training["training_pairs"],
            "students": [{"seed": s["seed"], "epochs": len(s["loss_curve"])-1,
                "initial_loss": s["loss_curve"][0], "final_loss": s["loss_curve"][-1],
                "parameter_bytes": s["parameter_bytes"]} for s in training["students"]]}
        write(output / "training_receipt.json", receipt)
        transformations["training_receipt.json"] = {"raw_sources": [raw_sources["training.json"]],
            "transformation": "Training counts, seeds, fixed epoch count, initial/final loss and parameter bytes only."}
        excluded = {p.name: source_info(p) for p in sorted(source.iterdir()) if p.suffix in (".npz", ".pt")}
        manifest = {"schema_version": 1, "purpose": "Portable dev-v3 evidence and two scientific figures; no fresh experiments.",
            "created_utc": datetime.now(timezone.utc).isoformat(), "raw_sources": raw_sources,
            "excluded_binary_artifacts": excluded, "original_experiment_source_sha256": protocol["source_sha256"],
            "internal_hash_references": "Hashes inside published JSON refer to raw experiment sources, not path-redacted copies.",
            "published_files": {}}
        readme = """# Literature baseline development evidence

Frozen Gate predictor, R4/K2; four reused validation pairs, not fresh confirmation.
PaFu rows include three student seeds. These are mechanism adaptations, not full
paper reproductions. CPU development only; equal K does not mean equal total cost.

- [Summary](summary.json), [protocol](protocol.json), [per-case results](episode_results.json).
- [Recorded oracle subsets](oracle_subsets.json): evaluator-only K2 hindsight/local references.
- [WR causal warm-up trace](wr_warmup_trace.json), [PaFu teacher diagnostics](pafu_teacher_diagnostic.json).
- [Training receipt](training_receipt.json), [provenance manifest](manifest.json).
- [Metrics figure](figures/metrics.png) ([PDF](figures/metrics.pdf)).
- [WR timeline](figures/wr_badcase_timeline.png) ([PDF](figures/wr_badcase_timeline.pdf)).

Model weights, training arrays and prediction vectors are excluded; source
hashes are retained. Nothing in this export is a new training/evaluation result.
WR examples are selected by earliest cue time, then episode ID, within failure
and success groups. Figure dots are source-pair means, not independent seed rows.

From the repository root, redraw both figures without raw data or model files:

```sh
python experiments/phase2/export_literature_report.py --render-only
```
"""
        (output / "README.md").write_text(readme)
        transformations["README.md"] = {"raw_sources": [], "transformation": "Archive index and rendering instructions; no new result claims."}
    selected = render(output)
    manifest["figure_generation"] = {"exporter": relative(__file__), "exporter_sha256": sha(__file__),
        "matplotlib_version": matplotlib.__version__, "numpy_version": np.__version__,
        "timeline_episode_ids": selected,
        "inputs": ["summary.json", "episode_results.json", "wr_warmup_trace.json"],
        "rendered_utc": datetime.now(timezone.utc).isoformat()}
    for path in sorted(p for p in output.rglob("*") if p.is_file() and p.name != "manifest.json"):
        name = str(path.relative_to(output))
        detail = transformations.get(name, manifest["published_files"].get(name, {})).copy()
        if path.parent.name == "figures":
            detail = {"derived_from_published_artifacts": manifest["figure_generation"]["inputs"],
                      "transformation": "Matplotlib visualization of existing scalar results and traces."}
        detail.update(published_sha256=sha(path), published_bytes=path.stat().st_size)
        manifest["published_files"][name] = detail
    write(output / "manifest.json", manifest)
    verify_archive(output, manifest)
    print(json.dumps({"output": relative(output), "files": len(manifest["published_files"]),
                      "published_bytes": sum(x["published_bytes"] for x in manifest["published_files"].values()),
                      "timeline_episode_ids": selected, "new_experiments_run": 0}, indent=2))


if __name__ == "__main__":
    main()
