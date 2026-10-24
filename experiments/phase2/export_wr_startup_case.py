"""Publish one existing WR startup trace with exact cached observation pixels.

This is an export, not an experiment: no model, encoder, simulator, or policy is
run. Selection is explicitly post hoc and is accompanied by the full 192-row
fresh development denominator. All displayed loss values are archived results.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def verify(path, expected):
    if sha(path) != expected:
        raise AssertionError("Source digest mismatch: " + str(path))


def arm_payload(row, raw_steps):
    shift = row["prefix"]
    cue = row["cue_original_time"] + shift
    cue_set = set().union(*map(set, row["cue_times"]))
    steps = []
    for raw in raw_steps:
        scalar = next((s for s in raw["scalars"] if s["time"] == cue), None)
        resident = bool((set(raw["recent"]) | set(raw["retained"])) & cue_set)
        assert resident == raw["cue_resident"]
        assert raw["original_time"] == raw["time"] - shift
        steps.append({
            "time": raw["time"], "original_time": raw["original_time"],
            "recent": raw["recent"], "retained": raw["retained"],
            "removed": raw["deleted"],
            "recent_original": [t-shift for t in raw["recent"]],
            "retained_original": [t-shift for t in raw["retained"]],
            "removed_original": None if raw["deleted"] is None else raw["deleted"]-shift,
            "cue_resident": resident,
            "cue_copy_times_original": sorted((set(raw["recent"]) | set(raw["retained"])) & cue_set),
            "confirmed": None if scalar is None else scalar["event"],
            "surprise": None if scalar is None else scalar["score"],
            "threshold": None if scalar is None else scalar["threshold"],
            "score_valid": None if scalar is None else scalar["valid"],
            "cue_onset_scalar_present": scalar is not None,
            "confirmed_original_times": [s["time"]-shift for s in raw["scalars"] if s["event"]],
        })
        steps[-1]["cue_copy_times_original"] = [t-shift for t in steps[-1]["cue_copy_times_original"]]
    assert [s["time"] for s in steps] == list(range(row["query"]+1))
    first_loss = next((s["time"] for s in steps if s["time"] >= cue and not s["cue_resident"]), None)
    assert row["cue_first_loss"] == [first_loss]
    confirmed = next((s["time"] for s in steps if s["confirmed"]), None)
    assert confirmed == row["cue_first_confirmed_event"]
    first_frame_removed = next((s["time"] for s in steps if s["removed"] == cue), None)
    recent_exit = next((s["time"] for s in steps if s["time"] >= cue and cue not in s["recent"]), None)
    return {
        "prefix": shift, "query": row["query"], "original_query": row["original_query"],
        "steps": steps, "feature_mse": row["feature_mse"], "mean_mse": row["mean_mse"],
        "alpha": None,
        "alpha_note": "This WR export has no archived alpha diagnostic; it is not inferred from pictures or MSE.",
        "kept_times": row["kept_times"], "kept_original_times": row["kept_original_times"],
        "cue_survived": row["cue_survived"], "cue_onset_record": row["cue_onset_record"],
        "cue_first_loss": first_loss,
        "cue_first_loss_original": None if first_loss is None else first_loss-shift,
        "cue_first_confirmed_event": confirmed,
        "cue_first_confirmed_original": None if confirmed is None else confirmed-shift,
        "cue_first_frame_removed_original": None if first_frame_removed is None else first_frame_removed-shift,
        "cue_first_frame_recent_exit_original": recent_exit-shift,
        "logical_memory_maxima": row["logical_memory_maxima"],
        "online_seconds": row["online_seconds"],
    }


def figure(output, case, rgb):
    fig = plt.figure(figsize=(13.8, 11.5))
    grid = fig.add_gridspec(4, 4, height_ratios=[1.15, 1.0, .8, 1.15],
                           left=.10, right=.97, top=.87, bottom=.12, hspace=.65, wspace=.75)
    selected_times = [0, 2, 5, case["query"]]
    titles = ["First observed frame", "Cue visible (copies t=2,3,4)", "Cue no longer visible", "Query: same observed RGB"]
    for i, t in enumerate(selected_times):
        ax = fig.add_subplot(grid[0, i])
        ax.imshow(rgb[case["rgb_indices"][t]], interpolation="nearest")
        ax.set_title(f"{titles[i]}\nOriginal t={t}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    timeline = fig.add_subplot(grid[1, :])
    states = []
    for key in ("cold", "prefix4"):
        group = []
        for step in case["arms"][key]["steps"]:
            if step["original_time"] < 0:
                continue
            c = set(case["evidence_equivalent_times"])
            group.append(1 if c & set(step["recent_original"]) else
                         2 if c & set(step["retained_original"]) else 0)
        states.append(group)
    timeline.imshow(states, aspect="auto", interpolation="nearest", vmin=0, vmax=2,
                    cmap=ListedColormap(["#e3e8ed", "#65a3c6", "#4f8e6b"]), extent=[-.5, case["query"]+.5, 1.5, -.5])
    timeline.set_yticks([0, 1], ["Cold start", "+4 common prefix"])
    timeline.set_xticks(range(case["query"]+1)); timeline.tick_params(labelsize=9)
    timeline.set_xlabel("Original time t; prefix execution time = t + 4", fontsize=10)
    timeline.set_title("Where is at least one equivalent cue copy?   Grey: absent   Blue: recent R=4   Green: retained K=2", fontsize=10, loc="left", pad=30)
    for t, label in [(2, "cue observed"), (8, "t2 copy deleted"), (10, "last copy deleted"), (17, "query")]:
        timeline.axvline(t, ymin=.04, ymax=.96, color="#263746", lw=.7, ls="--")
        timeline.text(t, -.65, label, rotation=0, ha="center", va="bottom", fontsize=8)
    note = fig.add_subplot(grid[2, :]); note.axis("off")
    note.text(0, .94, "SAME onset surprise = 5789.924087; different startup state", fontsize=12, weight="semibold", va="top")
    note.text(0, .58, "Cold: threshold unavailable at original t=2; never event-confirmed.\n"
              "Prefix: threshold = 0 at original t=2 / execution 6; confirmed one step later at original t=3 / execution 7.", fontsize=10, va="top")
    note.text(0, .04, "Onset t=2 leaves recent at t=6, is deleted from K at t=8; equivalent t=3 and t=4 copies delay complete loss until t=10.", fontsize=9, color="#53606c", va="top")
    mse = fig.add_subplot(grid[3, :2])
    xs = np.arange(3); width = .32
    for i, (key, label, color) in enumerate((("cold", "Cold", "#b36a5f"), ("prefix4", "Common prefix", "#4f8e6b"))):
        arm = case["arms"][key]; values = arm["feature_mse"]+[arm["mean_mse"]]
        bars = mse.bar(xs+(i-.5)*width, values, width, color=color, label=label)
        for bar, value in zip(bars, values):
            mse.text(bar.get_x()+bar.get_width()/2, value+.002, f"{value:.4f}", ha="center", fontsize=8)
    mse.set_xticks(xs, ["h=3", "h=6", "shared mean"]); mse.set_ylabel("Archived feature MSE (lower is better)", fontsize=9)
    mse.set_ylim(0, .14); mse.legend(frameon=False, fontsize=9, loc="upper left")
    mse.set_title("Selected saved prediction: 0.086862 → 0.061073", fontsize=10, loc="left")
    pop = fig.add_subplot(grid[3, 2:])
    pop.barh([0], [41], color="#4f8e6b", label="MSE improves")
    pop.barh([0], [21], left=[41], color="#b36a5f", label="MSE worsens")
    pop.text(20.5, 0, "41 improve", color="white", ha="center", va="center", fontsize=10)
    pop.text(51.5, 0, "21 worsen", color="white", ha="center", va="center", fontsize=10)
    pop.set_xlim(0, 62); pop.set_ylim(-1, 1); pop.set_yticks([]); pop.set_xticks([0, 41, 62])
    pop.set_xlabel("All 62 rescued early-cue episodes (not just this example)", fontsize=9)
    pop.set_title("192 fresh episodes: 62 rescued; 130 later cues unchanged", fontsize=10, loc="left")
    pop.text(0, -.80, "Post hoc illustrative case: earliest cue onset, then smallest ID\namong the 41 improved rescues. Not a held-out success estimate.", fontsize=9, color="#53606c")
    for ax in [mse, pop]: ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("WR startup failure: observed cue is not event-protected, then all copies are evicted", x=.10, y=.97, ha="left", fontsize=15, weight="semibold")
    fig.text(.10, .925, f"Episode {case['episode_id']}  |  frozen Gate reader  |  R=4, K=2 in both arms", fontsize=11, color="#53606c")
    fig.text(.10, .035, "Exact cached RGB observations only; no predicted RGB, new model execution, or oracle animation.\n"
             "Common prefix repeats the first already-observed frame four times with zero actions. It adds processing; this is not equal-total-cost superiority.\n"
             "Selected-case peak logical bytes: 124,215 → 124,255 (+40). Dataset maxima: 124,431 → 124,463 (+32). Shift-only model check: 896/896 exact.", fontsize=9, color="#53606c")
    fig.savefig(output/"wr_startup_case.png", dpi=180)
    fig.savefig(output/"wr_startup_case.pdf")
    plt.close(fig)


def run(args):
    source, data, output = args.source.resolve(), args.data.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True); (output/"frames").mkdir(exist_ok=True)
    manifest = read(source/"manifest.json")
    used = ["episode_results.json", "wr_trace.jsonl.gz", "summary.json", "protocol.json"]
    for name in used:
        verify(source/name, manifest["published_files"][name]["sha256"])
    data_manifest = read(data/"manifest.json")
    for name in ["records.json", "unique_rgb.npz", "unique_rgb_sha256.json"]:
        verify(data/name, data_manifest["artifact_sha256"][name])
    rows = read(source/"episode_results.json")
    keyed = {(r["episode_id"], r["prefix"], r["policy"]): r for r in rows}
    fresh = [r for r in rows if r["split"] == "fresh_development" and r["policy"] == "wr" and r["prefix"] == 0]
    rescued = [r for r in fresh if not r["cue_survived"] and keyed[(r["episode_id"], 4, "wr")]["cue_survived"]]
    improved = [r for r in rescued if keyed[(r["episode_id"], 4, "wr")]["mean_mse"] < r["mean_mse"]]
    worsened = [r for r in rescued if keyed[(r["episode_id"], 4, "wr")]["mean_mse"] > r["mean_mse"]]
    assert (len(fresh), len(rescued), len(improved), len(worsened)) == (192, 62, 41, 21)
    later = [r for r in fresh if r["cue_original_time"] >= 4]
    assert len(later) == 130
    for row in later:
        prefixed = keyed[(row["episode_id"], 4, "wr")]
        assert row["cue_survived"] and prefixed["cue_survived"]
        assert row["kept_original_times"] == prefixed["kept_original_times"]
        assert row["mean_mse"] == prefixed["mean_mse"]
    for row in rescued:
        prefixed = keyed[(row["episode_id"], 4, "wr")]
        assert row["cue_original_time"] in (2, 3)
        assert row["cue_onset_record"]["threshold"] is None
        assert row["cue_first_confirmed_event"] is None
        assert prefixed["cue_onset_record"]["threshold"] is not None
        assert prefixed["cue_first_confirmed_event"] is not None
    selected = sorted(improved, key=lambda r: (r["cue_original_time"], r["episode_id"]))[0]
    episode = selected["episode_id"]
    trace = {0: [], 4: []}
    with gzip.open(source/"wr_trace.jsonl.gz", "rt") as f:
        for line in f:
            raw = json.loads(line)
            if raw["episode_id"] == episode:
                trace[raw["prefix"]].append(raw)
    arms = {key: arm_payload(keyed[(episode, shift, "wr")], trace[shift]) for key, shift in [("cold", 0), ("prefix4", 4)]}
    assert arms["cold"]["cue_first_frame_removed_original"] == 8
    assert arms["cold"]["cue_first_loss_original"] == 10
    assert arms["prefix4"]["cue_first_confirmed_original"] == 3
    record = next(r for r in read(data/"records.json") if r["id"] == episode)
    verify(data/record["eval_path"], record["eval_sha256"])
    with np.load(data/record["eval_path"]) as ev:
        evidence = ev["evidence"].tolist()
    with np.load(data/"unique_rgb.npz") as archive:
        rgb = archive["rgb"]
    pixel_hashes = read(data/"unique_rgb_sha256.json")
    frames = []
    for t, idx in enumerate(record["rgb_indices"]):
        pixels = rgb[idx]
        assert hashlib.sha256(pixels.tobytes()).hexdigest() == pixel_hashes[idx]
        name = f"frames/t{t:03d}.png"
        Image.fromarray(pixels).save(output/name)
        assert np.array_equal(np.asarray(Image.open(output/name).convert("RGB")), pixels)
        frames.append({"original_time": t, "path": name, "rgb_index": idx,
                       "rgb_sha256": pixel_hashes[idx], "png_sha256": sha(output/name),
                       "role": "observed" if t <= record["query"] else "true_future_evaluation_only"})
    full_summary = read(source/"summary.json")
    max_bytes = {str(p): max(r["logical_memory_maxima"]["total_logical_bytes"] for r in rows if r["policy"] == "wr" and r["prefix"] == p) for p in [0, 4]}
    assert max_bytes == {"0": 124431, "4": 124463}
    case = {
        "schema_version": 1, "case": "wr_startup", "episode_id": episode,
        "pair": selected["pair"], "condition": selected["condition"], "split": "fresh_development",
        "query": record["query"], "cue_onset": selected["cue_original_time"], "evidence": evidence,
        "evidence_equivalent_times": selected["cue_times"][0],
        "clock_note": "Original t maps to execution t+prefix. Negative original times are prefix copies of the first observed t=0 frame, not previously unseen observations.",
        "scalar_note": "surprise/threshold/confirmed refer to the onset copy (original t=2). null after deletion means its scalar is no longer resident; no future reconstruction is inserted.",
        "selection": {"posthoc": True, "eligible_episodes": len(improved),
                      "rule": "Among rescued fresh WR episodes with strict MSE improvement, sort by original cue onset then episode_id and take first.",
                      "purpose": "Illustrate an already-measured storage-stage failure; not a prevalence or held-out performance estimate."},
        "arms": arms, "frames": frames, "rgb_indices": record["rgb_indices"],
        "milestones": [
            {"original_time": 2, "label_zh": "看到 cue：冷启动无阈值"},
            {"original_time": 3, "label_zh": "prefix 对照确认事件"},
            {"original_time": 6, "label_zh": "cue 首帧离开 recent"},
            {"original_time": 8, "label_zh": "首份 cue 被删，其余副本尚在"},
            {"original_time": 10, "label_zh": "最后副本丢失"},
            {"original_time": 17, "label_zh": "query：比较最终槽位"}],
        "population": {"fresh_episodes": 192, "early_cue_episodes": 62, "rescued": 62, "mse_improved": 41,
                       "mse_worsened": 21, "mse_tied": 0, "later_cue_episodes": 130, "later_unchanged": 130},
        "costs": {"recent_R": 4, "retained_K": 2, "extra_executed_steps": 4,
                  "same_information": "Repeat first already-observed frame; zero executed actions in added prefix.",
                  "dataset_max_logical_bytes": max_bytes,
                  "dataset_max_extra_bytes": 32,
                  "case_extra_peak_bytes": arms["prefix4"]["logical_memory_maxima"]["total_logical_bytes"]-arms["cold"]["logical_memory_maxima"]["total_logical_bytes"],
                  "equal_total_cost": False,
                  "shift_only_checks": full_summary["frozen_model_shift_checks"],
                  "shift_only_prediction_max_abs_error": full_summary["shift_prediction_max_abs_error"]},
        "figure_png": "wr_startup_case.png", "figure_pdf": "wr_startup_case.pdf",
        "limitations": ["Same Gate visual task, different failure stage; not a new visual environment.",
                        "Observed early-cue startup failure is not a demonstrated high-surprise distractor failure.",
                        "Retaining evidence does not guarantee lower feature MSE: 21 of 62 rescues worsen.",
                        "Feature MSE is not a physical-coordinate or planning-success measurement.",
                        "This diagnostic changes input processing and scalar-state cost; it is not equal-total-cost superiority.",
                        "No new held-out test, oracle search, alpha computation, prediction, or policy evaluation was run."],
    }
    save(output/"case.json", case)
    save(output/"summary.json", {k: case[k] for k in ["schema_version", "episode_id", "selection", "population", "costs", "limitations"]})
    figure(output, case, rgb)
    (output/"README.md").write_text("""# WR startup: an existing storage-stage failure

This package exports an already-completed diagnostic. It runs no model, policy,
encoder, simulator, training, or oracle search. `case.json` contains every saved
WR step for both input versions and paths to exact cached PNG observations.

The selected fresh-development episode is `1eb3ccff25424834a0780a10`. Selection
is post hoc: among the 41 rescued episodes whose archived feature MSE improved,
sort by cue onset and episode ID, then take the first. This is an explanatory
case, not a held-out performance estimate.

Original cue copies occur at t=2,3,4. The cold-start onset has a valid surprise
score but no available MAD threshold. Original t=2 exits recent at t=6 and is
deleted from retained memory at t=8; other equivalent copies remain until t=10.
Thus **t=10 is the first complete cue loss**, not t=8. In the shared-prefix arm,
the onset event is confirmed at original t=3 / execution time 7. The final
retained original timestamps are cold [12,13], prefix [2,13]; query t=17.

Archived mean h3/h6 feature MSE: 0.08686156369655085 → 0.06107325690368278.
No alpha diagnostic is inferred, and pictures are never predicted RGB.

The full denominator is 192 fresh episodes: 62 early cues rescued (41 MSE
improvements, 21 worsened), 130 later-cue episodes with unchanged memory/loss.
Fixing storage therefore does not guarantee improved prediction.

Both arms have R=4/K=2, but the common four-frame prefix adds processing and
scalar state. It repeats the first already-observed frame with zero added
actions. Original t maps to execution t+4; negative original times denote
prefix copies, not new observations. Selected-case peak logical state rises
124,215→124,255 bytes; maxima across the complete WR diagnostic rise
124,431→124,463. This is not equal-total-cost superiority. The archived 896
shift-only frozen-predictor checks have exactly zero prediction difference.

Reproduce with `.venv/bin/python experiments/phase2/export_wr_startup_case.py`.
Input hashes, PNG pixel roundtrips, trace residency, first loss, population
counts and output hashes are verified or recorded in `provenance.json`.
""")
    provenance = {"source_result_dir": str(source.relative_to(ROOT)), "source_data_dir": str(data.relative_to(ROOT)),
                  "export_script": str(Path(__file__).resolve().relative_to(ROOT)), "export_script_sha256": sha(__file__),
                  "sources": {str((source/name).relative_to(ROOT)): sha(source/name) for name in used},
                  "data_sources": {str((data/name).relative_to(ROOT)): sha(data/name) for name in ["manifest.json", "records.json", "unique_rgb.npz", "unique_rgb_sha256.json", record["eval_path"]]},
                  "new_model_calls": 0, "new_encoder_calls": 0, "new_policy_replays": 0,
                  "exact_png_pixel_roundtrips": len(frames), "source_manifest_digests_verified": True,
                  "trace_residency_and_first_loss_verified": True,
                  "outputs": {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name != "provenance.json"}}
    save(output/"provenance.json", provenance)
    print(json.dumps({"output": str(output), "episode_id": episode, "png_frames": len(frames),
                      "cold_mse": arms["cold"]["mean_mse"], "prefix_mse": arms["prefix4"]["mean_mse"],
                      "first_complete_loss_original": arms["cold"]["cue_first_loss_original"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT/"results/literature_diagnosis/wr-startup-20261022")
    parser.add_argument("--data", type=Path, default=ROOT/"build/literature-baselines-20261022/fresh-data")
    parser.add_argument("--output", type=Path, default=ROOT/"results/literature_diagnosis/visual-casebook-20261024/wr")
    run(parser.parse_args())
