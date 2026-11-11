"""Build the bounded-memory panel that lists every project baseline and every literature baseline.

    python reports/memory_results_20261111/build_panel.py [fragment_for_publishing.html]

Every number is read from a result file or a project report; nothing is typed in here:

  Gate       results/memory_baselines/gate_team/summary.json
  Synergy    results/memory_baselines/synergy/summary.json            (literature baselines)
             results/memory_baselines/synergy/released_scores.json    (project policies, reader checks)
  Motion v2  docs/MOTION_V2_STEP3_4_RECOVERY1_REPORT_ZH.md, section 6, ordinary interval, h = 3
             docs/MOTION_V2_3_1_RESUBMISSION_REPORT_ZH.md             (recent-only arm)

The page is made with the concise-report builder (~/.claude/skills/concise-report). Two project
changes, both because this panel was asked to show all baselines rather than the strongest one:
the builder's five-method cap is lifted for this build, and panel_template.html (a copy of the
skill's template) adds per-block row lists and one heading per method group.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUILDER = Path.home() / ".claude/skills/concise-report/scripts/build_report.py"
PROJECT, LITERATURE, OURS = "Project baselines", "Literature baselines", "Ours"

METHODS = [("floor", "No history", "gray", None), ("oracle", "Oracle", "orange", None)]
METHODS += [(k, label, "aqua", PROJECT) for k, label in [
    ("fifo", "FIFO"), ("reservoir", "Reservoir"), ("buckets", "Temporal buckets"), ("diversity", "Visual diversity"),
    ("surprise", "Prediction surprise"), ("keep_first", "Keep first"), ("attention", "Attention"),
    ("proxy", "Attention-proxy scorer"), ("sensitivity", "Full-set sensitivity"),
    ("appearance", "Appearance-only student"), ("fut_set", "Future-attention student"),
    ("fut_ind", "Future-attention student"), ("shuf_set", "Shuffled-label student"),
    ("shuf_ind", "Shuffled-label student"), ("dist_set", "Distilled-sensitivity student"),
    ("dist_ind", "Distilled-sensitivity student"), ("cost_ind", "Deletion-cost student"),
    ("init_ind", "Independent students")]]
METHODS += [(k, label, "yellow", LITERATURE) for k, label in [
    ("pafu", "PaFu-KV"), ("pafu_repo", "PaFu-KV"), ("c2r", "Compress to Remember"),
    ("c2r_slots", "Compress to Remember"), ("fm", "FrameMorrow"), ("fm_slots", "FrameMorrow")]]
METHODS += [("ours", "Deletion-cost scorer", "blue", OURS)]
GROUP = {k: g for k, _, _, g in METHODS}
ORDER = [k for k, *_ in METHODS]


def figure(base, rows, higher_is_better, **extra):
    """rows: key -> (value or None, who or None, hover note or None). Sorted best first inside each group."""
    def rank(key):
        value = rows[key][0]
        group = [None, PROJECT, LITERATURE, OURS].index(GROUP[key])
        if value is None:
            return (group, 1, ORDER.index(key))
        return (group, 0, -value if higher_is_better else value) if GROUP[key] else (group, 0, ORDER.index(key))
    fig = {k: v for k, v in base.items() if k not in ("values", "who", "notes", "labels")}
    fig["rows"] = sorted(rows, key=rank)
    fig["values"] = {k: v[0] for k, v in rows.items() if v[0] is not None}
    fig["who"] = {k: v[1] for k, v in rows.items() if v[1]}
    fig["notes"] = {k: v[2] for k, v in rows.items() if v[2]}
    fig.update(extra)
    return fig


def gate(unit):
    s = json.loads((ROOT / "results/memory_baselines/gate_team/summary.json").read_text())["scores"]
    old = unit["figure"]["values"]

    def row(name, who=None, extra=""):
        m = s[name]
        cue = "" if m["cue_kept"] is None else f" Cue frame kept in {100 * m['cue_kept']:.1f}% of episodes."
        return (round(100 * m["route_accuracy"], 2), who, f"Latent MSE {m['normalized_mse']:.4f}.{cue}{extra}")

    pct = lambda name: f"{100 * s[name]['route_accuracy']:.1f}%"
    seeds = s["pacc_recurrent"]["route_accuracy_by_seed"]
    failed = sum(v < 0.6 for v in seeds.values())
    assert failed == 1, seeds
    full, c2r = s["full_history"]["route_accuracy_by_seed"], s["pacc_full"]["route_accuracy_by_seed"]
    assert all(c2r[k] > full[k] for k in full), (c2r, full)     # it beats the full-history predictor it copies, on every seed
    rows = {
        "floor": row("recent_only", "recent 4 frames only"), "oracle": row("oracle_keyframe", "cue frame kept by hand"),
        "fifo": row("fifo"), "reservoir": row("reservoir"), "buckets": row("temporal_buckets"),
        "diversity": row("visual_diversity"), "surprise": row("prediction_surprise"),
        "attention": row("attention", "retired from the README table"),
        "proxy": row("attention_proxy_student", "ours' twin, trained on attention labels"),
        "pafu": row("pafu_kv", f"as published; teacher {pct('pafu_teacher_top2')}"),
        "pafu_repo": row("pafu_kv_repo", "the repo's earlier adaptation"),
        "c2r": row("pacc_full", f"as published: reads the whole history; teacher {pct('full_history')}"),
        "c2r_slots": row("pacc_recurrent", f"held to 2 slots; fails on {failed} of {len(seeds)} seeds",
                         " Per seed: " + ", ".join(f"{100 * v:.1f}%" for v in seeds.values()) + "."),
        "fm": row("fm_dinov2_published", f"as published: ranks the whole history; teacher {pct('fm_dinov2_teacher_top2')}"),
        "fm_slots": row("fm_dinov2_online", "held to 2 slots"),
        "ours": row("deletion_cost_student"),
    }
    for key, name in (("floor", "floor"), ("oracle", "oracle"), ("ours", "ours"), ("diversity", "baseline")):
        assert rows[key][0] == old[name], (key, rows[key][0], old[name])     # the 10 Nov panel's numbers still hold
    unit["figure"] = figure(unit["figure"], rows, True)
    unit["footnote"] = ("Teacher is the signal a literature method is trained to copy, scored directly; Compress to Remember beats its "
                        "teacher, the predictor reading the whole history, on all three seeds. "
                        "FrameMorrow uses DINOv2 features, as published; its I-JEPA variant was not run.")


def synergy(unit):
    lit = json.loads((ROOT / "results/memory_baselines/synergy/summary.json").read_text())["readers"]
    rel = json.loads((ROOT / "results/memory_baselines/synergy/released_scores.json").read_text())["readers"]
    old = unit["figure"]["values"]

    def row(source, name, who=None):
        first, second = source["multihead"][name]["S"], source["item_attention"][name]["S"]
        lost = first["needed_evidence_lost"]
        note = "Deletes nothing, so it has no needed-card rate." if lost is None else f"Needed card deleted {100 * lost:.1f}% of the time."
        return (round(first["regret"], 4), who, f"{note} Second reader: regret {second['regret']:.4f}")

    reg = lambda source, name: f"{source['multihead'][name]['S']['regret']:.4f}"
    n = rel["multihead"]["fifo"]["S"]["n"]
    three = rel["multihead"]["three_cards"]["S"]["regret"]
    check = max(rel[r]["deletion_path_check"]["max_abs_diff_from_released_cost"] for r in rel)
    assert check < 1e-5, check                         # the compressor's scoring path is on the released cost scale
    rows = {
        "floor": row(rel, "fifo", "delete the oldest card"), "oracle": row(rel, "oracle", "lowest-error deletion"),
        "attention": row(rel, "attention"), "sensitivity": row(rel, "full_set_sensitivity"),
        "appearance": row(rel, "appearance_only_student"),
        "fut_set": row(rel, "future_attention_student_set", "set-conditioned"),
        "fut_ind": row(rel, "future_attention_student_independent", "independent scoring"),
        "shuf_set": row(rel, "shuffled_label_student_set", "set-conditioned"),
        "shuf_ind": row(rel, "shuffled_label_student_independent", "independent scoring"),
        "dist_set": row(rel, "distilled_sensitivity_student_set", "set-conditioned"),
        "dist_ind": row(rel, "distilled_sensitivity_student_independent", "independent scoring"),
        "cost_ind": row(rel, "deletion_cost_student_independent", "ours with independent scoring"),
        "pafu": row(lit, "pafu_kv", f"teacher {reg(lit, 'pafu_teacher_top2')}"),
        "c2r_slots": row(lit, "pacc", f"2 learned slots from all 3 cards; teacher {three:+.4f}"),
        "fm": row(lit, "fm_published", f"teacher {reg(lit, 'fm_teacher_top2')}"),
        "ours": row(rel, "deletion_cost_student_set", "set-conditioned"),
    }
    for key, name in (("floor", "floor"), ("ours", "ours"), ("dist_set", "baseline")):
        assert rows[key][0] == old[name], (key, rows[key][0], old[name])
    assert rows["ours"][0] == round(lit["multihead"]["deletion_cost_student"]["S"]["regret"], 4)
    unit["figure"] = figure(unit["figure"], rows, False, labels={"floor": "No guidance", "ours": "Deletion-cost student"},
                            metric=f"Prediction regret, locked test, {n:,} synergy episodes",
                            direction="error above the best single deletion; lower is better")
    unit["footnote"] = (f"Regret is error above the best single deletion, not an MSE, so it can be negative. Compress to Remember "
                        f"scores below its teacher, the reader given all three cards ({three:+.4f}), as it does on Gate; "
                        f"the cause is untested.")


def motion(unit):
    def table(path, start, stop, cells_per_row, position):
        section = (ROOT / path).read_text().split(start, 1)[1].split(stop, 1)[0]
        out = {}
        for line in section.splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) == cells_per_row and cells[1] == "3" and cells[0] not in out:
                out[cells[0]] = (float(cells[position].split()[0]), cells)
        return out

    main = table("docs/MOTION_V2_STEP3_4_RECOVERY1_REPORT_ZH.md", "### 普通间隔（主要）", "### Long gap", 5, 3)
    ref = table("docs/MOTION_V2_3_1_RESUBMISSION_REPORT_ZH.md", "a 为 recent-only", "六项必需门槛按原始", 6, 4)
    old = unit["figure"]["values"]

    def row(name, who=None):
        value, cells = main[name]
        return (round(value, 4), who, f"Keeps both cue frames in {float(cells[2].split()[0]):.1f}% of episodes")

    def tied(prefix):                      # arms that kept their initial weights are identical; list them once
        arms = {k: v[0] for k, v in main.items() if k.startswith(prefix) and "INIT" in k}
        assert len(set(arms.values())) == 1, arms
        return next(iter(arms)), len(arms)

    floor, oracle = ref["a"][0], main["参照/b"][0]
    assert abs(floor - ref["a−b"][0] - oracle) < 1e-9      # same items: recent-only minus (a − b) is the section-6 oracle
    ind, n_ind = tied("independent/")
    ours, n_set = tied("set/")
    assert "set/use_time_position **INIT**" in main and n_set == 8 and n_ind == 8
    rows = {
        "floor": (round(floor, 4), "recent 4 frames only", None), "oracle": row("参照/b", "two cue frames kept by hand"),
        "fifo": row("fifo"), "reservoir": row("reservoir"), "buckets": row("temporal_buckets"),
        "diversity": row("diversity"), "surprise": row("surprise"), "keep_first": row("keep_first"),
        "attention": row("attention"),
        "fut_set": row("set/future_attention", "set-conditioned"),
        "fut_ind": row("independent/future_attention", "independent scoring"),
        "shuf_set": row(ours, "set-conditioned; untrained, same as ours"),
        "init_ind": row(ind, f"{n_ind} arms, all untrained"),
        "pafu": (None, None, None), "c2r": (None, None, None), "fm": (None, None, None),
        "ours": row(ours, "untrained"),
    }
    for key, name in (("floor", "floor"), ("oracle", "oracle"), ("attention", "baseline"), ("ours", "ours")):
        assert rows[key][0] == old[name], (key, rows[key][0], old[name])
    unit["figure"] = figure(unit["figure"], rows, False, labels={"ours": "Deletion-cost student"})
    unit["footnote"] = ("No deletion-cost or shuffled-label student trained past its initial weights, ours included. "
                        "Literature baselines were not run: the frozen reader and features are on a teammate's storage.")


def pusht(unit):
    who = unit["figure"].get("who", {})
    rows = {"floor": (None, who.get("floor"), None), "oracle": (None, who.get("oracle"), None),
            "pafu": (None, None, None), "c2r": (None, None, None), "fm": (None, None, None),
            "ours": (None, None, None)}
    unit["figure"] = figure(unit["figure"], rows, True)


def main():
    spec = json.loads((HERE / "spec/base_10nov.json").read_text())
    units = {u["name"]: u for u in spec["units"]}
    gate(units["Gate"]); synergy(units["Synergy"]); motion(units["Motion v2"]); pusht(units["PushT, hidden centre of gravity"])
    for name in ("Synergy", "Motion v2"):                 # six-frame strips: smaller tiles, so the wider figure column fits
        units[name]["example"]["tile"] = 72
    spec.update(
        page_title="Memory Baselines Panel",
        subtitle="4 recent frames + 2 memory slots · frozen I-JEPA features · frozen predictor · 11 Nov 2026",
        intro="One bar per method, scored on the same items within a block: the two reference points, every project "
              "baseline, every literature baseline, then ours.",
        methods=[{"key": k, "label": label, "color": color, **({"group": group} if group else {})}
                 for k, label, color, group in METHODS])
    (HERE / "spec/spec.json").write_text(json.dumps(spec, indent=1, ensure_ascii=False) + "\n")

    module_spec = importlib.util.spec_from_file_location("concise_report_builder", BUILDER)
    builder = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(builder)
    builder.MAX_METHODS = len(METHODS)                    # this panel lists all baselines on request
    builder.TEMPLATE = HERE / "panel_template.html"
    check = builder.check_unit
    builder.check_unit = lambda u, keys, where, warn: check(u, keys & set(u["figure"]["rows"]), where, warn)
    targets = [[str(HERE / "memory-panel-baselines.html"), "--standalone"]] + [[p] for p in sys.argv[1:2]]
    for target in targets:
        sys.argv = ["build_report.py", str(HERE / "spec/spec.json"), *target]
        builder.main()


if __name__ == "__main__":
    main()
