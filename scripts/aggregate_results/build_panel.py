"""Recreate the results panel from one run directory.

    python scripts/aggregate_results/build_panel.py --run runs/<run_id>

Needs <run>/aggregate/summary.json (scripts/aggregate_results/aggregate.py). Every bar comes from
the run's replay tables; nothing is read from results/ and retrained models never enter the panel.
The block texts, example frames and page template are the static parts of the 11 Nov panel
(reports/memory_results_20261111/spec/base_10nov.json, spec/img/, panel_template.html), read-only.
A row whose seeds are not all present shows "pending, k/n" instead of a number.

Writes <run>/aggregate/panel.html (opens locally), panel_spec.json and panel_check.json, which
compares every bar with the reference panel at the precision the panel displays.
"""
import argparse
import base64
import html
import json
import mimetypes
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/memory_results_20261111"
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
HEAD = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
        "<style>body{margin:0;font:14px system-ui,sans-serif}img{max-width:100%}"
        "[hidden]{display:none!important}</style></head><body>")
WHO_GATE = {"floor": "recent 4 frames only", "oracle": "cue frame kept by hand", "attention": "retired from the README table",
            "proxy": "ours' twin, trained on attention labels", "pafu_repo": "the repo's earlier adaptation",
            "fm_slots": "held to 2 slots"}
WHO_SYNERGY = {"floor": "delete the oldest card", "oracle": "lowest-error deletion", "fut_set": "set-conditioned",
               "fut_ind": "independent scoring", "shuf_set": "set-conditioned", "shuf_ind": "independent scoring",
               "dist_set": "set-conditioned", "dist_ind": "independent scoring", "cost_ind": "ours with independent scoring",
               "ours": "set-conditioned"}
MOTION = {"oracle": "reference/b", "fifo": "baseline/fifo", "reservoir": "baseline/reservoir", "buckets": "baseline/temporal_buckets",
          "diversity": "baseline/diversity", "surprise": "baseline/surprise", "keep_first": "baseline/keep_first",
          "attention": "baseline/attention", "fut_set": "student/set_future_attention",
          "fut_ind": "student/independent_future_attention", "ours": "student/set_use_time_position"}


def figure(base, rows, higher_is_better, **extra):
    """rows: key -> {value, who, note, pending}. Sorted best first inside each group, like the reference panel."""
    def rank(key):
        value = rows[key].get("value")
        group = [None, PROJECT, LITERATURE, OURS].index(GROUP[key])
        if value is None:
            return (group, 1, ORDER.index(key))
        return (group, 0, -value if higher_is_better else value) if GROUP[key] else (group, 0, ORDER.index(key))
    fig = {k: v for k, v in base.items() if k not in ("values", "who", "notes", "labels", "pending", "rows")}
    fig["rows"] = sorted(rows, key=rank)
    fig["values"] = {k: v["value"] for k, v in rows.items() if v.get("value") is not None}
    fig["who"] = {k: v["who"] for k, v in rows.items() if v.get("who")}
    fig["notes"] = {k: v["note"] for k, v in rows.items() if v.get("note")}
    fig["pending"] = {k: v["pending"] for k, v in rows.items() if v.get("pending")}
    fig.update(extra)
    return fig


def gate(unit, table, results):
    rows = {}
    pct = lambda name: f"{100 * table[name]['route_accuracy']:.1f}%" if table.get(name, {}).get("complete") else "pending"
    for r in results:
        key = r.get("panel_key")
        if not key:
            continue
        e = table.get(r["id"], {})
        row = {"who": WHO_GATE.get(key)}
        if e.get("complete"):
            cue = "" if e["cue_kept"] is None else f" Cue frame kept in {100 * e['cue_kept']:.1f}% of episodes."
            row.update(value=round(100 * e["route_accuracy"], 2), note=f"Latent MSE {e['normalized_mse']:.4f}.{cue}")
        else:
            row["pending"] = f"{len(e.get('seeds_present', []))}/{len(e.get('seeds_expected', [1, 2, 3]))} seeds"
        rows[key] = row
    rows["pafu"]["who"] = f"as published; teacher {pct('pafu_teacher_top2')}"
    rows["c2r"]["who"] = f"as published: reads the whole history; teacher {pct('full_history')}"
    rows["fm"]["who"] = f"as published: ranks the whole history; teacher {pct('fm_dinov2_teacher_top2')}"
    slots = table.get("pacc_recurrent", {})
    if slots.get("complete"):
        seeds = slots["route_accuracy_by_seed"]
        failed = sum(v < 0.6 for v in seeds.values())
        rows["c2r_slots"]["who"] = "held to 2 slots" + (f"; fails on {failed} of {len(seeds)} seeds" if failed else "")
        rows["c2r_slots"]["note"] += " Per seed: " + ", ".join(f"{100 * v:.1f}%" for v in seeds.values()) + "."
    else:
        rows["c2r_slots"]["who"] = "held to 2 slots"
    unit["figure"] = figure(unit["figure"], rows, True)
    c2r, full = table.get("pacc_full", {}), table.get("full_history", {})
    beats = c2r.get("complete") and full.get("complete") and all(
        c2r["route_accuracy_by_seed"][s] > full["route_accuracy_by_seed"][s] for s in full["route_accuracy_by_seed"])
    unit["footnote"] = ("Teacher is the signal a literature method is trained to copy, scored directly"
                        + ("; Compress to Remember beats its teacher, the predictor reading the whole history, on all three seeds. "
                           if beats else ". ")
                        + "FrameMorrow uses DINOv2 features, as published; its I-JEPA variant was not run.")


def synergy(unit, block, results):
    first, second = block["readers"]["multihead"], block["readers"]["item_attention"]
    rows = {}
    for r in results:
        key = r.get("panel_key")
        if not key:
            continue
        e = first.get(r["id"], {})
        row = {"who": WHO_SYNERGY.get(key)}
        if e.get("complete"):
            s = e["metrics"]["S"]
            lost = s["needed_evidence_lost"]
            note = "Deletes nothing, so it has no needed-card rate." if lost is None else f"Needed card deleted {100 * lost:.1f}% of the time."
            other = second.get(r["id"], {})
            if other.get("complete"):
                note += f" Second reader: regret {other['metrics']['S']['regret']:.4f}"
            row.update(value=round(s["regret"], 4), note=note)
        else:
            row["pending"] = f"{len(e.get('seeds_present', []))}/{len(e.get('seeds_expected', [1, 2, 3]))} seeds"
        rows[key] = row
    regret = lambda name: f"{first[name]['metrics']['S']['regret']:.4f}" if first.get(name, {}).get("complete") else "pending"
    rows["pafu"]["who"] = f"teacher {regret('pafu_teacher_top2')}"
    rows["fm"]["who"] = f"teacher {regret('fm_teacher_top2')}"
    three = first.get("three_cards", {})
    teacher = f"{three['metrics']['S']['regret']:+.4f}" if three.get("complete") else "pending"
    rows["c2r_slots"]["who"] = f"2 learned slots from all 3 cards; teacher {teacher}"
    n = next((e["metrics"]["S"]["n"] for e in first.values() if "metrics" in e), 2304)
    unit["figure"] = figure(unit["figure"], rows, False, labels={"floor": "No guidance", "ours": "Deletion-cost student"},
                            metric=f"Prediction regret, locked test, {n:,} synergy episodes",
                            direction="error above the best single deletion; lower is better")
    c2r = first.get("pacc", {})
    below = c2r.get("complete") and three.get("complete") and c2r["metrics"]["S"]["regret"] < three["metrics"]["S"]["regret"]
    unit["footnote"] = ("Regret is error above the best single deletion, not an MSE, so it can be negative. "
                        + (f"Compress to Remember scores below its teacher, the reader given all three cards ({teacher}), "
                           "as it does on Gate; the cause is untested." if below else
                           f"The reader given all three cards, which Compress to Remember copies, scores {teacher}."))


def motion(unit, block):
    rows = {k: {} for k in ("floor", *MOTION, "shuf_set", "init_ind", "pafu", "c2r", "fm")}
    if block and block.get("complete"):
        table = block["online"]["ordinary"]
        cell = lambda name, who=None: {"value": round(table[name]["position_mse"]["3"], 4), "who": who,
                                       "note": f"Keeps both cue frames in {100 * table[name]['success_rate']:.1f}% of episodes"}
        rows["floor"] = {"value": round(block["reference_arms"]["a"]["position_mse"]["3"], 4), "who": "recent 4 frames only"}
        for key, name in MOTION.items():
            rows[key] = cell(name)
        rows["oracle"]["who"] = "two cue frames kept by hand"
        rows["fut_set"]["who"], rows["fut_ind"]["who"] = "set-conditioned", "independent scoring"
        untrained = block["untrained_arms"]
        tied = lambda arch: sorted(f"student/{a.replace('/', '_')}" for a in untrained if a.startswith(arch + "/"))
        ind, sets = tied("independent"), tied("set")
        assert len({table[a]["position_mse"]["3"] for a in ind}) == 1 and len({table[a]["position_mse"]["3"] for a in sets}) == 1
        rows["shuf_set"] = cell(sets[0], "set-conditioned; untrained, same as ours")
        rows["init_ind"] = cell(ind[0], f"{len(ind)} arms, all untrained")
        rows["ours"]["who"] = "untrained"
    unit["figure"] = figure(unit["figure"], rows, False, labels={"ours": "Deletion-cost student"})
    unit["footnote"] = ("No deletion-cost or shuffled-label student trained past its initial weights, ours included. "
                        "Recomputed from the study's saved loss arrays; nothing can be rerun, and the literature baselines "
                        "were never run: the frozen reader and features are on a teammate's storage.")


def pusht(unit):
    who = unit["figure"].get("who", {})
    rows = {"floor": {"who": who.get("floor")}, "oracle": {"who": who.get("oracle")}, "pafu": {}, "c2r": {}, "fm": {}, "ours": {}}
    unit["figure"] = figure(unit["figure"], rows, True)


def embed(src, base):
    path = base / src
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--path", default="replay", help="which path's tables fill the panel (default: replay)")
    args = ap.parse_args()
    out = args.run / "aggregate"
    summary = json.loads((out / "summary.json").read_text())
    manifest = yaml.safe_load((ROOT / "configs/reproduction.yaml").read_text())
    reference = json.loads((ROOT / "configs/reference_values.json").read_text())["panel"]
    spec = json.loads((REPORT / "spec/base_10nov.json").read_text())
    units = {u["name"]: u for u in spec["units"]}
    gate(units["Gate"], summary["gate"].get(args.path, {}), manifest["results"]["gate"])
    synergy(units["Synergy"], summary["synergy"].get(args.path, {"readers": {"multihead": {}, "item_attention": {}}}),
            manifest["results"]["synergy"])
    motion(units["Motion v2"], summary["motion_v2"].get("replay"))
    pusht(units["PushT, hidden centre of gravity"])
    for name in ("Synergy", "Motion v2"):
        units[name]["example"]["tile"] = 72
    spec.update(page_title="Memory Baselines Panel",
                subtitle=f"4 recent frames + 2 memory slots · frozen I-JEPA features · frozen predictor · recreated from run {args.run.name}",
                intro="One bar per method, scored on the same items within a block: the two reference points, every project "
                      "baseline, every literature baseline, then ours.",
                methods=[{"key": k, "label": label, "color": color, **({"group": group} if group else {})}
                         for k, label, color, group in METHODS])
    (out / "panel_spec.json").write_text(json.dumps(spec, indent=1, ensure_ascii=False) + "\n")

    tolerance = {"Gate": {}, "Synergy": {}}
    for unit, env in (("Gate", "gate"), ("Synergy", "synergy")):
        for r in manifest["results"][env]:
            if r.get("panel_key"):
                tolerance[unit][r["panel_key"]] = (r.get("replay") or {}).get("tolerance", "exact")
    check, same, total, by_state = {}, 0, 0, {}
    for name, ref in reference.items():
        fig = units[name]["figure"]
        rows = {}
        for key in ref["rows"]:
            theirs, mine = ref["values"].get(key), fig["values"].get(key)
            kind = tolerance.get(name, {}).get(key, "exact")
            if key in fig["pending"]:
                state = "pending"
            elif theirs is None and mine is None:
                state = "not run in both"
            elif theirs == mine:
                state = "identical"
            elif kind == "unstable":
                state = "unstable training, differs"
            elif kind in ("per_seed", "tie_sensitive") and theirs is not None and mine is not None \
                    and abs(mine - theirs) <= (1.0 if name == "Gate" else 0.01 * max(abs(theirs), 0.01) + 0.0005):
                state = f"within tolerance ({kind})"
            else:
                state = "differs"
            rows[key] = {"reference": theirs, "recreated": mine, "tolerance": kind, "state": state}
            if theirs is not None:
                total += 1; same += state == "identical"
                by_state[state] = by_state.get(state, 0) + 1
        check[name] = {"rows": rows, "extra_rows": sorted(set(fig["rows"]) - set(ref["rows"]))}
    result = {"run": args.run.name, "path": args.path, "bars_with_a_reference_value": total, "identical_at_displayed_precision": same,
              "bars_by_state": by_state,
              "tolerance_note": "Gate bars are percentages (within 1 point); Synergy bars are regrets (within 1% of the value plus 0.0005)",
              "units": check}
    (out / "panel_check.json").write_text(json.dumps(result, indent=1) + "\n")

    for u in spec["units"]:
        for row in u.get("example", {}).get("rows", []):
            row["images"] = [embed(s, REPORT / "spec") for s in row["images"]]
    payload = json.dumps(spec, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    page = (REPORT / "panel_template.html").read_text()
    page = page.replace("__PAGE_TITLE__", html.escape(spec["page_title"])).replace("__SPEC__", payload)
    (out / "panel.html").write_text(HEAD + page + "</body></html>")
    print(f"panel: {same} of {total} bars identical to the reference panel at displayed precision; " +
          ", ".join(f"{k}: {v}" for k, v in sorted(by_state.items())))
    for name, block in check.items():
        states = [r["state"] for r in block["rows"].values()]
        print(f"  {name:34s} " + ", ".join(f"{s}: {states.count(s)}" for s in sorted(set(states))))
    print(f"wrote {out / 'panel.html'}")


if __name__ == "__main__":
    main()
