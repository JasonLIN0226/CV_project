"""Pool one run directory into tables and compare them with the frozen reference values.

    python scripts/aggregate_results/aggregate.py --run runs/<run_id>

Reads only the run directory, configs/reproduction.yaml and configs/reference_values.json; it never
opens results/. For every result in the manifest it reports, separately for each path (replay,
retrain, ...): per-seed and pooled metrics, the difference from the reference value, how many of
the expected seeds are present, and where each number came from (host, device, code hash,
checkpoint). Replayed artifacts and newly trained models are never mixed: each path has its own
table. Writes <run>/aggregate/summary.json and <run>/aggregate/tables.md.

Verdicts by tolerance class (set per result in the manifest):
    exact          deterministic replay: the primary metric must match to 1e-9 and secondary ones to 1e-5
    tie_sensitive  deterministic code whose tie-breaking depends on CPU rounding: within 0.01
    per_seed       model trained in this run: reported per seed; "within 1 point" when the pooled primary metric is within 0.01
    unstable       training known to flip between runs: no verdict, per-seed values only
A path other than replay lists only the policies that path produced.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def provenance(payloads):
    return {"files": len(payloads), "hosts": sorted({p["host"] for p in payloads}),
            "devices": sorted({p["device"] for p in payloads}),
            "source_sha256": sorted({p["source_sha256"] for p in payloads}),
            "torch": sorted({p["torch"] for p in payloads})}


def verdict(tolerance, complete, primary_diff, secondary_diff=0.0, replay=True):
    if not complete:
        return "incomplete"
    if primary_diff is None:
        return "no reference"
    if tolerance == "unstable":
        return "unstable training: read per seed"
    if tolerance == "exact" and replay:
        return "identical" if abs(primary_diff) < 1e-9 and abs(secondary_diff) < 1e-5 else "differs"
    return "within 1 point" if abs(primary_diff) <= 0.01 else "differs by more than 1 point"


# ---------------------------------------------------------------- Gate

def gate(folder, results, reference, seeds):
    payloads = [json.loads(p.read_text()) for p in sorted(folder.glob("*_seed*.json"))]
    rows, owner = defaultdict(list), defaultdict(list)
    for payload in payloads:
        for r in payload.get("rows", []):
            rows[r["policy"]].append(r); owner[r["policy"]].append(payload)
    known = {r["id"]: r for r in results}
    out = {}
    expected = {k for k, v in known.items() if v["replay"].get("task")} if folder.name == "replay" else set()
    for policy in sorted(set(rows) | expected):
        spec = known.get(policy, {})
        rr = rows.get(policy, [])
        present = sorted({r["seed"] for r in rr})
        entry = {"group": spec.get("group"), "memory": spec.get("memory"), "seeds_present": present,
                 "seeds_expected": seeds, "complete": present == seeds}
        if rr:
            per_seed = {str(s): float(np.mean([r["route_accuracy"] for r in rr if r["seed"] == s])) for s in present}
            cue = [r["cue_kept"] for r in rr if r["cue_kept"] is not None]
            files = list({id(p): p for p in owner[policy]}.values())      # one payload per file, not per row
            entry.update(route_accuracy=float(np.mean([r["route_accuracy"] for r in rr])),
                         normalized_mse=float(np.mean([r["normalized_mse"] for r in rr])),
                         cue_kept=float(np.mean(cue)) if cue else None, route_accuracy_by_seed=per_seed,
                         rows=len(rr), provenance=provenance(files))
            checks = [p["info"]["original_validation_check"] for p in files if "original_validation_check" in p.get("info", {})]
            students = [p["info"]["students"] for p in files if "students" in p.get("info", {})]
            if students:
                entry["checkpoints"] = students
            if checks:
                entry["original_validation_rows_checked"] = int(sum(v["rows"] for c in checks for v in c.values()))
                entry["original_validation_route_mismatches"] = int(sum(v["route_mismatches"] for c in checks for v in c.values()))
        ref = reference.get(policy)
        primary = secondary = None
        if ref and rr:
            primary = entry["route_accuracy"] - ref["route_accuracy"]
            secondary = entry["normalized_mse"] - ref["normalized_mse"]
            entry["reference"] = {"route_accuracy": ref["route_accuracy"], "normalized_mse": ref["normalized_mse"],
                                  "route_accuracy_by_seed": ref["route_accuracy_by_seed"]}
            entry["difference"] = {"route_accuracy": primary, "normalized_mse": secondary,
                                   "route_accuracy_by_seed": {s: per_seed[s] - ref["route_accuracy_by_seed"][s]
                                                              for s in per_seed if s in ref["route_accuracy_by_seed"]}}
        tolerance = (spec.get("replay") or {}).get("tolerance", "per_seed")
        entry["tolerance"] = tolerance
        entry["verdict"] = verdict(tolerance, entry["complete"], primary, secondary or 0.0, replay=folder.name == "replay")
        out[policy] = entry
    return out


# ---------------------------------------------------------------- Synergy

def synergy(folder, results, reference, readers, seeds):
    found = defaultdict(lambda: defaultdict(dict))     # reader -> policy -> seed -> (metrics, payload)
    for path in sorted(folder.glob("*_seed*.json")):
        payload = json.loads(path.read_text())
        for name, record in payload.get("policies", {}).items():
            found[payload["reader"]][name][record["seed"]] = (record["metrics"], payload)
    known = {r["id"]: r for r in results}
    out = {}
    for reader in readers:
        table = {}
        expected_policies = {k for k, v in known.items() if v["replay"].get("task")} if folder.name == "replay" else set()
        for policy in sorted(set(found[reader]) | expected_policies):
            spec = known.get(policy, {})
            runs = found[reader].get(policy, {})
            seeded = any(s is not None for s in runs)
            expected = seeds if (seeded or not runs and spec.get("group") in ("ours", "ablation", "literature")) else [None]
            present = sorted(runs, key=lambda s: (s is None, s))
            entry = {"group": spec.get("group"), "memory": spec.get("memory"), "seeds_present": present,
                     "seeds_expected": expected, "complete": bool(runs) and present == expected}
            pooled = {}
            if runs:
                for task in "RDS":
                    lost = [m[task]["needed_evidence_lost"] for m, _ in runs.values()]
                    pooled[task] = {"n": next(iter(runs.values()))[0][task]["n"],
                                    "needed_evidence_lost": None if lost[0] is None else float(np.mean(lost)),
                                    "regret": float(np.mean([m[task]["regret"] for m, _ in runs.values()])),
                                    "regret_by_seed": {str(s): runs[s][0][task]["regret"] for s in present},
                                    "needed_evidence_lost_by_seed": {str(s): runs[s][0][task]["needed_evidence_lost"] for s in present}}
                entry.update(metrics=pooled, provenance=provenance([p for _, p in runs.values()]))
            ref = reference[reader]["policies"].get(policy)
            primary = secondary = None
            if ref and runs:
                diffs = {}
                for task in "RDS":
                    lost = None if pooled[task]["needed_evidence_lost"] is None else \
                        pooled[task]["needed_evidence_lost"] - ref[task]["needed_evidence_lost"]
                    diffs[task] = {"needed_evidence_lost": lost, "regret": pooled[task]["regret"] - ref[task]["regret"]}
                entry["reference"], entry["difference"] = ref, diffs
                lost = [abs(d["needed_evidence_lost"]) for d in diffs.values() if d["needed_evidence_lost"] is not None]
                regret = max(abs(d["regret"]) for d in diffs.values())
                primary, secondary = (max(lost), regret) if lost else (regret, 0.0)
            tolerance = (spec.get("replay") or {}).get("tolerance", "per_seed")
            entry["tolerance"] = tolerance
            entry["verdict"] = verdict(tolerance, entry["complete"], primary, secondary or 0.0, replay=folder.name == "replay")
            table[policy] = entry
        out[reader] = table
    stage = {}
    for path in sorted(folder.glob("stage_a_*_seed*.json")):
        payload = json.loads(path.read_text())
        stage[f"{payload['reader']}/{payload['split']}"] = payload["info"].get("against_released")
    checks = {}
    for path in sorted(folder.glob("references_*_seed*.json")):
        payload = json.loads(path.read_text())
        checks[payload["reader"]] = payload["info"].get("deletion_path_check")
    training = {}
    for path in sorted(folder.glob("train_students_*_seed*.json")):
        payload = json.loads(path.read_text())
        match = payload["info"].get("selected_step_matches_released")
        training[f"{payload['reader']}/{payload['mode']}"] = {
            "arms": len(payload["info"]["arms"]),
            "arms_selecting_the_released_step": None if match is None else int(sum(match.values()))}
    return {"readers": out, "stage_a_against_released": stage, "deletion_path_check": checks, "student_training": training}


# ---------------------------------------------------------------- Motion v2

def motion(folder, reference):
    path = folder / "evidence_seed0.json"
    if not path.exists():
        return {"complete": False, "verdict": "incomplete"}
    payload = json.loads(path.read_text())
    cells, worst, success = 0, 0.0, 0.0
    for interval, table in reference["online"].items():
        for method, ref in table.items():
            mine = payload["online"][interval][method]
            for h in ("3", "6"):
                cells += 1
                worst = max(worst, abs(mine["position_mse"][h] - ref["position_mse"][h]),
                            abs(mine["feature_mse"][h] - ref["feature_mse"][h]))
            success = max(success, abs(100 * mine["success_rate"] - ref["success_rate_percent"]))
    for arm, ref in reference["reference_arms"].items():
        for h in ("3", "6"):
            cells += 1
            worst = max(worst, abs(payload["reference_arms"][arm]["position_mse"][h] - ref["position_mse"][h]))
    return {"complete": True, "online": payload["online"], "reference_arms": payload["reference_arms"],
            "student_selection": payload["student_selection"], "untrained_arms": payload["untrained_arms"],
            "published_cells_compared": cells, "max_abs_difference_from_published": worst,
            "max_abs_success_rate_difference_percent": success,
            # the reports print losses to 8 decimals and success rates to 3
            "verdict": "identical to the published precision" if worst < 5e-9 and success < 5e-4 else "differs",
            "provenance": provenance([payload])}


# ---------------------------------------------------------------- report

def markdown(summary):
    lines = [f"# Run {summary['run_id']}: aggregate tables", "",
             "Replayed artifacts and retrained models are separate tables. `diff` is this run minus the reference value.", ""]
    for path, table in summary.get("gate", {}).items():
        lines += [f"## Gate, {path} (route accuracy, 192 validation episodes)", "",
                  "| Policy | Group | Memory | Seeds | Per seed | Pooled | Reference | Diff | Verdict |", "|---|---|---|---|---|---|---|---|---|"]
        for policy, e in sorted(table.items(), key=lambda kv: -(kv[1].get("route_accuracy") or -1)):
            if "route_accuracy" not in e:
                lines.append(f"| {policy} | {e['group']} | {e['memory']} | 0/{len(e['seeds_expected'])} | | | | | {e['verdict']} |")
                continue
            per = " / ".join(f"{100 * v:.2f}" for v in e["route_accuracy_by_seed"].values())
            ref = f"{100 * e['reference']['route_accuracy']:.2f}%" if "reference" in e else ""
            diff = f"{100 * e['difference']['route_accuracy']:+.2f}" if "difference" in e else ""
            lines.append(f"| {policy} | {e['group']} | {e['memory']} | {len(e['seeds_present'])}/{len(e['seeds_expected'])} | {per} "
                         f"| {100 * e['route_accuracy']:.2f}% | {ref} | {diff} | {e['verdict']} |")
        lines.append("")
    for path, block in summary.get("synergy", {}).items():
        for reader, table in block["readers"].items():
            lines += [f"## Synergy, {path}, {reader} reader (S task: needed card deleted, regret)", "",
                      "| Policy | Group | Seeds | S lost | S regret | Reference lost | Reference regret | Verdict |", "|---|---|---|---|---|---|---|---|"]
            for policy, e in sorted(table.items(), key=lambda kv: (kv[1].get("metrics", {}).get("S", {}).get("regret", 9))):
                if "metrics" not in e:
                    lines.append(f"| {policy} | {e['group']} | 0/{len(e['seeds_expected'])} | | | | | {e['verdict']} |")
                    continue
                s = e["metrics"]["S"]
                lost = "n/a" if s["needed_evidence_lost"] is None else f"{100 * s['needed_evidence_lost']:.2f}%"
                ref = e.get("reference", {}).get("S")
                rl = "" if not ref else ("n/a" if ref["needed_evidence_lost"] is None else f"{100 * ref['needed_evidence_lost']:.2f}%")
                rr = "" if not ref else f"{ref['regret']:+.4f}"
                lines.append(f"| {policy} | {e['group']} | {len(e['seeds_present'])}/{len(e['seeds_expected'])} | {lost} | {s['regret']:+.4f} "
                             f"| {rl} | {rr} | {e['verdict']} |")
            lines.append("")
    m = summary.get("motion_v2", {}).get("replay")
    if m and m.get("complete"):
        lines += ["## Motion v2, replay from tracked loss arrays (position MSE at +3, ordinary interval)", "",
                  "| Method | Position MSE +3 | Both cue frames kept |", "|---|---|---|"]
        for method, v in sorted(m["online"]["ordinary"].items(), key=lambda kv: kv[1]["position_mse"]["3"]):
            lines.append(f"| {method} | {v['position_mse']['3']:.4f} | {100 * v['success_rate']:.1f}% |")
        lines += ["", f"{m['published_cells_compared']} published cells compared, largest difference "
                      f"{m['max_abs_difference_from_published']:.1e}: {m['verdict']}.",
                  f"Student arms that kept their initial weights on every seed: {len(m['untrained_arms'])} of {len(m['student_selection'])}.", ""]
    lines += ["## Not reproducible in this run", ""]
    for env, items in summary["status"].items():
        for item in items:
            lines.append(f"- {env} / {item['id']} ({item['path']}): {item['status']}" + (f" - {item['note']}" if item.get("note") else ""))
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=ROOT / "configs/reproduction.yaml")
    ap.add_argument("--reference", type=Path, default=ROOT / "configs/reference_values.json")
    args = ap.parse_args()
    manifest = yaml.safe_load(args.manifest.read_text())
    reference = json.loads(args.reference.read_text())
    env = manifest["environments"]
    summary = {"run_id": args.run.name, "reference_sources": reference["sources"], "gate": {}, "synergy": {}, "motion_v2": {}}
    for folder in sorted(p for p in (args.run / "gate").glob("*") if p.is_dir()):
        summary["gate"][folder.name] = gate(folder, manifest["results"]["gate"], reference["gate"], env["gate"]["seeds"])
    for folder in sorted(p for p in (args.run / "synergy").glob("*") if p.is_dir() and p.name != "validate"):
        summary["synergy"][folder.name] = synergy(folder, manifest["results"]["synergy"], reference["synergy"],
                                                  ["multihead", "item_attention"], env["synergy"]["seeds"]["students"])
    if (args.run / "motion_v2/replay").is_dir():
        summary["motion_v2"]["replay"] = motion(args.run / "motion_v2/replay", reference["motion_v2"])
    status = defaultdict(list)
    for name, results in manifest["results"].items():
        for r in results:
            for path in ("replay", "retrain"):
                state = (r.get(path) or {}).get("status")
                if state in ("assets_missing", "not_run", "extracted_only"):
                    status[name].append({"id": r["id"], "path": path, "status": state,
                                         "note": (r.get(path) or {}).get("note") or r.get("note")})
    summary["status"] = status
    out = args.run / "aggregate"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (out / "tables.md").write_text(markdown(summary))
    for name in ("gate", "synergy"):
        for path, table in summary[name].items():
            tables = [table] if name == "gate" else list(table["readers"].values())
            verdicts = defaultdict(int)
            for t in tables:
                for e in t.values():
                    verdicts[e["verdict"]] += 1
            print(f"{name:8s} {path:16s} " + ", ".join(f"{k}: {v}" for k, v in sorted(verdicts.items())))
    if summary["motion_v2"].get("replay"):
        print(f"motion_v2 replay           {summary['motion_v2']['replay']['verdict']}")
    print(f"wrote {out / 'summary.json'} and {out / 'tables.md'}")


if __name__ == "__main__":
    main()
