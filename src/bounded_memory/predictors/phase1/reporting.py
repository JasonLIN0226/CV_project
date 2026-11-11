import csv
import copy
from pathlib import Path

import numpy as np

from .config import SPLITS, CONDITIONS, scientific_hash
from .data import contact_sheet, manifest
from .evaluation import active_tasks
from .stats import clustered_interval, paired_improvement
from .utils import read_json, write_json, file_hash, code_hash


def _raw_records(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    strings = {"episode", "pair", "task", "split", "condition", "sibling"}
    integers = {"seed", "horizon", "query", "gap", "history_count"}
    return [{k: v if k in strings else int(v) if k in integers else float(v) for k, v in row.items()} for row in rows]


def _comparison_config(c):
    value = copy.deepcopy(c)
    for key in ["output", "device", "threads"]:
        value.pop(key, None)
    value.get("experiment", {}).pop("arm", None)
    return value


def _development_artifacts(root, task):
    root = Path(root)
    c = read_json(root/"resolved_config.json")
    metrics = read_json(root/"evaluation/metrics.json")
    experiment = c.get("experiment", {})
    if experiment.get("stage") != "development" or active_tasks(c) != [task]:
        raise ValueError("Each comparison arm must be a single-task development experiment")
    if c["mode"] != "confirm" or c["encoder"]["backend"] != "ijepa" or c["training"]["seeds"] != [17, 29, 43] or c["training"]["epochs"] != 120:
        raise ValueError("Research development comparison requires frozen I-JEPA, all three fixed seeds and 120 epochs")
    if metrics["evaluation_code_hash"] != code_hash("evaluation.py", "stats.py"):
        raise ValueError("Evaluation/statistics provenance differs from the comparison implementation")
    if metrics["raw_sha256"] != file_hash(root/"evaluation/raw_records.csv"):
        raise ValueError("Raw evaluation checksum mismatch")
    proof = metrics.get("training_completion", {})
    expected = {f"{task}/{seed}" for seed in c["training"]["seeds"]}
    if not proof.get("complete") or proof.get("required_epochs") != 120 or set(proof.get("models", {})) != expected:
        raise ValueError("All three 120-epoch models must have completion proofs before comparison")
    for key in expected:
        p, ck = proof["models"][key], metrics["checkpoints"].get(key, {})
        if not p.get("complete") or p.get("completed_epochs") != 120 or p.get("selection_metric") != "validation_mse":
            raise ValueError(f"{key} has incomplete training/selection proof")
        if p["signature"]["scientific_hash"] != scientific_hash(c) or ck.get("best_sha256") != p["best_sha256"] or ck.get("best_epoch") != p["best_epoch"]:
            raise ValueError(f"{key} selected checkpoint provenance mismatch")
        logpath = root/"predictor"/task/f"seed_{p['seed']}"/"training.json"
        if file_hash(logpath) != p["training_json_sha256"]:
            raise ValueError(f"{key} training log checksum mismatch")
        training = read_json(logpath)
        logs = training["logs"]
        if len(logs) != 120 or training["signature"] != p["signature"]:
            raise ValueError(f"{key} actual training log is incomplete or provenance differs")
        if min(logs, key=lambda r: r["validation_mse"])["epoch"] != p["best_epoch"]:
            raise ValueError(f"{key} best epoch is not earliest minimum validation MSE")
    raw = _raw_records(root/"evaluation/raw_records.csv")
    by = {}
    for row in raw:
        key = tuple(row[k] for k in ["seed", "task", "split", "horizon", "condition", "pair", "episode"])
        if key in by or row["task"] != task or not np.isfinite(row["normalized_mse"]):
            raise ValueError("Duplicate, wrong-task or non-finite raw record")
        by[key] = row
    for seed in c["training"]["seeds"]:
        for split in SPLITS[1:]:
            for h in c["data"]["horizons"]:
                for condition in CONDITIONS+["persistence"]:
                    rows = [r for r in raw if (r["seed"], r["split"], r["horizon"], r["condition"]) == (seed, split, h, condition)]
                    pairs = {}
                    for row in rows:
                        pairs.setdefault(row["pair"], []).append(row["episode"])
                    if len(pairs) != c["data"]["pairs"][split] or any(len(set(v)) != 2 or len(v) != 2 for v in pairs.values()):
                        raise ValueError(f"Incomplete raw slice {task}/{split}/h{h}/seed{seed}/{condition}")
    return c, metrics, raw, read_json(root/"dataset/split_audit.json"), read_json(root/"probes/metrics.json")


def development_comparison(reference_root, candidate_root, task, motion_diagnostic=None, output=None):
    """Prespecified, paired candidate adoption gate. SUPPORT is never Phase 1 GO."""
    result = {"task": task, "status": "INCOMPLETE", "reasons": [], "checks": [],
              "scope": "development adoption only; old revealed tests are development evidence, never independent confirmation",
              "reference": str(reference_root), "candidate": str(candidate_root)}
    try:
        if task not in {"gate", "motion"}:
            raise ValueError("Unknown development task")
        ref, rm, rr, ra, rp = _development_artifacts(reference_root, task)
        cand, cm, cr, ca, cp = _development_artifacts(candidate_root, task)
        expected_arm = "gate_no_rank" if task == "gate" else "motion_wide_span"
        if ref["experiment"]["arm"] != "reference" or cand["experiment"]["arm"] != expected_arm:
            raise ValueError("Unexpected reference/candidate arm")
        if _comparison_config(ref) != _comparison_config(cand):
            raise ValueError("A comparison changed a scientific setting beyond its single experiment arm")
        if rm["cache_signature"] != cm["cache_signature"]:
            raise ValueError("Reference and candidate feature cache provenance differs")
        for seed in cand["training"]["seeds"]:
            rs = rm["training_completion"]["models"][f"{task}/{seed}"]["signature"]
            cs = cm["training_completion"]["models"][f"{task}/{seed}"]["signature"]
            for key in ["cache_signature", "normalization_sha256", "basis_sha256", "code_hash"]:
                if rs[key] != cs[key]:
                    raise ValueError(f"Paired arms have different {key}")
        keys = ["seed", "task", "split", "horizon", "condition", "pair", "episode"]
        reference = {tuple(r[k] for k in keys): r for r in rr}
        candidate = {tuple(r[k] for k in keys): r for r in cr}
        if set(reference) != set(candidate):
            raise ValueError("Raw pair/sibling keys differ across arms")
        checks, reasons = result["checks"], result["reasons"]
        if not ra["passed"]:
            reasons.append("Reference data audit failed")
        status, original_reasons, original_checks = decision(cand, ca, cp, cm)
        checks.append({"kind": "candidate_original_task_rules", "passed": status == "DEVELOPMENT_PASS", "status": status,
                       "checks": original_checks, "reasons": original_reasons})
        if status != "DEVELOPMENT_PASS":
            reasons.extend(f"Candidate original task rule: {r}" for r in original_reasons)
        if status == "DEVELOPMENT_INCOMPLETE":
            raise ValueError("Candidate lacks complete original task checks")
        critical = "test_distractor_shift" if task == "gate" else "test_long_gap"
        repeats, rule = cand["evaluation"]["bootstrap"], cand["decision"]
        for split in SPLITS[2:]:
            for horizon in cand["data"]["horizons"]:
                for seed in cand["training"]["seeds"]+["all_seeds_mean"]:
                    selected = [r for r in cr if r["split"] == split and r["horizon"] == horizon and r["condition"] == "privileged_small_budget"
                                and (seed == "all_seeds_mean" or r["seed"] == seed)]
                    grouped = {}
                    for r in selected:
                        key = tuple(r[k] for k in keys)
                        basekey = list(key)
                        basekey[4] = "recent_only"
                        grouped.setdefault((r["pair"], r["episode"]), []).append(
                            [reference[key]["normalized_mse"], r["normalized_mse"], reference[tuple(basekey)]["normalized_mse"]])
                    pairkeys = sorted(grouped)
                    values = np.array([np.mean(grouped[k], axis=0) for k in pairkeys])
                    groups = [k[0] for k in pairkeys]
                    name = f"{task}/{split}/h{horizon}/seed{seed}"
                    if split == critical:
                        absolute = clustered_interval(values[:, 0]-values[:, 1], groups, repeats)
                        ok = absolute["mean"] > 0 and absolute["low"] > 0
                        checks.append({"kind": "critical_absolute_mse_reduction", "slice": name, "passed": ok,
                                       "absolute_mse_reduction": absolute,
                                       "reference_privileged_mse": float(values[:, 0].mean()), "candidate_privileged_mse": float(values[:, 1].mean())})
                        if not ok:
                            reasons.append(name+" Candidate privileged absolute MSE did not reliably improve over matched reference")
                    cross = paired_improvement(values[:, 2], values[:, 1], groups, repeats, rule["baseline_floor"], rule["absolute_latent_gain"])
                    threshold = rule["latent_relative_improvement"] if cross["scale"].startswith("relative") else rule["absolute_latent_gain"]
                    ok = cross["mean"] >= threshold and cross["low"] > 0
                    checks.append({"kind": "candidate_vs_reference_recent", "slice": name, "passed": ok, "improvement": cross,
                                   "reference_recent_mse": float(values[:, 2].mean()), "candidate_privileged_mse": float(values[:, 1].mean())})
                    if not ok:
                        reasons.append(name+" Candidate did not meet original latent-benefit rule against reference recent predictions")
        if task == "motion":
            diagnostic = motion_diagnostic
            if diagnostic is None:
                path = Path(candidate_root)/"probes/motion_precision.json"
                diagnostic = read_json(path) if path.exists() else None
            if diagnostic is None or diagnostic.get("scope") != "validation_only":
                raise ValueError("Motion adoption requires the prespecified train-fit, validation-only continuous-speed diagnostic")
            proof = cm["training_completion"]["models"][f"motion/{cand['training']['seeds'][0]}"]["signature"]
            if diagnostic.get("cache_signature") != cm["cache_signature"] or diagnostic.get("input_basis_sha256") != proof["basis_sha256"] or diagnostic.get("normalization_sha256") != proof["normalization_sha256"]:
                raise ValueError("Motion precision diagnostic cache/basis/normalization provenance mismatch")
            declared = diagnostic.get("comparisons", [])
            required = {"wide_two_vs_current_two", "wide_span_two_vs_single_start", "wide_span_two_vs_single_end"}
            observed = {r["comparison"] for r in declared if r.get("split") == "validation" and r.get("passed") and r["absolute_mse_reduction"]["low"] > 0}
            ok = diagnostic.get("passed") and required.issubset(observed)
            checks.append({"kind": "motion_validation_precision", "passed": bool(ok), "diagnostic": diagnostic})
            if not ok:
                reasons.append("Wide-span two-frame evidence did not beat current two frames and both fixed single-frame controls on validation")
        result["status"] = "REJECTED" if reasons else "SUPPORTED"
        result["candidate_original_status"] = status
        result["preregistered_rules"] = rule
        result["raw_sha256"] = {"reference": rm["raw_sha256"], "candidate": cm["raw_sha256"]}
        result["experiment"] = {"reference": ref["experiment"], "candidate": cand["experiment"]}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result["status"] = "INCOMPLETE"
        result["reasons"].append(str(exc))
    result["reasons"] = list(dict.fromkeys(result["reasons"]))
    if output is not None:
        root = Path(output)
        write_json(root/"development_comparison.json", result)
        from .utils import atomic
        lines = [f"# {task} 单项开发比较", "", f"**{result['status']}** — 仅决定候选是否进入独立确认，不构成 Phase 1 GO。", "",
                 "所有检查使用三种子各 120 轮训练、共同 validation MSE 选模、完整 pair 配对统计；逐种子与三种子平均均保留。", ""]
        lines += ["- "+reason for reason in result["reasons"]] or ["候选满足预定开发采用条件；仍须两个任务均获支持并在全新固定数据上通过完整确认。"]
        lines += ["", "排序损失引起放大、或时间跨度改善速度精度，均为待检验的机制解释；性能比较本身不证明根因。", "",
                  "完整切片、置信区间和原始文件校验见 development_comparison.json。"]
        atomic(root/"development_comparison.md", lambda f: f.write(("\n".join(lines)+"\n").encode()))
    return result


def decision(c, audit, probes, metrics):
    if c["mode"] == "smoke" or c["encoder"]["backend"].startswith("smoke"):
        return "SMOKE_ONLY — NOT A RESEARCH CONCLUSION", ["仅验证软件流程；smoke pixels 不是 JEPA，任何数值均不能触发研究 GO。"], []
    rule = c["decision"]
    tasks = active_tasks(c)
    failures, uncertain, checks = [], [], []
    if not audit["passed"]:
        failures.append("环境没有真正制造可审计的记忆需求。")
    # Completeness is part of the gate: a missing seed/task/OOD/horizon is never a passing result.
    probe_keys = {(p["task"], p["split"], p["variant"], p.get("horizon")) for p in probes["results"]}
    metric_keys = {(r["seed"], r["task"], r["split"], r["condition"], r["horizon"]) for r in metrics["per_seed"]}
    comparison_keys = {(r["seed"], r["task"], r["split"], r["alternative"], r["horizon"]) for r in metrics["comparisons"]}
    missing = []
    for task in tasks:
        for split in SPLITS[2:]:
            for variant in ["gate_cue" if task == "gate" else "motion_two_frame", "recent_only"]:
                if (task, split, variant, None) not in probe_keys:
                    missing.append(f"probe {task}/{split}/{variant}")
            for h in c["data"]["horizons"]:
                if (task, split, "state_decoder_on_true_target", h) not in probe_keys:
                    missing.append(f"readout {task}/{split}/h{h}")
                for seed in c["training"]["seeds"]:
                    for condition in CONDITIONS+["persistence"]:
                        if (seed, task, split, condition, h) not in metric_keys:
                            missing.append(f"metric {seed}/{task}/{split}/{condition}/h{h}")
                    for condition in ["privileged_small_budget", "full_history", "random_small_budget", "wrong_evidence"]:
                        if (seed, task, split, condition, h) not in comparison_keys:
                            missing.append(f"comparison {seed}/{task}/{split}/{condition}/h{h}")
    if missing:
        uncertain.append(f"工程／结果不完整：缺少 {len(missing)} 个必需切片，例如 {missing[0]}。")
    for p in probes["results"]:
        if p["task"] not in tasks or p["split"] not in SPLITS[2:]:
            continue
        if p["variant"] in {"gate_cue", "motion_two_frame"}:
            check = p["accuracy"]["mean"] >= rule["probe_accuracy"] and p["wilson_effective_pairs"][0] > .5
            checks.append({"kind": "evidence_probe", "task": p["task"], "split": p["split"], "passed": check})
            if not check:
                failures.append(f"{p['task']}/{p['split']} 编码器／池化尚未可靠表达证据；对照 mean 与 grid2 必须另建目录并仅按 validation 选择。")
        if p["variant"] == "recent_only" and abs(p["accuracy"]["mean"]-.5) > 1e-7:
            failures.append(f"{p['task']}/{p['split']} recent probe 偏离严格配对 chance，检查输入泄漏。")
        if p["variant"] == "state_decoder_on_true_target":
            readout_ok = p["accuracy"]["mean"] >= rule["readout_accuracy"] if p["task"] == "gate" else np.sqrt(p["position_mse"]["mean"]) <= rule["readout_position_rmse"]
            if not readout_ok:
                failures.append(f"{p['task']}/{p['split']}/h{p['horizon']} readout 不足以支持结论；不能全部归因于预测器。")
    for cmp in metrics["comparisons"]:
        if cmp["task"] not in tasks or cmp["split"] not in SPLITS[2:]:
            continue
        name = f"{cmp['task']}/{cmp['split']}/h{cmp['horizon']}/seed{cmp['seed']}"
        if cmp["alternative"] == "privileged_small_budget":
            latent = cmp["improvement"]["normalized_mse"]
            # Gate: lane error. Motion: continuous position error; paired positions can be on the same side.
            task_metric = "state_accuracy" if cmp["task"] == "gate" else "state_position_mse"
            state = cmp["improvement"][task_metric]
            lt = rule["latent_relative_improvement"] if latent["scale"].startswith("relative") else rule["absolute_latent_gain"]
            st = rule["state_relative_improvement"] if state["scale"].startswith("relative") else rule["absolute_state_gain"] if cmp["task"] == "gate" else rule["absolute_position_mse_gain"]
            ok = latent["mean"] >= lt and latent["low"] > 0 and state["mean"] >= st and state["low"] > 0
            checks.append({"kind": "privileged_benefit", "slice": name, "passed": ok})
            if not ok:
                if latent["high"] < lt or state["high"] < st:
                    failures.append(name+" 记忆收益不足：检查 predictor 可利用性、背景 latent 与任务错误及 OOD 泛化。")
                else:
                    uncertain.append(name+" 区间跨过推进阈值，增加样本／训练种子前不能定论。")
        elif cmp["alternative"] == "wrong_evidence":
            shift = cmp["toward_wrong_future"]
            ok = shift["mean"] >= rule["wrong_evidence_gain"] and shift["low"] > 0
            checks.append({"kind": "content_intervention", "slice": name, "passed": ok})
            if not ok:
                (failures if shift["high"] < rule["wrong_evidence_gain"] else uncertain).append(name+" 内容替换干预未可靠支持历史使用。")
        elif cmp["alternative"] == "full_history":
            value = cmp["improvement"]["normalized_mse"]
            if value["mean"] < -rule["full_history_tolerance"]:
                failures.append(name+" full-history 更差，先检查优化及干扰；可另训独立命名的 sanity model。")
    for row in metrics["per_seed"]:
        if row["task"] not in tasks or row["split"] not in SPLITS[2:]:
            continue
        if row["condition"] == "recent_only" and abs(row["metrics"]["pair_accuracy"]["mean"]-.5) > 1e-7:
            failures.append("recent-only 配对准确率不等于 chance，结构／评估边界异常。")
        if row["almost_indistinguishable_fraction"] > 0.1:
            failures.append(f"{row['task']}/{row['split']} 大量目标 latent pair 几乎不可区分，latent 指标无法支持任务结论。")
    if len(c["training"]["seeds"]) < rule["min_seeds"] or c["mode"] != "confirm":
        uncertain.append("pilot 或训练种子不足：正式 GO 至少需要 confirm 模式及 3 个预先指定的种子。")
    if min(c["data"]["pairs"][s] for s in SPLITS[2:]) < rule["min_pairs"]:
        uncertain.append("测试基础 pair 数量低于预设下界；样本量不足。")
    if not checks:
        uncertain.append("缺少确认指标。")
    experiment = c.get("experiment", {})
    if experiment:
        complete = metrics.get("training_completion", {})
        expected = {f"{task}/{seed}" for task in tasks for seed in c["training"]["seeds"]}
        proven = {key for key, value in complete.get("models", {}).items() if value.get("complete")}
        if not complete.get("complete") or proven != expected:
            uncertain.append("工程／结果不完整：缺少所有预定模型完成训练的 checkpoint 证明。")
        for key in proven:
            proof = complete["models"][key]
            if proof.get("completed_epochs") != c["training"]["epochs"] or proof.get("selection_metric") != "validation_mse" or proof.get("signature", {}).get("scientific_hash") != scientific_hash(c):
                uncertain.append(f"工程／结果不完整：{key} 的完整轮次、共同选模规则或科学方案 provenance 不匹配。")
        if experiment.get("stage") == "confirmation" and tasks != c["data"]["tasks"]:
            uncertain.append("工程／结果不完整：独立第一阶段确认必须包含两个预定任务。")
        if experiment.get("stage") not in {"development", "confirmation"}:
            uncertain.append("工程／结果不完整：实验阶段未声明为 development 或 confirmation。")
        for task in tasks:
            for split in SPLITS[2:]:
                for h in c["data"]["horizons"]:
                    for condition in ["privileged_small_budget", "full_history", "random_small_budget", "wrong_evidence"]:
                        if ("all_seeds_mean", task, split, condition, h) not in comparison_keys:
                            uncertain.append(f"工程／结果不完整：缺少 {task}/{split}/{condition}/h{h} 三种子平均。")
    status = "REVISE" if failures else "INCONCLUSIVE" if uncertain else "GO"
    if experiment.get("stage") == "development":
        status = "DEVELOPMENT_REVISE" if failures else "DEVELOPMENT_INCOMPLETE" if uncertain else "DEVELOPMENT_PASS"
    return status, list(dict.fromkeys(failures+uncertain)), checks


def figures(c, probes, metrics, raw):
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(Path(c["output"])/".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(c["output"])/"figures"
    root.mkdir(exist_ok=True)
    label = "SMOKE ONLY - NOT JEPA" if c["mode"] == "smoke" else "Frozen pretrained I-JEPA diagnostics"
    def save(fig, name):
        fig.suptitle(label, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, .95))
        fig.savefig(root/name, dpi=140)
        plt.close(fig)
    evidence = [p for p in probes["results"] if p["variant"] != "state_decoder_on_true_target" and p["split"] == "test_id"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(evidence)), [p["accuracy"]["mean"] for p in evidence])
    ax.set_xticks(range(len(evidence)), [p["task"]+"\n"+p["variant"] for p in evidence], rotation=15)
    ax.axhline(.5, color="gray", linestyle="--", label="chance")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Linear probe accuracy")
    ax.legend()
    save(fig, "encoder_probe.png")
    # Explicitly facet task + horizon, avoiding an average that hides a failing task/horizon.
    hs = c["data"]["horizons"]
    for name, xfield, yfield, splitfilter in [
        ("prediction_error.png", "condition", "normalized_mse", "test_id"),
        ("delay_benefit.png", "gap", "normalized_mse", None),
        ("id_ood.png", "split", "state_accuracy", None),
        ("memory_quality.png", "history_feature_bytes", "state_accuracy", "test_id"),
        ("wrong_evidence.png", "condition", "pair_margin", "test_id")]:
        fig, axes = plt.subplots(len(active_tasks(c)), len(hs), figsize=(6*len(hs), 3.5*len(active_tasks(c))), squeeze=False)
        for ti, task in enumerate(active_tasks(c)):
            for hi, h in enumerate(hs):
                ax = axes[ti, hi]
                rows = [r for r in raw if r["task"] == task and r["horizon"] == h and (splitfilter is None or r["split"] == splitfilter)]
                conds = CONDITIONS[:4] if name != "wrong_evidence.png" else ["recent_only", "privileged_small_budget", "wrong_evidence"]
                for cond in conds:
                    part = [r for r in rows if r["condition"] == cond]
                    keys = sorted({r[xfield] for r in part})
                    means = [np.mean([r[yfield] for r in part if r[xfield] == k]) for k in keys]
                    if xfield == "condition":
                        ax.bar(cond, means[0], label=cond)
                        ax.tick_params(axis="x", labelrotation=20)
                    elif xfield == "history_feature_bytes":
                        ax.scatter(keys, means, label=cond, s=30)
                    else:
                        ax.plot(keys, means, marker="o", markersize=3, label=cond)
                ax.set_title(f"{task}, horizon={h}")
                ax.set_xlabel(xfield)
                ax.set_ylabel(yfield+" (seed means)")
                if xfield != "condition":
                    ax.legend(fontsize=7)
        save(fig, name)
    fig, ax = plt.subplots(figsize=(8, 4))
    for key, ck in metrics["checkpoints"].items():
        log = read_json(Path(c["output"])/"predictor"/ck["task"]/f"seed_{ck['seed']}"/"training.json")["logs"]
        ax.plot([r["epoch"] for r in log], [r["train_mse"] for r in log], label=f"{key} train")
        ax.plot([r["epoch"] for r in log], [r["validation_mse"] for r in log], linestyle="--", label=f"{key} validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("normalized latent MSE")
    ax.legend()
    save(fig, "training_curves.png")
    # Success/failure ranked by paired future classification then MSE, REAL RGB only.
    m = manifest(c)
    for task in active_tasks(c):
        candidates = [r for r in raw if r["task"] == task and r["split"] == "test_id" and r["condition"] == "privileged_small_budget" and r["seed"] == c["training"]["seeds"][0] and r["horizon"] == hs[-1]]
        candidates.sort(key=lambda r: (-r["pair_accuracy"], r["normalized_mse"]))
        for name, r in [("best", candidates[0]), ("worst", candidates[-1])]:
            pair = [rec for rec in m["records"] if rec["pair"] == r["pair"]]
            if "experiment" in c:
                from .experiment import selected_contact_sheet
                selected_contact_sheet(c, pair, f"example_{task}_{name}_TRUE_RGB.png")
            else:
                contact_sheet(c, pair, f"example_{task}_{name}_TRUE_RGB.png")


def report(c):
    root = Path(c["output"])
    audit = read_json(root/"dataset/split_audit.json")
    probes = read_json(root/"probes/metrics.json")
    metrics = read_json(root/"evaluation/metrics.json")
    if metrics["evaluation_code_hash"] != code_hash("evaluation.py", "stats.py"):
        raise ValueError("Evaluation/statistics code changed; reevaluate before reporting")
    if file_hash(root/"evaluation/raw_records.csv") != metrics["raw_sha256"]:
        raise ValueError("Evaluation raw checksum mismatch")
    with open(root/"evaluation/raw_records.csv", newline="") as f:
        raw = list(csv.DictReader(f))
    strings = {"episode", "pair", "task", "split", "condition"}
    integers = {"seed", "horizon", "query", "gap", "history_count"}
    raw = [{k: v if k in strings else int(v) if k in integers else float(v) for k, v in r.items()} for r in raw]
    status, reasons, checks = decision(c, audit, probes, metrics)
    figures(c, probes, metrics, raw)
    write_json(root/"decision.json", {"status": status, "reasons": reasons, "checks": checks,
               "preregistered_rules": c["decision"], "scientific_hash": scientific_hash(c), "mode": c["mode"]})
    pooled = [r for r in metrics["comparisons"] if r["seed"] == "all_seeds_mean" and r["split"] in SPLITS[2:]]
    experiment = c.get("experiment", {})
    title = "开发比较：单项修改结果（不构成 Phase 1 GO）" if experiment.get("stage") == "development" else "第一阶段实验结果"
    lines = [f"# {title}", "", f"**{status}** · 冻结 I-JEPA · 种子 {c['training']['seeds']}", "",
             "同一个预测器比较四种历史输入。所有超参数和推进阈值在训练前固定，checkpoint只由validation选择。", "",
             "## 四条件预测", "",
             "表中为预先指定seed的平均normalized MSE；配对准确率表示预测更接近本条轨迹的未来还是sibling未来（chance=50%）。", "",
             "|任务|测试|h|近期|完整历史|关键B=2|随机B=2|关键历史配对准确率|", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for task in active_tasks(c):
        for split in SPLITS[2:]:
            for h in c["data"]["horizons"]:
                part = [r for r in metrics["per_seed"] if r["task"] == task and r["split"] == split and r["horizon"] == h]
                means = [np.mean([r["metrics"]["normalized_mse"]["mean"] for r in part if r["condition"] == cond]) for cond in CONDITIONS[:4]]
                accuracy = np.mean([r["metrics"]["pair_accuracy"]["mean"] for r in part if r["condition"] == "privileged_small_budget"])
                lines.append(f"|{task}|{split}|{h}|"+"|".join(f"{v:.5f}" for v in means)+f"|{accuracy:.1%}|")
    lines += ["", "## 关键历史相对近期观测的收益", "",
              "百分比为误差下降，括号为95%配对置信区间；gate任务指标是lane错误率，motion是位置MSE。区间按完整pair重采样，条件于本次预先指定的seed。", "",
              "|任务|测试|h|latent误差下降|任务误差下降|错误证据方向|", "|---|---|---:|---|---|---|"]
    for r in pooled:
        if r["alternative"] != "privileged_small_budget":
            continue
        name = "state_accuracy" if r["task"] == "gate" else "state_position_mse"
        values = [r["improvement"][k] for k in ["normalized_mse", name]]
        text = [f"{v['mean']:.1%} [{v['low']:.1%}, {v['high']:.1%}]" if v["scale"].startswith("relative") else
                f"{v['mean']:.5g} [{v['low']:.5g}, {v['high']:.5g}] absolute" for v in values]
        w = next(x for x in pooled if x["alternative"] == "wrong_evidence" and (x["task"],x["split"],x["horizon"]) == (r["task"],r["split"],r["horizon"]))["toward_wrong_future"]
        lines.append(f"|{r['task']}|{r['split']}|{r['horizon']}|"+"|".join(text)+f"|{w['mean']:.3f} [{w['low']:.3f}, {w['high']:.3f}]|")
    lines += ["", "错误证据方向应为正：保持时间/动作/近期输入不变，只换关键历史内容，预测应向sibling的错误未来移动。", "",
              "## 证据与数据检查", "",
              "|任务|测试|历史证据probe|近期probe|", "|---|---|---:|---:|"]
    for task in active_tasks(c):
        for split in SPLITS[2:]:
            variant = "gate_cue" if task == "gate" else "motion_two_frame"
            def accuracy(name):
                return next(p["accuracy"]["mean"] for p in probes["results"] if p["task"] == task and p["split"] == split and p["variant"] == name)
            lines.append(f"|{task}|{split}|{accuracy(variant):.1%}|{accuracy('recent_only'):.1%}|")
    lines += ["", f"数据审计：{'PASS' if audit['passed'] else 'FAIL'}；共{sum(r['pairs'] for r in audit['splits'])} pair / {sum(r['episodes'] for r in audit['splits'])} episode。",
              "近期RGB/动作严格相同，历史/可见未来不同，pair不跨split；feature配对相等审计通过。遮挡、提示、真实未来请结合figures/pair_*.png检查。",
              "motion单帧对照、true-target decoder及逐seed完整指标保存在probes/metrics.json、evaluation/metrics.json。", "",
              "## 训练与开销", "",
              "两个任务分别训练，同一任务的四种条件共用模型。视觉投影仅在train拟合，训练不读取simulator标签。", ""]
    if experiment:
        lines += [f"实验臂：`{experiment['arm']}`；任务：{active_tasks(c)}。所有新实验臂按四条件、全部 horizon 等权 validation normalized MSE 选 checkpoint，平局取较早轮次。",
                  "门控候选仅删除排序项；运动候选仅改变两帧选取。各任务的训练目标与选帧规则以 resolved_config.json 中固定实验臂为准。", ""]
    else:
        lines += ["训练目标是 latent MSE、配对差分 MSE 与配对排序损失；validation 使用同一目标。", ""]
    for key, ck in metrics["checkpoints"].items():
        log = read_json(root/"predictor"/ck["task"]/f"seed_{ck['seed']}"/"training.json")["logs"]
        lines.append(f"- {key}：完成{len(log)}轮，最佳epoch{ck['best_epoch']}，累计训练{log[-1]['seconds']:.1f}秒，best.pt SHA256 `{ck['best_sha256']}`。")
    ix = read_json(root/"features/index.json")
    lines += [f"- 数据生成：{manifest(c)['generation_seconds']:.1f}秒；冻结编码：{ix['seconds']:.1f}秒。",
              "- 累计训练为完整轮次耗时，不含排队、中断等待或未完成轮次；evaluation中checkpoint的training_seconds记录选中模型保存时的累计值。",
              "- 5120维float32每条历史20KiB，B=2为40KiB；本次latency在CPU上测量缓存特征预测器forward，不包含编码或端到端延迟，见evaluation/latency_memory.json。", "",
              "## 阶段结论", ""]
    if reasons:
        for kind in ["证据", "记忆收益", "内容替换", "readout", "full-history", "工程／结果不完整", "pilot 或训练种子不足", "测试基础 pair 数量", "chance", "区间"]:
            selected = [r for r in reasons if kind in r]
            if selected:
                lines.append(f"- {kind}：{len(selected)}项未通过或不确定，完整切片依据见decision.json。")
        if c["mode"] == "smoke":
            lines.append("- SMOKE ONLY：仅检查软件流程，不是JEPA实验结果。")
    else:
        lines.append("- 当前实验臂满足任务内的原检查；仍须独立开发比较支持候选后才能进入完整确认。" if experiment.get("stage") == "development" else "- 所有预先固定的任务、horizon、测试split和seed满足原推进规则，可进入第二阶段。")
    lines += ["", "GO要求证据可读、latent与任务收益达到原10%阈值且区间支持、错误证据方向正确、ID/OOD与三seed成立。REVISE/INCONCLUSIVE如实保留；不代表历史原则上无用，也不证明方法创新。", "",
              "只看本报告即可了解主结果。原始评估在evaluation/raw_records.csv，固定方案在resolved_config.json，逐条件训练曲线在predictor/{task}/seed_*/training.csv。figures只展示真实RGB，没有生成未来画面。"]
    from .utils import atomic
    atomic(root/"report.md", lambda f: f.write(("\n".join(lines)+"\n").encode()))
    return {"status": status, "report": str(root/"report.md")}
