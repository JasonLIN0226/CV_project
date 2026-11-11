"""Two predeclared task-local ablations; confirmation is gated by actual development evidence."""
import copy
import shutil
from pathlib import Path

import numpy as np

from .config import prepare, scientific_hash
from .data import require_audit
from .cache import load_index, load_normalization
from .utils import code_hash, digest, file_hash, read_json, write_json, seed_all


ARMS = {"gate": ("reference", "gate_no_rank"),
        "motion": ("reference", "motion_wide_span")}


def selected_contact_sheet(c, pair, name):
    """Show ACTUAL privileged timestamps without changing the frozen dataset generator."""
    from PIL import Image, ImageDraw
    from .data import load_episode
    from .inputs import selected_evidence
    panels = []
    for rec in pair:
        obs = load_episode(c, rec)
        raw = load_episode(c, rec, True, keys=("evidence",))["evidence"]
        evidence = selected_evidence(c, rec, raw).tolist()
        q = rec["query"]
        times = evidence+[q-c["data"]["recent"]+1, q]+[q+h for h in c["data"]["horizons"]]
        row = []
        for t in times:
            tile = Image.new("RGB", (160, 187), "white")
            tile.paste(Image.fromarray(obs["rgb"][t]).resize((160, 160)), (0, 27))
            label = "selected evidence" if t in evidence else "recent" if t <= q else "TRUE future"
            ImageDraw.Draw(tile).text((3, 6), f"t={t} {label}", fill="black")
            row.append(tile)
        panels.append(row)
    canvas = Image.new("RGB", (160*len(panels[0]), 187*len(panels)), "white")
    for r, row in enumerate(panels):
        for col, tile in enumerate(row):
            canvas.paste(tile, (160*col, 187*r))
    path = Path(c["output"])/"figures"/name
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _link(source, destination):
    source, destination = Path(source).resolve(), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() != source:
            raise ValueError("Cached artifact link points to another run")
    elif destination.exists():
        raise ValueError("Refusing to replace an existing artifact with a cache link")
    else:
        destination.symlink_to(source, target_is_directory=source.is_dir())


def _copy_verified(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if file_hash(destination) != file_hash(source):
            raise ValueError("Existing reused artifact checksum differs")
    else:
        shutil.copy2(source, destination)


def import_basis(source_config, destination_config, task, index):
    """Reuse EXACT train-fitted arrays; bind a new, explicit experiment provenance envelope."""
    source = Path(source_config["output"])/"predictor"/task/"basis.npz"
    meta = read_json(source.with_suffix(".json"))
    norm_sha = file_hash(Path(source_config["output"])/"features/normalization.npz")
    expected = digest([scientific_hash(source_config), index["signature"], code_hash("model.py"), norm_sha, task])
    if meta["signature"] != expected or meta["sha256"] != file_hash(source) or meta["fit_split"] != "train":
        raise ValueError("Source basis provenance is not the validated train-only basis")
    manifest = require_audit(source_config)
    ids = {r["id"] for r in manifest["records"] if r["task"] == task and r["split"] == "train"}
    with np.load(source, allow_pickle=False) as arrays:
        if set(arrays["train_ids"].tolist()) != ids or any(not np.isfinite(arrays[k]).all()
                for k in ["input_mean", "projection", "target_mean", "output_basis"]):
            raise ValueError("Source basis train IDs or finite arrays are invalid")
    target = Path(destination_config["output"])/"predictor"/task/"basis.npz"
    _copy_verified(source, target)
    signature = digest([scientific_hash(destination_config), index["signature"], code_hash("model.py"), norm_sha, task])
    write_json(target.with_suffix(".json"), {**meta, "signature": signature,
        "reused_from": str(source), "source_signature": meta["signature"],
        "source_scientific_hash": scientific_hash(source_config),
        "reuse_scope": "byte-identical train-only arrays; no refit, no test targets"})


def prepare_shared(source_output, destination):
    """Preserve v2; link large immutable payloads and copy only audited small metadata."""
    source_output, destination = Path(source_output).resolve(), Path(destination).resolve()
    if source_output == destination or source_output in destination.parents:
        raise ValueError("Development artifacts must be outside the original experiment")
    original = read_json(source_output/"resolved_config.json")
    original["output"] = str(source_output)
    if "experiment" in original:
        raise ValueError("Development must start from the original v2 run, not another selected arm")
    manifest, index = require_audit(original), load_index(original)
    load_normalization(original)
    if original["mode"] != "smoke":
        if original["data"]["seed"] != 6301 or original["training"]["seeds"] != [17, 29, 43] or original["training"]["epochs"] != 120:
            raise ValueError("Not the preregistered v2 development source")
    base = copy.deepcopy(original)
    base["output"] = str(destination)
    prepare(base)
    for rel in ["dataset/manifest.json", "dataset/split_audit.json", "features/index.json",
                "features/pair_feature_audit.json", "features/normalization.json", "features/normalization.npz"]:
        _copy_verified(source_output/rel, destination/rel)
    for rel in ["dataset/observations", "dataset/evaluation_only"]:
        _link(source_output/rel, destination/rel)
    for split in original["data"]["pairs"]:
        _link(source_output/"features"/split, destination/"features"/split)
    for task in original["data"]["tasks"]:
        import_basis(original, base, task, index)
    for image in (source_output/"figures").glob("pair_*.png"):
        _copy_verified(image, destination/"figures"/image.name)
    write_json(destination/"reuse_manifest.json", {"source": str(source_output),
        "source_scientific_hash": scientific_hash(original), "dataset_signature": manifest["signature"],
        "cache_signature": index["signature"], "index_sha256": file_hash(source_output/"features/index.json"),
        "normalization_sha256": file_hash(source_output/"features/normalization.npz"),
        "basis_sha256": {t: file_hash(source_output/"predictor"/t/"basis.npz") for t in original["data"]["tasks"]},
        "source_is_read_only": True, "no_reencoding": True})
    return base


def arm_config(base, root, task, arm):
    if task not in ARMS or arm not in ARMS[task]:
        raise ValueError("Only the two preregistered task-local contrasts are allowed")
    c = copy.deepcopy(base)
    c["output"], c["device"] = str(Path(root).resolve()/task/arm), "cpu"
    c["experiment"] = {"stage": "development", "arm": arm, "tasks": [task]}
    prepare(c)
    for folder in ["dataset", "features"]:
        _link(Path(base["output"])/folder, Path(c["output"])/folder)
    import_basis(base, c, task, load_index(base))
    for image in (Path(base["output"])/"figures").glob(f"pair_{task}_*.png"):
        _copy_verified(image, Path(c["output"])/"figures"/image.name)
    if arm == "motion_wide_span":
        manifest = require_audit(c)
        for split in c["data"]["pairs"]:
            pair = [r for r in manifest["records"] if r["task"] == task and r["split"] == split and r["scene_index"] == 0]
            selected_contact_sheet(c, pair, f"selected_{task}_{split}.png")
    return c


def development(c, source_output, task, stop_after_epoch=None):
    from .cli import preflight
    from .probes import probe, motion_precision
    from .training import train
    from .evaluation import evaluate
    from .reporting import report, development_comparison
    root = Path(c["output"])
    if not preflight(c)["passed"]:
        raise RuntimeError("Development CPU preflight failed before science")
    base = prepare_shared(source_output, root/"shared")
    if scientific_hash(c) != scientific_hash(base):
        raise ValueError("Development configuration must match the fixed v2 source; no extra protocol changes")
    base["threads"] = c["threads"]
    diagnostic = None
    if task == "motion":
        seed_all(base["training"]["seeds"][0], base["threads"])
        diagnostic = motion_precision(base, splits=["validation"])
    for arm in ARMS[task]:
        ac = arm_config(base, root, task, arm)
        seed_all(ac["training"]["seeds"][0], ac["threads"])
        if not (Path(ac["output"])/"probes/metrics.json").exists():
            probe(ac)
        result = train(ac, stop_after_epoch)
        if result["interrupted"]:
            return {"interrupted": True, "task": task, "arm": arm}
        if stop_after_epoch is not None and stop_after_epoch < ac["training"]["epochs"]:
            return {"interrupted": False, "stopped_for_resume_test": True}
        evaluate(ac)
        report(ac)
    result = development_comparison(root/task/ARMS[task][0], root/task/ARMS[task][1], task,
        motion_diagnostic=diagnostic, output=root/task)
    write_json(root/task/"comparison.json", result)
    development_report(root)
    return result


def development_report(root, verify=False):
    root = Path(root)
    tasks = {}
    for task in ARMS:
        p = root/task/"comparison.json"
        tasks[task] = read_json(p) if p.exists() else {"status": "INCOMPLETE", "reasons": ["Actual task comparison not complete"]}
        if verify and tasks[task]["status"] == "SUPPORTED":
            from .reporting import development_comparison
            from .evaluation import training_completion
            issues = []
            for arm in ARMS[task]:
                arm_root = root/task/arm
                try:
                    config = read_json(arm_root/"resolved_config.json")
                    current = training_completion(config)
                    recorded = read_json(arm_root/"evaluation/metrics.json")["training_completion"]
                    if not current["complete"] or current != recorded:
                        issues.append(f"{task}/{arm}: current checkpoints do not match evaluated complete models")
                except (OSError, ValueError, KeyError) as exc:
                    issues.append(f"{task}/{arm}: {exc}")
            if issues:
                tasks[task] = {"status": "INCOMPLETE", "reasons": issues}
            else:
                diagnostic = read_json(root/"shared/probes/motion_precision.json") if task == "motion" else None
                tasks[task] = development_comparison(root/task/ARMS[task][0], root/task/ARMS[task][1], task,
                    motion_diagnostic=diagnostic)
    ready = all(v["status"] == "SUPPORTED" for v in tasks.values())
    result = {"stage": "development", "confirmation_ready": ready,
        "status": "READY_FOR_CONFIRMATION" if ready else "DEVELOPMENT_REVISE" if any(v["status"] == "REJECTED" for v in tasks.values()) else "DEVELOPMENT_INCOMPLETE",
        "tasks": tasks, "independent_phase1_result": False,
        "fixed_confirmation": {"data_seed": 9137, "pairs_per_task": {"train": 512, "validation": 96,
            "test_id": 256, "test_long_gap": 256, "test_distractor_shift": 256},
            "training_seeds": [17, 29, 43], "epochs": 120}}
    write_json(root/"decision.json", result)
    lines = ["# 第一阶段开发对照", "", "**"+result["status"]+"** · 开发证据，不是独立 Phase 1 GO。", "",
             "Gate 只删除 ranking；motion 只改变 privileged B=2 时间选择。对照也按共同 validation MSE 重新训练。", ""]
    for task, values in tasks.items():
        lines += [f"- {task}: {values['status']}"]
        lines += ["  - "+str(reason) for reason in values.get("reasons", [])]
    lines += ["", "只有两任务均 SUPPORTED，才允许 seed9137 新轨迹确认；不调阈值、种子或额外模块。",
              "开发结果来自已揭示的 v2 数据，仅支持候选筛选。各任务实际曲线、原始评估与对照见子目录。"]
    (root/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return result


def confirmation_config(c, development_root):
    dev = development_report(development_root, verify=True)
    if not dev["confirmation_ready"]:
        raise ValueError("Both actual development comparisons must be SUPPORTED before confirmation")
    if c["mode"] == "smoke":
        raise ValueError("Smoke development can never authorize scientific confirmation")
    cc = copy.deepcopy(c)
    cc["mode"] = "confirm"
    cc["data"]["seed"] = 9137
    cc["data"]["pairs"] = {"train": 512, "validation": 96, "test_id": 256,
                            "test_long_gap": 256, "test_distractor_shift": 256}
    cc["training"]["seeds"], cc["training"]["epochs"] = [17, 29, 43], 120
    cc["experiment"] = {"stage": "confirmation", "arm": "confirm_candidate", "tasks": ["gate", "motion"]}
    prepare(cc)
    write_json(Path(cc["output"])/"development_authorization.json", {"development_root": str(Path(development_root).resolve()),
        "decision_sha256": file_hash(Path(development_root)/"decision.json"),
        "comparisons": {t: file_hash(Path(development_root)/t/"comparison.json") for t in ARMS},
        "confirmation_data_seed": 9137, "rules_unchanged": True})
    return cc


def confirmation(c, development_root, features_only=False):
    from .cli import preflight
    from .data import generate, validate_data
    from .cache import encode, fit_normalization
    from .probes import probe
    from .training import train
    from .evaluation import evaluate
    from .reporting import report
    cc = confirmation_config(c, development_root)
    if not preflight(cc, check_checkpoint=features_only)["passed"]:
        raise RuntimeError("Effective confirmation preflight failed before science")
    seed_all(cc["training"]["seeds"][0], cc["threads"])
    if features_only:
        generate(cc)
        validate_data(cc)
        encode(cc)
        return {"stage": "confirmation_features", "output": cc["output"]}
    fit_normalization(cc)
    probe(cc)
    result = train(cc)
    if result["interrupted"]:
        return result
    evaluate(cc)
    return report(cc)
