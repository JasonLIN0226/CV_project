import copy
import signal
import time
from itertools import product
from pathlib import Path

import numpy as np
import torch

from .config import scientific_hash
from .inputs import FeatureStore, collate
from .model import DynamicsPredictor, fit_basis
from .utils import seed_all, rng_state, restore_rng, save_torch, write_csv, write_json, device_of, code_hash, file_hash


TRAIN_CONDITIONS = ("recent_only", "full_history", "privileged_small_budget", "random_small_budget")
EXPERIMENT_ARMS = {"reference", "gate_no_rank", "motion_wide_span", "confirm_candidate"}


def experiment_tasks(c):
    """Restrict a development arm without changing the reusable dataset config."""
    if "experiment" not in c:
        return list(c["data"]["tasks"])
    experiment = c["experiment"]
    if isinstance(experiment, dict) and "protocol_version" in experiment:
        if experiment["protocol_version"] != 4:
            raise ValueError("Unknown explicit experiment protocol version")
        from .v4_pipeline import v4_tasks
        return v4_tasks(c)
    if not isinstance(experiment, dict) or experiment.get("arm") not in EXPERIMENT_ARMS:
        raise ValueError("Unknown experiment arm")
    tasks = experiment.get("tasks")
    if not isinstance(tasks, list) or not tasks or len(set(tasks)) != len(tasks) or any(t not in c["data"]["tasks"] for t in tasks):
        raise ValueError("Experiment tasks must be distinct configured tasks")
    if experiment["arm"] == "gate_no_rank" and tasks != ["gate"]:
        raise ValueError("gate_no_rank requires only the gate task")
    if experiment["arm"] == "motion_wide_span" and tasks != ["motion"]:
        raise ValueError("motion_wide_span requires only the motion task")
    if experiment["arm"] == "confirm_candidate" and tasks != c["data"]["tasks"]:
        raise ValueError("confirm_candidate requires both configured tasks")
    if experiment.get("stage") not in {"development", "confirmation"}:
        raise ValueError("Experiment stage must be development or confirmation")
    if (experiment["arm"] == "confirm_candidate") != (experiment["stage"] == "confirmation"):
        raise ValueError("Only confirm_candidate uses the confirmation stage")
    return list(tasks)


def rank_weight(c, task):
    tasks = experiment_tasks(c)
    if task not in tasks:
        raise ValueError("Task is outside the experiment arm")
    if c.get("experiment", {}).get("protocol_version") == 4:
        return c["training"]["rank_weight"]
    if c.get("experiment", {}).get("arm") in {"gate_no_rank", "confirm_candidate"} and task == "gate":
        return 0.0
    return c["training"]["rank_weight"]


def selection_metric(c):
    experiment_tasks(c)
    return "validation_mse" if "experiment" in c else "validation_loss"


def validation_score(c, results):
    # Each condition has all validation siblings and horizons. Their MSE means
    # therefore form the fixed, equally weighted condition x horizon score.
    key = "mse" if "experiment" in c else "loss"
    selection_metric(c)
    score = float(np.mean([results[condition][key] for condition in TRAIN_CONDITIONS]))
    if not np.isfinite(score):
        raise RuntimeError("Non-finite validation selection score")
    return score


def checkpoint_signature(c, store, task):
    implementation = ["training.py", "model.py", "inputs.py"]
    if c.get("experiment", {}).get("protocol_version") == 4:
        implementation += ["common_anchor.py", "v4_pipeline.py", "v4_adoption.py"]
    signature = {"scientific_hash": scientific_hash(c), "cache_signature": store.ix["signature"],
            "code_hash": code_hash(*implementation),
            "normalization_sha256": file_hash(Path(c["output"])/"features/normalization.npz"),
            "task": task, "basis_sha256": file_hash(Path(c["output"])/"predictor"/task/"basis.npz")}
    if "experiment" in c:
        signature.update(experiment=copy.deepcopy(c["experiment"]), checkpoint_selection=selection_metric(c))
    return signature


def trusted_checkpoint(path, device="cpu"):
    # Only locally created, provenance-checked PyTorch checkpoints. NEVER load untrusted external .pt files.
    return torch.load(path, map_location=device, weights_only=False)


def paired_rows(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["pair"], []).append(row)
    pairs = list(groups.values())
    if any(len(p) != 2 or p[0]["sibling"] != p[1]["id"] for p in pairs):
        raise ValueError("Training/validation batches require complete sibling pairs")
    return pairs


def regression_loss(prediction, target, pair_weight, rank_weight=0):
    # Both terms use observed TRAIN targets, never hidden state or test information.
    mse = (prediction-target).square().mean()
    difference = (prediction[::2]-prediction[1::2])-(target[::2]-target[1::2])
    pair_mse = difference.square().mean()
    other = target.reshape(-1,2,*target.shape[1:]).flip(1).reshape_as(target)
    own_error = (prediction-target).square().mean(-1)
    other_error = (prediction-other).square().mean(-1)
    separation = (target-other).square().mean(-1).clamp_min(1e-6)
    rank_loss = torch.nn.functional.softplus(-4*(other_error-own_error)/separation).mean()
    return mse+pair_weight*pair_mse+rank_weight*rank_loss, mse, pair_mse


def validation_loss(c, store, model, pairs, device, task=None):
    model.eval()
    results = {}
    task = pairs[0][0]["task"] if task is None else task
    ranking = rank_weight(c, task)
    with torch.inference_mode():
        for condition in TRAIN_CONDITIONS:
            losses = []
            for start in range(0, len(pairs), c["training"]["batch_size"]//2):
                rows = [r for p in pairs[start:start+c["training"]["batch_size"]//2] for r in p]
                samples = [store.sample(r, condition) for r in rows]
                batch = collate([s[0] for s in samples], device)
                target = torch.from_numpy(np.stack([s[1] for s in samples])).to(device)
                loss, mse, pair_mse = regression_loss(model(batch), target, c["training"]["pair_weight"], ranking)
                losses.append((float(loss), float(mse), float(pair_mse), len(rows)))
            results[condition] = {key: sum(row[i]*row[3] for row in losses)/sum(row[3] for row in losses)
                                  for i, key in enumerate(["loss", "mse", "pair_mse"])}
    return results


def train(c, stop_after_epoch=None, resume=True, *, v4_authorization=None):
    tasks = experiment_tasks(c)
    metric = selection_metric(c)
    v4 = c.get("experiment", {}).get("protocol_version") == 4
    authorization = None
    if v4:
        from .v4_pipeline import verify_gate_authorization
        from .common_anchor import HISTORY_CONDITIONS, TeacherForwardAccounting, common_anchor_loss, recent_teacher_prediction
        authorization = verify_gate_authorization(c, v4_authorization)
    device = device_of(c)
    store = FeatureStore(c)
    if v4 and authorization["research_authorized"] and store.ix["signature"] != authorization["gate_cache_signature"]:
        raise ValueError("Authorized Gate feature cache differs from actual training cache")
    stop = {"requested": False}
    def handler(signum, frame):
        stop["requested"] = True
        print("Termination signal: saving epoch-boundary restart snapshot after current batch.", flush=True)
    old_handlers = {s: signal.getsignal(s) for s in [signal.SIGTERM, signal.SIGINT]}
    for s in old_handlers:
        signal.signal(s, handler)
    try:
        for task, seed in product(tasks, c["training"]["seeds"]):
            ranking = rank_weight(c, task)
            train_rows = [r for r in store.manifest["records"] if r["task"] == task and r["split"] == "train"]
            val_rows = [r for r in store.manifest["records"] if r["task"] == task and r["split"] == "validation"]
            train_pairs, val_pairs = paired_rows(train_rows), paired_rows(val_rows)
            store.max_cached = len(train_rows)+len(val_rows)
            basis = fit_basis(c, store, task)
            if task not in store.bases:
                store.projected.clear()
            store.bases = {task: basis}
            sig = checkpoint_signature(c, store, task)
            seed_all(seed, c["threads"])
            root = Path(c["output"])/"predictor"/task/f"seed_{seed}"
            root.mkdir(parents=True, exist_ok=True)
            model = DynamicsPredictor(len(store.norm["mean"]), c["model"], c["data"]["horizons"], basis).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=c["training"]["lr"], weight_decay=c["training"]["weight_decay"])
            scheduler = torch.optim.lr_scheduler.StepLR(opt, c["training"]["lr_step"], c["training"]["lr_gamma"])
            epoch, step, best, logs, seconds = 0, 0, float("inf"), [], 0.0
            teacher_accounting = TeacherForwardAccounting() if v4 else None
            discarded_teacher = {"calls": 0, "examples": 0}
            if resume and (root/"latest.pt").exists():
                ck = trusted_checkpoint(root/"latest.pt", device)
                if ck["signature"] != sig or ck["seed"] != seed or ck["task"] != task:
                    raise ValueError("Checkpoint/config mismatch")
                if "experiment" in c and (ck.get("experiment") != c["experiment"] or ck.get("checkpoint_selection") != metric):
                    raise ValueError("Checkpoint experiment/selection mismatch")
                model.load_state_dict(ck["model"])
                opt.load_state_dict(ck["optimizer"])
                scheduler.load_state_dict(ck["scheduler"])
                restore_rng(ck["rng"])
                epoch, step, best, logs, seconds = ck["epoch"], ck["step"], ck["best"], ck["logs"], ck["seconds"]
                if v4:
                    if ck.get("v4_training_authorization") != authorization:
                        raise ValueError("V4 checkpoint actual authorization mismatch")
                    counters = ck["extra_teacher_forward_accounting"]
                    teacher_accounting.extra_forward_calls = counters["extra_forward_calls"]
                    teacher_accounting.extra_forward_examples = counters["extra_forward_examples"]
                    discarded_teacher = dict(ck["discarded_teacher_forward_accounting"])
            def snapshot():
                # CPU snapshot avoids keeping a second GPU model; also captures optimizer/scheduler/RNG.
                def cpu_tree(obj):
                    if torch.is_tensor(obj):
                        return obj.detach().cpu().clone()
                    if isinstance(obj, dict):
                        return {k: cpu_tree(v) for k, v in obj.items()}
                    if isinstance(obj, list):
                        return [cpu_tree(v) for v in obj]
                    return copy.deepcopy(obj)
                return {"model": cpu_tree(model.state_dict()), "optimizer": cpu_tree(opt.state_dict()),
                        "scheduler": scheduler.state_dict(), "rng": rng_state(), "epoch": epoch, "step": step,
                        "best": best, "logs": copy.deepcopy(logs), "seconds": seconds, "seed": seed, "task": task,
                        "signature": sig, "model_config": c["model"], "horizons": c["data"]["horizons"],
                        "feature_dim": len(store.norm["mean"]), "resume_granularity": "epoch_boundary",
                        "checkpoint_selection": metric,
                        **({"v4_training_authorization": authorization,
                            "extra_teacher_forward_accounting": dict(vars(teacher_accounting)),
                            "discarded_teacher_forward_accounting": dict(discarded_teacher),
                            "common_anchor_weight": c["experiment"]["common_anchor_weight"]} if v4 else {}),
                        **({"experiment": copy.deepcopy(c["experiment"])} if "experiment" in c else {})}
            if not (root/"latest.pt").exists():
                save_torch(root/"latest.pt", snapshot())
            while epoch < c["training"]["epochs"]:
                start_ck = snapshot()
                began = time.perf_counter()
                model.train()
                rng = np.random.default_rng(seed+100003*epoch)
                order = rng.permutation(len(train_pairs))
                conditions = list(TRAIN_CONDITIONS)
                losses, counts = [], {k: 0 for k in conditions}
                condition_losses = {k: [] for k in conditions}
                anchor_losses = {k: [] for k in conditions}
                for start in range(0, len(order), c["training"]["batch_size"]//2):
                    pairs = [train_pairs[i] for i in order[start:start+c["training"]["batch_size"]//2]]
                    for condition in rng.permutation(conditions):
                        samples = []
                        for pair in pairs:
                            sample_seed = int(rng.integers(2**32))
                            # Identical random selection for siblings, resampled each epoch.
                            samples.extend(store.sample(r, str(condition), np.random.default_rng(sample_seed)) for r in pair)
                        counts[str(condition)] += len(samples)
                        batch = collate([s[0] for s in samples], device)
                        target = torch.from_numpy(np.stack([s[1] for s in samples])).to(device)
                        opt.zero_grad(set_to_none=True)
                        prediction = model(batch)
                        loss, mse, pair_mse = regression_loss(prediction, target, c["training"]["pair_weight"], ranking)
                        anchor = 0.0
                        if v4 and str(condition) in HISTORY_CONDITIONS:
                            # Current parameters, same step/context, no-grad eval
                            # teacher. Reference has the same extra forward but
                            # retains its original supervised loss at weight 0.
                            teacher = recent_teacher_prediction(model, batch, accounting=teacher_accounting)
                            ordered_rows = [row for pair in pairs for row in pair]
                            common = common_anchor_loss(prediction, teacher, ordered_rows, condition=str(condition), task=task)
                            anchor = float(common.detach())
                            if c["experiment"]["common_anchor_weight"] == 1:
                                loss = loss + common
                        if not torch.isfinite(loss):
                            raise RuntimeError("Non-finite predictor loss")
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), c["training"]["clip"], error_if_nonfinite=True)
                        opt.step()
                        losses.append((float(mse.detach()), float(pair_mse.detach()), len(samples)))
                        condition_losses[str(condition)].append(losses[-1])
                        if v4:
                            anchor_losses[str(condition)].append((anchor, len(samples)))
                        step += 1
                        if stop["requested"]:
                            start_ck["interrupted_partial_epoch"] = epoch
                            if v4:
                                boundary = start_ck["extra_teacher_forward_accounting"]
                                discarded_teacher["calls"] += teacher_accounting.extra_forward_calls-boundary["extra_forward_calls"]
                                discarded_teacher["examples"] += teacher_accounting.extra_forward_examples-boundary["extra_forward_examples"]
                                start_ck["extra_teacher_forward_accounting"] = dict(vars(teacher_accounting))
                                start_ck["discarded_teacher_forward_accounting"] = dict(discarded_teacher)
                            save_torch(root/"latest.pt", start_ck)
                            return {"interrupted": True, "restart_epoch": epoch}
                val = validation_loss(c, store, model, val_pairs, device, task)
                val_loss = float(np.mean([v["loss"] for v in val.values()]))
                val_mse = float(np.mean([v["mse"] for v in val.values()]))
                score = validation_score(c, val)
                scheduler.step()
                epoch += 1
                seconds += time.perf_counter()-began
                logs.append({"epoch": epoch, "step": step,
                             "train_mse": sum(a*n for a, _, n in losses)/sum(n for _, _, n in losses),
                             "train_pair_mse": sum(a*n for _, a, n in losses)/sum(n for _, _, n in losses),
                             "validation_mse": val_mse, "validation_loss": val_loss,
                             "selection_score": score, "selection_metric": metric,
                             "lr": opt.param_groups[0]["lr"], "seconds": seconds, "mix_counts": str(counts),
                             **({"teacher_extra_forward_calls": teacher_accounting.extra_forward_calls,
                                 "teacher_extra_forward_examples": teacher_accounting.extra_forward_examples,
                                 "discarded_teacher_forward_calls": discarded_teacher["calls"],
                                 "discarded_teacher_forward_examples": discarded_teacher["examples"],
                                 "common_anchor_weight": c["experiment"]["common_anchor_weight"],
                                 "train_common_anchor_mse": sum(a*n for values in anchor_losses.values() for a, n in values)/sum(counts.values()),
                                 **{f"train_{cond}_common_anchor_mse": sum(a*n for a, n in values)/sum(n for _, n in values)
                                    for cond, values in anchor_losses.items()}} if v4 else {}),
                             **{f"train_{cond}_{key}": sum(r[i]*r[2] for r in values)/sum(r[2] for r in values)
                                for cond, values in condition_losses.items() for i, key in enumerate(["mse", "pair_mse"])},
                             **{f"val_{cond}_{key}": value for cond, values in val.items() for key, value in values.items()}})
                improved = score < best
                best = min(best, score)
                ck = snapshot()
                save_torch(root/"latest.pt", ck)
                if improved:
                    save_torch(root/"best.pt", ck)
                write_csv(root/"training.csv", logs)
                write_json(root/"training.json", {"task": task, "seed": seed, "logs": logs, "signature": sig, "mode": c["mode"],
                           "checkpoint_selection": metric,
                           **({"v4_training_authorization": authorization,
                               "extra_teacher_forward_accounting": dict(vars(teacher_accounting)),
                               "discarded_teacher_forward_accounting": dict(discarded_teacher),
                               "common_anchor_weight": c["experiment"]["common_anchor_weight"]} if v4 else {}),
                           **({"experiment": copy.deepcopy(c["experiment"])} if "experiment" in c else {})})
                print(f"task={task} seed={seed} epoch={epoch} train={logs[-1]['train_mse']:.5f} validation={val_mse:.5f} "
                      f"recent={val['recent_only']['mse']:.5f} privileged={val['privileged_small_budget']['mse']:.5f} "
                      f"pair={val['privileged_small_budget']['pair_mse']:.5f}", flush=True)
                if stop["requested"]:
                    return {"interrupted": True, "restart_epoch": epoch}
                if stop_after_epoch is not None and epoch >= stop_after_epoch:
                    break
    finally:
        for s, old in old_handlers.items():
            signal.signal(s, old)
    return {"interrupted": False}


def load_predictor(c, seed, store, device, task):
    if task not in experiment_tasks(c):
        raise ValueError("Task is outside the experiment arm")
    basis = fit_basis(c, store, task)
    store.bases = {task: basis}
    store.projected.clear()
    ck = trusted_checkpoint(Path(c["output"])/"predictor"/task/f"seed_{seed}"/"best.pt", device)
    if ck["signature"] != checkpoint_signature(c, store, task) or ck["seed"] != seed or ck["task"] != task:
        raise ValueError("Evaluation/checkpoint configuration mismatch")
    if "experiment" in c and (ck.get("experiment") != c["experiment"] or ck.get("checkpoint_selection") != selection_metric(c)):
        raise ValueError("Evaluation/checkpoint experiment/selection mismatch")
    model = DynamicsPredictor(ck["feature_dim"], ck["model_config"], ck["horizons"], basis).to(device)
    model.load_state_dict(ck["model"])
    return model.eval(), ck
