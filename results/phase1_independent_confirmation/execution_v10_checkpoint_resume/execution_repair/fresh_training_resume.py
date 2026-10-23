from budget_guard import CAP
"""Execution-only restart wrapper for the frozen fresh-training kernel.

Completed models are verified and skipped. The interrupted model may resume
only from its exact latest checkpoint. All training mathematics, samples,
optimizer settings, epochs, and completion gates come from the frozen kernel.
"""
import copy
import time
from pathlib import Path

import numpy as np
import torch

from phase1.centered_transport import CenteredTransportPredictor
from phase1.exact_runtime_cache import cached_validation_loss
from phase1.model import DynamicsPredictor
from phase1.proposal_trial import cpu_tree, finite_tree, verify_completed_model
from phase1.repair_trial import train_epoch, learning_rate_factor
from phase1.training import TRAIN_CONDITIONS
from phase1.utils import (
    digest,
    file_hash,
    read_json,
    restore_rng,
    rng_state,
    save_torch,
    seed_all,
    write_csv,
    write_json,
)
from resume_checkpoint import equal
from staged_data import verify_permit

ROOT = Path(__file__).resolve().parent.parent


def _archive(path):
    path = Path(path)
    if not path.exists():
        return None
    target = path.with_name(path.name + ".sigterm9086")
    if target.exists():
        target = path.with_name(path.name + ".sigterm9086." + file_hash(path)[:12])
    path.replace(target)
    return str(target)


def _load_retained(root, signature, model, optimizer, scheduler):
    latest_path = root / "latest.pt"
    training_path = root / "training.json"
    if not latest_path.is_file() or not training_path.is_file():
        raise ValueError("Partial model lacks latest.pt/training.json")
    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    training = read_json(training_path)
    logs = training.get("logs", [])
    if (
        training.get("signature") != signature
        or training.get("seed") != signature["seed"]
        or latest.get("signature") != signature
        or latest.get("logs") != logs
        or not isinstance(latest.get("epoch"), int)
        or latest["epoch"] != len(logs)
        or not 1 <= latest["epoch"] < 120
        or latest.get("step") != latest["epoch"] * 64
        or not finite_tree(latest.get("model"))
        or not finite_tree(latest.get("optimizer"))
    ):
        raise ValueError("Partial checkpoint does not match the frozen run")
    if latest.get("scheduler", {}).get("last_epoch") != latest["epoch"]:
        raise ValueError("Partial scheduler boundary mismatch")
    if any(float(state["step"]) != latest["step"] for state in latest["optimizer"]["state"].values()):
        raise ValueError("Partial optimizer boundary mismatch")
    restore_rng(latest["rng"])
    model.load_state_dict(latest["model"], strict=True)
    optimizer.load_state_dict(latest["optimizer"])
    scheduler.load_state_dict(latest["scheduler"])
    restored = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": rng_state(),
    }
    for key, expected in restored.items():
        if not equal(expected, latest[key]):
            raise ValueError("Execution-only checkpoint restoration is not exact: " + key)
    _archive(root / "partial_outcome.json")
    write_json(
        root / "execution_resume_provenance.json",
        {
            "original_latest_sha256": file_hash(latest_path),
            "original_best_sha256": file_hash(root / "best.pt"),
            "source_lock_sha256": file_hash(ROOT / "source_lock.json"),
            "signature_preserved": True,
            "retained_epochs": latest["epoch"],
            "retained_steps": latest["step"],
            "model_optimizer_scheduler_all_rng_exact": True,
            "no_model_forward_for_restoration": True,
            "next_epoch": latest["epoch"] + 1,
            "no_original_checkpoint_or_epoch_rewritten": True,
            "execution_repair_reason": "Parent OOD session SIGTERM",
        },
    )
    return latest


def _epoch_row(*, epoch, step, score, elapsed, seconds, counts, means, values, lr):
    train = {
        f"train_{condition}_{name}": means[condition][index]
        for condition in means
        for index, name in enumerate(("loss", "mse", "pair_mse", "anchor_mse"))
    }
    validation = {
        f"val_{condition}_{name}": value
        for condition, fields in values.items()
        for name, value in fields.items()
    }
    return {
        "epoch": epoch + 1,
        "step": step,
        "validation_mse": score,
        "epoch_seconds": elapsed,
        "seconds": seconds,
        "mix_counts": counts,
        "lr": lr,
        **train,
        **validation,
    }


def run_one_model(
    c,
    task,
    seed,
    role,
    root,
    store,
    basis,
    train,
    validation,
    permit,
    sequence_guard,
    stop,
    deadline,
    reference_path=None,
):
    verify_permit(c, permit)
    from phase1.repair_trial import actual_allocation_start

    allocation_start, native = actual_allocation_start()
    if (
        task not in ("gate", "motion")
        or seed not in (17, 29, 43)
        or role not in ("reference", "candidate")
        or c["training"]
        != {
            "seeds": [17, 29, 43],
            "epochs": 120,
            "batch_size": 64,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "clip": 1.0,
            "lr_step": 40,
            "lr_gamma": 0.5,
            "pair_weight": 2.0,
            "rank_weight": 0.5,
        }
    ):
        raise ValueError("Frozen original training contract changed")
    if (
        sequence_guard.status != "NATIVE_RECONCILED_COMPLETE_CONFIRMATION_GUARD"
        or sequence_guard.cap_seconds != CAP
    ):
        raise ValueError("Complete actual authorized cumulative sequence guard required")
    retained_epochs = sequence_guard.state.get("epochs", {}).get(
        f"{role}/{task}/{seed}", 0
    )
    sequence_guard.check(
        allocation_start,
        task,
        seed,
        role,
        retained_epochs=retained_epochs,
    )
    root = Path(root)
    seed_all(seed, 4)
    model = DynamicsPredictor(len(basis["input_mean"]), c["model"], (3, 6), basis)
    reference_sha, frozen = None, None
    if role == "candidate":
        if reference_path is None:
            raise ValueError("Fresh same-seed completed reference required")
        reference_sha = file_hash(reference_path)
        reference_checkpoint = torch.load(
            reference_path, map_location="cpu", weights_only=False
        )
        if (
            reference_checkpoint["signature"]["role"] != "reference"
            or reference_checkpoint["signature"]["task"] != task
            or reference_checkpoint["signature"]["seed"] != seed
            or reference_checkpoint["signature"]["permit_sha256"] != file_hash(permit)
        ):
            raise ValueError("Reference is not this fresh full confirmation")
        sequence_guard.require_completed_reference(
            reference_path, reference_checkpoint["signature"]
        )
        model.load_state_dict(reference_checkpoint["model"], strict=True)
        frozen = cpu_tree(model.state_dict())
        model = CenteredTransportPredictor(model, basis)
    signature = {
        "role": role,
        "task": task,
        "seed": seed,
        "permit_sha256": file_hash(permit),
        "canonical_config_digest": digest(c),
        "cache_signature": store.ix["signature"],
        "normalization_sha256": file_hash(
            Path(c["output"]) / "features/normalization.npz"
        ),
        "basis_sha256": file_hash(
            Path(c["output"]) / "predictor" / task / "basis.npz"
        ),
        "fresh_reference_best_sha256": reference_sha,
        "selection": "equal_four_condition_two_horizon_mse_earliest_minimum",
    }

    if root.exists() and (root / "completion_proof.json").is_file():
        proof = verify_completed_model(root, signature, frozen)
        recorded = read_json(root / "completion_proof.json")
        if recorded != proof:
            raise ValueError("Existing completion proof differs from independent replay")
        write_json(
            root / "execution_skip_proof.json",
            {
                "status": "VERIFIED_COMPLETE_REUSED_WITHOUT_MODEL_FORWARD",
                "completion_sha256": file_hash(root / "completion_proof.json"),
                "source_lock_sha256": file_hash(ROOT / "source_lock.json"),
            },
        )
        return proof

    resume_existing = root.exists()
    if resume_existing and not (root / "latest.pt").is_file():
        raise ValueError("Existing model directory has no resumable checkpoint")
    model = model.cuda()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.0003 if role == "candidate" else 0.001,
        weight_decay=0.0001,
    )
    scheduler = (
        torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
        if role == "candidate"
        else torch.optim.lr_scheduler.StepLR(optimizer, 40, 0.5)
    )
    root.mkdir(parents=True, exist_ok=resume_existing)
    logs, best, step, seconds = [], float("inf"), 0, 0.0

    def snapshot(epoch):
        return {
            "signature": signature,
            "model": cpu_tree(model.state_dict()),
            "optimizer": cpu_tree(optimizer.state_dict()),
            "scheduler": scheduler.state_dict(),
            "rng": rng_state(),
            "epoch": epoch,
            "step": step,
            "best": best,
            "logs": copy.deepcopy(logs),
            "seconds": seconds,
            "seed": seed,
            "task": task,
        }

    first_epoch = 0
    if resume_existing:
        retained = _load_retained(root, signature, model, optimizer, scheduler)
        logs = retained["logs"]
        best = retained["best"]
        step = retained["step"]
        seconds = retained["seconds"]
        first_epoch = retained["epoch"]
        if retained_epochs != first_epoch or len(logs) != first_epoch:
            raise ValueError("Checkpoint and budget-guard epoch boundary differ")
    else:
        if retained_epochs:
            raise ValueError("Budget guard records epochs for a model without a checkpoint")
        save_torch(root / "latest.pt", snapshot(0))

    for epoch in range(first_epoch, 120):
        epoch_wall_start = time.perf_counter()
        boundary = snapshot(epoch)
        began = time.perf_counter()
        result = train_epoch(
            c,
            store,
            model,
            optimizer,
            train,
            seed,
            epoch,
            "cuda",
            task if role == "candidate" else "reference_no_anchor",
            stop,
            deadline,
        )
        if result is None:
            save_torch(root / "latest.pt", boundary)
            write_json(
                root / "partial_outcome.json",
                {
                    "status": "TRUE_PARTIAL_STOP",
                    "retained_epochs": epoch,
                    "phase1_go": False,
                },
            )
            return None
        val = cached_validation_loss(c, store, model, validation, "cuda", task)
        score = float(np.mean([val[key]["mse"] for key in TRAIN_CONDITIONS]))
        elapsed = time.perf_counter() - began
        scheduler.step()
        step += result["steps"]
        seconds += elapsed
        row = _epoch_row(
            epoch=epoch,
            step=step,
            score=score,
            elapsed=elapsed,
            seconds=seconds,
            counts=result["counts"],
            means=result["means"],
            values=val,
            lr=optimizer.param_groups[0]["lr"],
        )
        logs.append(row)
        improved = score < best
        best = min(best, score)
        checkpoint = snapshot(epoch + 1)
        if (
            not finite_tree(row)
            or not finite_tree(checkpoint["model"])
            or not finite_tree(checkpoint["optimizer"])
        ):
            raise RuntimeError("Nonfinite actual confirmation training")
        save_torch(root / "latest.pt", checkpoint)
        if improved:
            save_torch(root / "best.pt", checkpoint)
        write_json(
            root / "training.json",
            {"signature": signature, "logs": logs, "seed": seed},
        )
        write_csv(root / "training.csv", logs)
        store.verify_boundaries()
        sequence_guard.record_epoch(
            task,
            seed,
            role,
            epoch + 1,
            time.perf_counter() - epoch_wall_start,
        )
        if (epoch + 1) % 20 == 0:
            sequence_guard.check(
                allocation_start,
                task,
                seed,
                role,
                retained_epochs=epoch + 1,
            )
    proof = verify_completed_model(root, signature, frozen)
    write_json(root / "completion_proof.json", proof)
    return proof
