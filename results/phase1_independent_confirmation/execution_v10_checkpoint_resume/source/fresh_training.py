from budget_guard import CAP
"""Preparation only: fresh reference/candidate kernel, no implicit submission.

The complete source-locked stage orchestrator, actual GPU ledger and full
remaining-sequence guard must exist before this kernel may be invoked.
"""
import copy
import time
from pathlib import Path

import numpy as np
import torch

from phase1.centered_transport import CenteredTransportPredictor
from phase1.exact_runtime_cache import cached_validation_loss
from phase1.model import DynamicsPredictor, fit_basis
from phase1.proposal_trial import cpu_tree, finite_tree, verify_completed_model
from phase1.repair_trial import (train_epoch, learning_rate_factor,
                                  actual_allocation_start)
from phase1.resource_guard import elapsed_from_allocation
from phase1.training import TRAIN_CONDITIONS, paired_rows
from phase1.utils import (seed_all, read_json, write_json, write_csv, file_hash,
                          digest, rng_state, save_torch)
from staged_data import verify_permit, StageCachedFeatureStore


def make_task_store(c, task):
    # Metadata-only runtime view: no legacy arm is set, so Motion timestamps and
    # Gate rank weight retain the exact canonical original definitions.
    runtime = copy.deepcopy(c)
    runtime["experiment"] = {"tasks": [task]}
    store = StageCachedFeatureStore(runtime)
    basis = fit_basis(c, store, task)
    store.bases = {task: basis}
    rows = [r for r in store.manifest["records"]
            if r["task"] == task and r["split"] in ("train", "validation")]
    train = paired_rows([r for r in rows if r["split"] == "train"])
    validation = paired_rows([r for r in rows if r["split"] == "validation"])
    if (len(train), len(validation)) != (512, 96):
        raise ValueError("Full original task sample counts required")
    for rec in rows:
        store.features(rec)
        store.projected_features(rec)
    store.prepare_validation(validation, "cuda")
    return store, basis, train, validation


def run_one_model(c, task, seed, role, root, store, basis, train, validation,
                  permit, sequence_guard, stop, deadline, reference_path=None):
    verify_permit(c, permit)
    allocation_start, native = actual_allocation_start()
    if (task not in ("gate", "motion") or seed not in (17, 29, 43)
            or role not in ("reference", "candidate")
            or c["training"] != {"seeds": [17, 29, 43], "epochs": 120,
                "batch_size": 64, "lr": .001, "weight_decay": .0001, "clip": 1.,
                "lr_step": 40, "lr_gamma": .5, "pair_weight": 2., "rank_weight": .5}):
        raise ValueError("Frozen original training contract changed")
    if (sequence_guard.status != "NATIVE_RECONCILED_COMPLETE_CONFIRMATION_GUARD"
            or sequence_guard.cap_seconds != CAP):
        raise ValueError("Complete actual authorized cumulative sequence guard required")
    sequence_guard.check(allocation_start, task, seed, role, retained_epochs=0)
    root = Path(root)
    resume_existing = root.exists()
    if resume_existing and (task, seed, role) != ('gate', 17, 'reference'):
        raise ValueError("Refuse implicit model retry/overwrite")
    seed_all(seed, 4)
    model = DynamicsPredictor(len(basis["input_mean"]), c["model"], (3, 6), basis)
    reference_sha, frozen = None, None
    if role == "candidate":
        if reference_path is None:
            raise ValueError("Fresh same-seed completed reference required")
        reference_sha = file_hash(reference_path)
        ck = torch.load(reference_path, map_location="cpu", weights_only=False)
        if (ck["signature"]["role"] != "reference" or ck["signature"]["task"] != task
                or ck["signature"]["seed"] != seed
                or ck["signature"]["permit_sha256"] != file_hash(permit)):
            raise ValueError("Reference is not this fresh full confirmation")
        # Caller must lock and independently verify all fresh references before
        # invoking any candidate. Never import a development checkpoint here.
        sequence_guard.require_completed_reference(reference_path, ck["signature"])
        model.load_state_dict(ck["model"], strict=True)
        frozen = cpu_tree(model.state_dict())
        model = CenteredTransportPredictor(model, basis)
    model = model.cuda()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                           lr=.0003 if role == "candidate" else .001,
                           weight_decay=.0001)
    scheduler = (torch.optim.lr_scheduler.LambdaLR(opt, learning_rate_factor)
                 if role == "candidate" else torch.optim.lr_scheduler.StepLR(opt, 40, .5))
    sig = {"role": role, "task": task, "seed": seed,
           "permit_sha256": file_hash(permit), "canonical_config_digest": digest(c),
           "cache_signature": store.ix["signature"],
           "normalization_sha256": file_hash(Path(c["output"])/"features/normalization.npz"),
           "basis_sha256": file_hash(Path(c["output"])/"predictor"/task/"basis.npz"),
           "fresh_reference_best_sha256": reference_sha,
           "selection": "equal_four_condition_two_horizon_mse_earliest_minimum"}
    root.mkdir(parents=True, exist_ok=resume_existing)
    logs, best, step, seconds = [], float("inf"), 0, 0.
    def snapshot(epoch):
        return {"signature": sig, "model": cpu_tree(model.state_dict()),
                "optimizer": cpu_tree(opt.state_dict()), "scheduler": scheduler.state_dict(),
                "rng": rng_state(), "epoch": epoch, "step": step, "best": best,
                "logs": copy.deepcopy(logs), "seconds": seconds, "seed": seed, "task": task}
    first_epoch = 0
    if resume_existing:
        from resume_checkpoint import load_allowed
        retained = load_allowed(root, task, seed, role, sig, model, opt, scheduler)
        logs, best, step, seconds = retained['logs'], retained['best'], retained['step'], retained['seconds']
        first_epoch = retained['epoch']
    else:
        save_torch(root/"latest.pt", snapshot(0))
    for epoch in range(first_epoch, 120):
        epoch_wall_start=time.perf_counter()
        boundary = snapshot(epoch)
        began = time.perf_counter()
        result = train_epoch(c, store, model, opt, train, seed, epoch, "cuda",
                             task if role == "candidate" else "reference_no_anchor",
                             stop, deadline)
        if result is None:
            save_torch(root/"latest.pt", boundary)
            write_json(root/"partial_outcome.json", {"status": "TRUE_PARTIAL_STOP",
                       "retained_epochs": epoch, "phase1_go": False})
            return None
        val = cached_validation_loss(c, store, model, validation, "cuda", task)
        score = float(np.mean([val[k]["mse"] for k in TRAIN_CONDITIONS]))
        elapsed = time.perf_counter()-began
        scheduler.step()
        step += result["steps"]
        seconds += elapsed
        row = {"epoch": epoch+1, "step": step, "validation_mse": score,
               "epoch_seconds": elapsed, "seconds": seconds, "mix_counts": result["counts"],
               "lr": opt.param_groups[0]["lr"],
               **{f"train_{k}_{name}": v[i] for k, v in result["means"].items()
                  for i, name in enumerate(("loss", "mse", "pair_mse", "anchor_mse"))},
               **{f"val_{k}_{name}": v for k, values in val.items() for name, v in values.items()}}
        logs.append(row)
        improved = score < best
        best = min(best, score)
        ck = snapshot(epoch+1)
        if not finite_tree(row) or not finite_tree(ck["model"]) or not finite_tree(ck["optimizer"]):
            raise RuntimeError("Nonfinite actual confirmation training")
        save_torch(root/"latest.pt", ck)
        if improved:
            save_torch(root/"best.pt", ck)
        write_json(root/"training.json", {"signature": sig, "logs": logs, "seed": seed})
        write_csv(root/"training.csv", logs)
        store.verify_boundaries()
        sequence_guard.record_epoch(task, seed, role, epoch+1, time.perf_counter()-epoch_wall_start)
        if (epoch+1) % 20 == 0:
            sequence_guard.check(allocation_start, task, seed, role, retained_epochs=epoch+1)
    proof = verify_completed_model(root, sig, frozen)
    write_json(root/"completion_proof.json", proof)
    return proof
