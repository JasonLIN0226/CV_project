"""Train the proposal's synthetic causal Gate reader on one structured arm."""
from __future__ import annotations

import argparse
import copy
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
CONTROL_SOURCE = HERE.parent / "predictor" / "control" / "src"
if str(CONTROL_SOURCE) not in sys.path:
    sys.path.insert(0, str(CONTROL_SOURCE))

from phase1.inputs import FeatureStore, collate  # noqa: E402
from phase1.config import scientific_hash  # noqa: E402
from phase1.model import fit_basis  # noqa: E402
from phase1.training import (  # noqa: E402
    TRAIN_CONDITIONS,
    paired_rows,
    regression_loss,
    validation_loss,
)
from phase1.utils import (  # noqa: E402
    file_hash,
    read_json,
    restore_rng,
    rng_state,
    save_torch,
    seed_all,
    write_csv,
    write_json,
)
from phase1.v5_synthetic_reader import SyntheticCausalReader  # noqa: E402


SELECTION = "equal_four_condition_two_horizon_mse_earliest_minimum"


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def finite_tree(value) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    return True


def train_epoch(c, store, model, optimizer, pairs, seed, epoch, device, stop):
    model.train()
    rng = np.random.default_rng(seed + 100003 * epoch)
    order = rng.permutation(len(pairs))
    counts = {condition: 0 for condition in TRAIN_CONDITIONS}
    totals = {
        condition: np.zeros(3, dtype=np.float64)
        for condition in TRAIN_CONDITIONS
    }
    steps = 0
    pair_batch = c["training"]["batch_size"] // 2
    for start in range(0, len(order), pair_batch):
        selected = [pairs[index] for index in order[start : start + pair_batch]]
        for condition in rng.permutation(TRAIN_CONDITIONS):
            samples = []
            for pair in selected:
                sample_seed = int(rng.integers(2**32))
                samples.extend(
                    store.sample(
                        row,
                        str(condition),
                        np.random.default_rng(sample_seed),
                    )
                    for row in pair
                )
            batch = collate([sample[0] for sample in samples], device)
            target = torch.from_numpy(
                np.stack([sample[1] for sample in samples])
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            loss, mse, paired = regression_loss(
                prediction,
                target,
                c["training"]["pair_weight"],
                c["training"]["rank_weight"],
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite synthetic causal loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                c["training"]["clip"],
                error_if_nonfinite=True,
            )
            optimizer.step()
            count = len(samples)
            counts[str(condition)] += count
            totals[str(condition)] += np.asarray(
                [
                    float(loss.detach()),
                    float(mse.detach()),
                    float(paired.detach()),
                ]
            ) * count
            steps += 1
            if stop["requested"]:
                return None
    if any(count != 2 * len(pairs) for count in counts.values()):
        raise RuntimeError("Incomplete balanced condition coverage")
    return {
        "steps": steps,
        "counts": counts,
        "means": {
            condition: (totals[condition] / counts[condition]).tolist()
            for condition in TRAIN_CONDITIONS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    args = parser.parse_args()
    c = json.loads(args.config.read_text())
    root = args.output.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Candidate root must first be prepared by prepare_candidate.py: {root}"
        )
    if Path(c["output"]).expanduser().resolve() != root:
        raise ValueError("Candidate config output must match --output")
    if not (root / "predictor/gate").is_dir():
        raise FileNotFoundError("Prepared candidate basis directory is missing")
    for seed in c["training"]["seeds"]:
        model_root = root / "predictor/gate" / f"seed_{seed}"
        if model_root.exists():
            raise FileExistsError(f"Refusing overwrite: {model_root}")
    device = torch.device(c["device"])
    if device.type == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one CUDA device is required")
        if "A100" not in torch.cuda.get_device_name(0):
            raise RuntimeError("Course A100 required")
    if c["data"]["tasks"] != ["gate"]:
        raise ValueError("Candidate training is Gate-only")
    seed_all(17, 4)
    store = FeatureStore(c)
    train_rows = [
        row
        for row in store.manifest["records"]
        if row["task"] == "gate" and row["split"] == "train"
    ]
    validation_rows = [
        row
        for row in store.manifest["records"]
        if row["task"] == "gate" and row["split"] == "validation"
    ]
    train_pairs = paired_rows(train_rows)
    validation_pairs = paired_rows(validation_rows)
    if (len(train_pairs), len(validation_pairs)) != (512, 96):
        raise ValueError("Original Gate pair counts changed")
    store.max_cached = len(train_rows) + len(validation_rows)
    basis = fit_basis(c, store, "gate")
    if set(basis["train_ids"].tolist()) != {row["id"] for row in train_rows}:
        raise ValueError("Basis train identities differ")
    store.bases = {"gate": basis}
    signature = {
        "reader": "SyntheticCausalReader",
        "task": "gate",
        "scientific_hash": scientific_hash(c),
        "dataset_signature": store.manifest["signature"],
        "cache_signature": store.ix["signature"],
        "normalization_sha256": file_hash(
            Path(c["output"]) / "features/normalization.npz"
        ),
        "basis_sha256": file_hash(
            Path(c["output"]) / "predictor/gate/basis.npz"
        ),
        "source_lock_sha256": file_hash(args.source_lock),
        "selection": SELECTION,
    }
    write_json(
        root / "run_contract.json",
        {
            "signature": signature,
            "training": c["training"],
            "data": c["data"],
            "experiment": c["experiment"],
        },
    )
    stop = {"requested": False}
    previous = {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGTERM, signal.SIGINT)
    }
    for signum in previous:
        signal.signal(signum, lambda *_: stop.update(requested=True))
    proofs = {}
    try:
        for seed in c["training"]["seeds"]:
            seed_all(seed, 4)
            model = SyntheticCausalReader(basis).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=c["training"]["lr"],
                weight_decay=c["training"]["weight_decay"],
            )
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                c["training"]["lr_step"],
                c["training"]["lr_gamma"],
            )
            model_root = root / "predictor/gate" / f"seed_{seed}"
            model_root.mkdir(parents=True)
            epoch = 0
            step = 0
            best = float("inf")
            logs = []
            elapsed_sum = 0.0

            def snapshot():
                return {
                    "model": cpu_tree(model.state_dict()),
                    "optimizer": cpu_tree(optimizer.state_dict()),
                    "scheduler": scheduler.state_dict(),
                    "rng": rng_state(),
                    "epoch": epoch,
                    "step": step,
                    "best": best,
                    "logs": copy.deepcopy(logs),
                    "seconds": elapsed_sum,
                    "seed": seed,
                    "task": "gate",
                    "signature": signature,
                    "resume_granularity": "epoch_boundary",
                    "checkpoint_selection": SELECTION,
                    "horizons": [3, 6],
                    "feature_dim": len(store.norm["mean"]),
                }

            save_torch(model_root / "latest.pt", snapshot())
            while epoch < c["training"]["epochs"]:
                began = time.perf_counter()
                result = train_epoch(
                    c,
                    store,
                    model,
                    optimizer,
                    train_pairs,
                    seed,
                    epoch,
                    device,
                    stop,
                )
                if result is None:
                    save_torch(model_root / "latest.pt", snapshot())
                    write_json(
                        root / "outcome.json",
                        {
                            "status": "CANDIDATE_INTERRUPTED",
                            "seed": seed,
                            "completed_epochs": epoch,
                        },
                    )
                    return
                validation = validation_loss(
                    c,
                    store,
                    model,
                    validation_pairs,
                    device,
                    "gate",
                )
                score = float(
                    np.mean(
                        [
                            validation[condition]["mse"]
                            for condition in TRAIN_CONDITIONS
                        ]
                    )
                )
                if not np.isfinite(score):
                    raise RuntimeError("Non-finite validation score")
                epoch_seconds = time.perf_counter() - began
                scheduler.step()
                epoch += 1
                step += result["steps"]
                elapsed_sum += epoch_seconds
                logs.append(
                    {
                        "epoch": epoch,
                        "step": step,
                        "validation_mse": score,
                        "epoch_seconds": epoch_seconds,
                        "seconds": elapsed_sum,
                        "mix_counts": str(result["counts"]),
                        "lr": optimizer.param_groups[0]["lr"],
                        **{
                            f"train_{condition}_{name}": value[index]
                            for condition, value in result["means"].items()
                            for index, name in enumerate(
                                ("loss", "mse", "pair_mse")
                            )
                        },
                        **{
                            f"val_{condition}_{name}": metric
                            for condition, values in validation.items()
                            for name, metric in values.items()
                        },
                    }
                )
                improved = score < best
                best = min(best, score)
                checkpoint = snapshot()
                if not finite_tree(checkpoint["model"]) or not finite_tree(
                    checkpoint["optimizer"]
                ):
                    raise RuntimeError("Non-finite candidate checkpoint")
                save_torch(model_root / "latest.pt", checkpoint)
                if improved:
                    save_torch(model_root / "best.pt", checkpoint)
                write_csv(model_root / "training.csv", logs)
                write_json(
                    model_root / "training.json",
                    {
                        "signature": signature,
                        "seed": seed,
                        "task": "gate",
                        "logs": logs,
                    },
                )
                write_json(
                    root / "progress.json",
                    {
                        "seed": seed,
                        "epoch": epoch,
                        "step": step,
                        "validation_mse": score,
                        "models_complete": len(proofs),
                    },
                )
                print(
                    f"CANDIDATE_GATE seed={seed} epoch={epoch}/120 "
                    f"step={step}/7680 validation={score:.8f}",
                    flush=True,
                )
                if stop["requested"]:
                    return
            latest = torch.load(
                model_root / "latest.pt",
                map_location="cpu",
                weights_only=False,
            )
            best_ck = torch.load(
                model_root / "best.pt",
                map_location="cpu",
                weights_only=False,
            )
            if latest["epoch"] != 120 or latest["step"] != 7680:
                raise ValueError("Candidate did not finish all prescribed epochs")
            scores = [row["validation_mse"] for row in latest["logs"]]
            if best_ck["epoch"] != int(np.argmin(scores)) + 1:
                raise ValueError("Candidate best checkpoint is not earliest minimum")
            proofs[f"gate/{seed}"] = {
                "complete": True,
                "seed": seed,
                "task": "gate",
                "completed_epochs": 120,
                "final_step": 7680,
                "best_epoch": best_ck["epoch"],
                "selection_metric": "validation_mse",
                "signature": {
                    "scientific_hash": signature["scientific_hash"],
                    "cache_signature": signature["cache_signature"],
                    "basis_sha256": signature["basis_sha256"],
                    "normalization_sha256": signature["normalization_sha256"],
                },
                "best_sha256": file_hash(model_root / "best.pt"),
                "latest_sha256": file_hash(model_root / "latest.pt"),
            }
            del model, optimizer
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        for signum, old in previous.items():
            signal.signal(signum, old)
    write_json(
        root / "all_models_completion.json",
        {
            "complete": True,
            "models": proofs,
            "phase1_go": False,
            "held_out_evaluation_executed": False,
        },
    )
    print((root / "all_models_completion.json").read_text(), end="")


if __name__ == "__main__":
    main()
