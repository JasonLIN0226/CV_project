"""Gate: how much do our scorer and Compress to Remember (two slots) depend on their initialisation seed?

    python scripts/train_baselines/gate_init_seeds.py --config configs/gate/team_bundle.yaml --seed 0 --device cuda \\
        --out <dir> [--predictors 17 29 43] [--inits 101 202 303] [--methods ours pacc_recurrent]

The frozen predictors (seeds 17, 29, 43) and the validation episodes are the team bundle's, as in the panel.
For each predictor and each extra initialisation seed:

    ours            our deletion-cost scorer, trained on that predictor's own teacher labels with the registered
                    update rule (AdamW 1e-3, weight decay 1e-4, 1,000 steps of 64, soft cross-entropy); only the
                    initial weights and minibatch order follow the extra seed. For seed = predictor seed the fit
                    is checked to equal the registered training.fit_pair, so nothing else differs.
    pacc_recurrent  Compress to Remember held to two persistent slots, the Gate implementation unchanged, with
                    the extra seed as its training seed (resumable from a checkpoint).

Each (method, predictor, init) unit writes <out>/<method>_init<init>_seed<predictor>.json with the usual Gate
metric rows and is skipped when it exists, so an interrupted Burst session continues where it stopped.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bounded_memory import runs  # noqa: E402
from bounded_memory.environments.gate.experiment import Scorer, open_assets  # noqa: E402
from bounded_memory.methods.deletion_cost import retention, training  # noqa: E402
from bounded_memory.methods.deletion_cost.scorer import paired_initialization  # noqa: E402


def fit_with_init(x, ya, yd, init, device):
    """training.fit_pair's update rule with any initialisation/minibatch seed (no seed whitelist)."""
    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    targets = [torch.as_tensor(y, dtype=torch.float32, device=device) for y in (ya, yd)]
    a, d = paired_initialization(init)
    models = [a.to(device), d.to(device)]
    optimizers = [torch.optim.AdamW(m.parameters(), lr=0.001, weight_decay=0.0001) for m in models]
    for batch in training.minibatch_indices(init, len(x)):
        idx = torch.as_tensor(batch, device=device)
        for model, optimizer, target in zip(models, optimizers, targets):
            optimizer.zero_grad(set_to_none=True)
            loss = -(target[idx] * torch.log_softmax(model(x[idx]), dim=-1)).sum(dim=-1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return {"attention_proxy_student": models[0].state_dict(), "deletion_cost_student": models[1].state_dict()}


def max_weight_difference(a, b):
    return max(float((a[k].float() - b[k].float()).abs().max()) for k in a)


def main():
    ap = runs.parser(__doc__)
    ap.add_argument("--predictors", nargs="+", type=int, default=[17, 29, 43])
    ap.add_argument("--inits", nargs="+", type=int, default=[101, 202, 303])
    ap.add_argument("--methods", nargs="+", default=["ours", "pacc_recurrent"], choices=("ours", "pacc_recurrent"))
    args = ap.parse_args()
    config = runs.load_config(args.config)
    units = [(m, p, i) for m in args.methods for p in args.predictors for i in args.inits]
    pending = [(m, p, i) for m, p, i in units
               if not (Path(args.out) / f"{m}_init{i}_seed{p}.json").exists() or args.force]
    if not pending:
        print("every unit is done")
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    assets = open_assets(config, args.device)
    labels = {}
    for method, predictor, init in pending:
        args.seed = predictor
        task = runs.Task(args, f"{method}_init{init}")
        scorer = Scorer(assets, predictor, task)
        if method == "ours":
            if predictor not in labels:
                x, ya, yd, _ = training.teacher_labels(scorer.model, scorer.train, predictor, assets.device)
                registered = training.fit_pair(x, ya, yd, seed=predictor, split="train", device=assets.device)
                mirror = fit_with_init(x, ya, yd, predictor, assets.device)
                labels[predictor] = (x, ya, yd, max_weight_difference(registered["deletion"], mirror["deletion_cost_student"]))
                task.log(f"predictor {predictor}: {len(x)} training states; mirror of the registered fit, max |dw| "
                         f"{labels[predictor][3]:.2e}")
            x, ya, yd, check = labels[predictor]
            task.info["registered_fit_mirror_max_abs_weight_diff"] = check
            models = training.as_models(fit_with_init(x, ya, yd, init, assets.device), assets.device)
            scorer.add(f"deletion_cost_student_init{init}", retention.scorer_selections(models["deletion_cost_student"], scorer.val))
        else:
            from bounded_memory.methods.pacc import gate as pacc
            folder = Path(args.out) / "checkpoints"
            folder.mkdir(parents=True, exist_ok=True)
            compressor, meta = pacc.train(scorer.model, scorer.train, init, assets.device, variant="recurrent",
                                          steps=config["baselines"]["pacc"]["steps"],
                                          checkpoint=folder / f"pacc_recurrent_init{init}_seed{predictor}_{assets.device.type}.pt",
                                          tag=f"{config['assets']['root']}|predictor{predictor}")
            prediction, gap = pacc.predict(scorer.model, compressor, scorer.val, assets.device, variant="recurrent")
            task.info["pacc_recurrent"] = {**meta, "validation_gap_to_full_history": gap}
            scorer.add(f"pacc_recurrent_init{init}", prediction=prediction)
        task.write(**scorer.payload(), predictor_seed=predictor, init_seed=init, method=method)


if __name__ == "__main__":
    main()
