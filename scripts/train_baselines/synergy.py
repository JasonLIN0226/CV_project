"""Train and score one literature baseline on Synergy for one reader and seed.

    python scripts/train_baselines/synergy.py --config configs/synergy/release.yaml --seed 17 --device cpu \\
        --out <dir> --reader multihead --method pafu_kv --stage-a <dir>

    pafu_kv       one stored score per card, the lowest is deleted; also scores its teacher
    pacc          Compress to Remember: three cards blended into two learned items, nothing deleted
    framemorrow   the released selector ranks the three cards; also scores its teacher. It never
                  sees the reader, so its choices are the same for both readers.

Trained on the TRAIN split and scored on the locked confirmation records with the frozen reader.
Writes <out>/<method>_<reader>_seed<seed>.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402

from bounded_memory import runs  # noqa: E402
from bounded_memory.environments.synergy import experiment  # noqa: E402


def main():
    ap = runs.parser(__doc__)
    ap.add_argument("--reader", required=True); ap.add_argument("--stage-a", type=Path, required=True)
    ap.add_argument("--method", required=True, choices=("pafu_kv", "pacc", "framemorrow"))
    ap.add_argument("--train-on-confirmation", action="store_true", help="mechanics check only, never a result")
    ap.add_argument("--limit", type=int, help="first N training records only (mechanics checks)")
    args = ap.parse_args()
    task = runs.Task(args, f"{args.method}_{args.reader}")
    if task.done():
        return
    config = runs.load_config(args.config)
    hyper = config["baselines"]
    conf_store = experiment.confirmation_store(config)
    conf = conf_store.records
    if args.train_on_confirmation:
        train_store, train = conf_store, conf
    else:
        train_store = experiment.train_store(config)
        train = train_store.split("train")
    train = train[:args.limit]
    cost = np.load(args.stage_a / f"{args.reader}_confirmation.npz")["eval_cost"]
    model = experiment.reader(config, args.reader, conf_store)
    policies = {}
    if args.method == "pafu_kv":
        from bounded_memory.methods.pafu_kv import synergy as pafu
        drop = pafu.pafu(model, train_store, train, conf_store, conf, args.seed, steps=hyper["pafu_kv"]["steps"])
        policies["pafu_kv"] = experiment.policy(conf, cost, drop=drop, seed=args.seed)
        teacher = pafu.pafu_teacher(model, conf_store, conf, "evaluation").argmin(1)
        policies["pafu_teacher_top2"] = experiment.policy(conf, cost, drop=teacher)
    elif args.method == "pacc":
        from bounded_memory.methods.pacc import synergy as pacc
        loss = pacc.pacc(model, train_store, train, conf_store, conf, args.seed, steps=hyper["pacc"]["steps"])
        policies["pacc"] = experiment.policy(conf, cost, loss=loss, seed=args.seed)
    else:
        from bounded_memory.methods.framemorrow import synergy as fm
        h = hyper["framemorrow"]
        drop = fm.framemorrow(train_store, train, conf_store, conf, args.seed, epochs=h["epochs"],
                              hidden=h["hidden"], layers=h["layers"])
        policies["fm_published"] = experiment.policy(conf, cost, drop=drop, seed=args.seed)
        policies["fm_teacher_top2"] = experiment.policy(conf, cost, drop=fm.framemorrow_teacher(conf_store, conf))
    for name, record in policies.items():
        experiment.describe(task, name, record)
    task.info["train_records"] = len(train)
    task.write(environment="synergy", reader=args.reader, records=len(conf), policies=policies)


if __name__ == "__main__":
    main()
