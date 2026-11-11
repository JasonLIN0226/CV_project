"""The project's six online heuristics on Gate for one predictor seed: FIFO, reservoir, temporal
buckets, visual diversity, prediction surprise and predictor attention.

    python scripts/run_baselines/gate.py --config configs/gate/team_bundle.yaml --seed 17 --device cpu --out <dir>

Writes <out>/heuristics_seed<seed>.json. Nothing is trained.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bounded_memory import runs  # noqa: E402
from bounded_memory.environments.gate.experiment import Scorer, open_assets  # noqa: E402
from bounded_memory.evaluation.heuristics import gate as heuristics  # noqa: E402


def main():
    args = runs.parser(__doc__).parse_args()
    task = runs.Task(args, "heuristics")
    if task.done():
        return
    scorer = Scorer(open_assets(runs.load_config(args.config), args.device), args.seed, task)
    for name, selected in heuristics.heuristics(scorer.model, scorer.val, args.seed, scorer.assets.device).items():
        scorer.add(name, selected)
    task.write(**scorer.payload())


if __name__ == "__main__":
    main()
