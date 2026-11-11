"""Gate reference points for one predictor seed: recent frames only, the privileged cue keyframe,
and the unbounded full history.

    python scripts/run_oracles/gate.py --config configs/gate/team_bundle.yaml --seed 17 --device cpu --out <dir>

Writes <out>/references_seed<seed>.json. Nothing is trained.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bounded_memory import runs  # noqa: E402
from bounded_memory.environments.gate.experiment import Scorer, open_assets  # noqa: E402
from bounded_memory.evaluation.oracles import gate as oracles  # noqa: E402


def main():
    args = runs.parser(__doc__).parse_args()
    task = runs.Task(args, "references")
    if task.done():
        return
    scorer = Scorer(open_assets(runs.load_config(args.config), args.device), args.seed, task)
    scorer.add("recent_only", oracles.recent_only(scorer.val))
    scorer.add("oracle_keyframe", oracles.oracle_keyframe(scorer.val))
    scorer.add("full_history", oracles.full_history(scorer.val))
    task.write(**scorer.payload())


if __name__ == "__main__":
    main()
