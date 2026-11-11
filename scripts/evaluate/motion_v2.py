"""Motion v2 artifact replay: recompute the published tables from the tracked loss arrays.

    python scripts/evaluate/motion_v2.py --config configs/motion_v2/evidence.yaml --seed 0 --device cpu --out <dir>

Nothing is run on the Motion environment: the frozen readers and features are not available to
this account (see configs/reproduction.yaml). This recomputes, from the evidence the study
committed, the online validation table for both intervals and both horizons, the reader
qualification arms, and which checkpoint each student kept. --seed is unused (the evidence already
pools three reader seeds and three student seeds). Writes <out>/evidence_seed0.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bounded_memory import runs  # noqa: E402
from bounded_memory.environments.motion_v2 import evidence  # noqa: E402


def main():
    args = runs.parser(__doc__).parse_args()
    task = runs.Task(args, "evidence")
    if task.done():
        return
    assets = runs.load_config(args.config)["assets"]
    online = {interval: evidence.online_table(assets["step34"], interval) for interval in ("ordinary", "long_gap")}
    arms = evidence.reference_arms(assets["step2"])
    selection = evidence.student_selection(assets["step34"])
    untrained = sorted(a for a, seeds in selection.items() if not any(s["is_trained"] for s in seeds.values()))
    task.log(f"{len(online['ordinary'])} methods; {len(untrained)} of {len(selection)} student arms kept their initial weights on every seed")
    for name in ("baseline/attention", "student/set_use_time_position", "reference/b"):
        task.log(f"{name:34s} position MSE +3 {online['ordinary'][name]['position_mse']['3']:.8f}")
    task.write(environment="motion_v2", online=online, reference_arms=arms, student_selection=selection,
               untrained_arms=untrained)


if __name__ == "__main__":
    main()
