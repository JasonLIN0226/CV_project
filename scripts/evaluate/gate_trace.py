"""Trace the released deletion-cost scorer on one real Gate validation episode.

    python scripts/evaluate/gate_trace.py --config configs/gate/team_bundle.yaml \
        --seed 17 --out runs/gate_trace_case [--episode-id ID] [--students DIR]

Requires a complete Gate asset tree, cached features and frozen model checkpoints.
Defaults to the first validation episode in manifest order and to CPU. No downloads
or training are performed. Writes gate_trace_seed<seed>.json; use --force to replace.
"""
import pickle
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bounded_memory import runs  # noqa: E402
from bounded_memory.data.io import file_hash  # noqa: E402
from bounded_memory.environments.gate.experiment import open_assets  # noqa: E402
from bounded_memory.environments.gate.protocol import SEEDS  # noqa: E402
from bounded_memory.evaluation.gate_trace import POLICY, format_trace, trace_episode  # noqa: E402
from bounded_memory.methods.deletion_cost.training import frozen_scorers  # noqa: E402


def select_episode(assets, episode_id):
    episodes = assets.split("validation")
    if not episodes:
        raise ValueError("Gate asset tree contains no validation episodes")
    if episode_id is None:
        return episodes[0]
    for episode in episodes:
        if episode.id == episode_id:
            return episode
    examples = ", ".join(e.id for e in episodes[:5])
    raise ValueError(f"validation episode {episode_id!r} not found; example IDs: {examples}")


def require_output_available(path, force):
    if (path.exists() or path.is_symlink()) and not force:
        raise FileExistsError(f"output already exists: {path}; use a different --out or pass --force")


def main(argv=None):
    ap = runs.parser(__doc__)
    ap.add_argument("--episode-id", help="validation episode ID (default: first in manifest order)")
    ap.add_argument("--students", type=Path, help="folder containing student_<seed>.pt")
    args = ap.parse_args(argv)
    if args.seed not in SEEDS:
        ap.error(f"--seed must be one of {SEEDS}")
    if args.threads < 1:
        ap.error("--threads must be positive")

    try:
        task = runs.Task(args, "gate_trace")
        require_output_available(task.path, args.force)
        config = runs.load_config(args.config)
        if not isinstance(config, dict) or config.get("environment") != "gate":
            raise ValueError("--config must name environment: gate")
        spec = config.get("assets")
        if not isinstance(spec, dict) or not spec.get("root"):
            raise ValueError("Gate config requires assets.root")
        students = args.students or spec.get("students")
        if not students:
            raise ValueError("no scorer checkpoint folder: pass --students or set assets.students")
        root, students = Path(spec["root"]).resolve(), Path(students).resolve()
        student_path = students / f"student_{args.seed}.pt"
        missing = [p for p in (root / "resolved_config.json", student_path) if not p.is_file()]
        if missing:
            raise FileNotFoundError("missing required Gate assets:\n  " + "\n  ".join(map(str, missing))
                                    + "\nObtain the Gate dataset/features and model bundle; this command does not download them.")

        assets = open_assets(config, args.device)
        episode = select_episode(assets, args.episode_id)
        predictor = assets.predictor(args.seed)
        scorers, student_info = frozen_scorers(students, args.seed, assets.device)
        result = trace_episode(assets, episode, scorers[POLICY], predictor, args.seed)
        task.info.update(
            asset_root=str(root), config_sha256=file_hash(args.config),
            entrypoint_sha256=file_hash(__file__),
            students={**student_info, "path": str(student_path)},
            predictor={"path": str(root / f"predictor/gate/seed_{args.seed}/best.pt"),
                       "sha256": assets.predictor_sha256[args.seed]},
        )
        # Check again after loading/replay, before the existing atomic JSON writer.
        require_output_available(task.path, args.force)
        task.write(**result)
        print(format_trace(result))
        return 0
    except KeyError as exc:
        print(f"Gate trace failed: configuration/asset metadata: {exc}", file=sys.stderr)
    except (OSError, ValueError, RuntimeError, pickle.UnpicklingError, yaml.YAMLError) as exc:
        print(f"Gate trace failed: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
