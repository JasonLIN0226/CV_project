"""Motion v2 online validation, run again on the frozen readers and features of the study.

    BM_MOTION_ROOT=/scratch/xy3257 python scripts/evaluate/motion_v2_online.py \\
        --config configs/motion_v2/frozen.yaml --seed 0 --device cuda --out <dir> [--intervals ordinary long_gap]

Checks every input against its recorded SHA256 (environments/motion_v2/frozen.py), then replays the
seven project baselines, the 54 tracked deletion-cost and ablation students and the three references
on the 96 validation scene pairs with the unchanged step-3/4 code, and compares each method's raw
losses with the tracked evidence (identical / close / differs). --verify-only stops after the check.

BM_MOTION_ROOT is the folder holding the original deployment folders (/scratch/xy3257 on Burst, or
wherever the motion-v2-frozen-assets-v1 tar was unpacked). BM_REPO defaults to this checkout. Runs on
one GPU with the settings of the original run: deterministic algorithms, no TF32, four threads,
CUBLAS_WORKSPACE_CONFIG=:4096:8. --seed is unused (reader, student and RNG seeds are fixed at 17/29/43).
Writes <out>/online_replay_seed<seed>.json and the full replay under <out>/online/<interval>/.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")      # must be set before the first CUDA matmul
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch  # noqa: E402

from bounded_memory import runs  # noqa: E402
from bounded_memory.environments.motion_v2 import frozen  # noqa: E402

HEADLINE = ("baseline/fifo", "baseline/attention", "student/set/use_time_position", "reference/b", "reference/posthoc_best")


def main():
    ap = runs.parser(__doc__)
    ap.add_argument("--intervals", nargs="+", default=["ordinary"], choices=frozen.INTERVALS)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("BM_REPO", str(runs.REPO))
    task = runs.Task(args, "online_replay")
    if task.done():
        return
    paths = frozen.Paths.from_config(runs.load_config(args.config)["assets"])

    report = frozen.verify(paths, args.intervals)
    task.log("inputs match their checksums: " + ", ".join(f"{k} {v.get('checked', 'ok')}" for k, v in report.items()))
    if args.verify_only:
        task.write(environment="motion_v2", verification=report, intervals=args.intervals, replay=None)
        return

    torch.set_num_interop_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if str(args.device).startswith("cuda"):
        task.info["gpu"] = torch.cuda.get_device_name(0)
        task.info["cublas_workspace_config"] = os.environ["CUBLAS_WORKSPACE_CONFIG"]

    assets = frozen.load(paths, args.device)
    results = {}
    for interval in args.intervals:
        output = Path(args.out) / "online" / interval
        task.log(f"replaying {interval} into {output}")
        summary = frozen.replay(assets, paths, interval, output)
        comparison = frozen.compare(output / "raw_losses", paths.evidence / "online" / interval / "raw_losses")
        task.log(f"{interval}: {comparison['counts']}")
        for name in HEADLINE:
            row = comparison["methods"].get(name.replace("/", "_"), {})
            if "position_mse_+3" in row:
                v = row["position_mse_+3"]
                task.log(f"  {name:32s} +3 position MSE replay {v['replay']:.6f} tracked {v['tracked']:.6f} ({row['verdict']})")
        results[interval] = {"output": str(output), "comparison": comparison, "summary": summary}
    task.write(environment="motion_v2", verification=report, intervals=args.intervals, replay=results)


if __name__ == "__main__":
    main()
