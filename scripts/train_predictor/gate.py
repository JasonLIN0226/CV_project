"""Regenerate the Gate assets from scratch: dataset, I-JEPA features, bases, state decoder and the
three predictors.

    python scripts/train_predictor/gate.py --stage all --work <tree> --device cpu

Stages (each resumes; rerun after an interruption):

    data        the clean-future Gate dataset from the repo generator (seed 6301); byte-identical to the team's
    features    frozen I-JEPA encoding, train-only normalisation, state-decoder probes
    predictors  three DynamicsPredictors (seeds 17, 29, 43), 120 epochs each
    pins        records the SHA256 of each trained checkpoint in <tree>/predictor/gate/checksums.json,
                which is how the new package verifies a predictor before loading it
    all         the four stages in order

Stage reports are written next to the tree (gate_features_report/, gate_training_report/).
Every stage runs the package: environments/gate/generate.py (scenes: design.py, structured_design.py),
environments/gate/features.py and environments/gate/train_predictor.py, on the predictor source in
predictors/phase1/. scripts/prepare_data/gate.py and scripts/encode/ijepa_gate.py run single stages.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "configs/gate/dataset_config_template.json"
STAGES = ("data", "features", "predictors", "pins")


def run(module, *args):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    command = [sys.executable, "-m", f"bounded_memory.environments.gate.{module}", *map(str, args)]
    print("==", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def stage_data(work, device, threads):
    if not (work / "dataset/manifest.json").exists():
        if work.exists():
            sys.exit(f"partial dataset at {work}; remove it and rerun")
        run("generate", "--source-config", TEMPLATE, "--output", work)
    path = work / "resolved_config.json"
    config = json.loads(path.read_text())
    manifest = json.loads((work / "dataset/manifest.json").read_text())
    splits = {}
    for record in manifest["records"]:
        splits[record["split"]] = splits.get(record["split"], 0) + 1
    if splits != {"train": 1024, "validation": 192} or not json.loads((work / "dataset/split_audit.json").read_text())["passed"]:
        sys.exit(f"unexpected dataset: {splits}")
    config.update(output=str(work), device=device, threads=threads)
    path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"dataset ok {splits}", flush=True)


def stage_features(work, device, threads):
    run("features", "--config", work / "resolved_config.json", "--output", work.parent / "gate_features_report")


def stage_predictors(work, device, threads):
    run("train_predictor", "--config", work / "resolved_config.json", "--output", work.parent / "gate_training_report")


def stage_pins(work, device, threads):
    folder = work / "predictor/gate"
    pins = {}
    for seed in (17, 29, 43):
        path = folder / f"seed_{seed}/best.pt"
        if not path.exists():
            sys.exit(f"predictor for seed {seed} is missing: {path}")
        pins[str(seed)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (folder / "checksums.json").write_text(json.dumps(pins, indent=2) + "\n")
    print(f"wrote {folder / 'checksums.json'}", flush=True)


def main(default_stage=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default=default_stage, required=default_stage is None, choices=STAGES + ("all",))
    ap.add_argument("--work", type=Path, required=True, help="the Gate asset tree to create or resume")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    args = ap.parse_args()
    work = args.work.resolve()
    for stage in (STAGES if args.stage == "all" else (args.stage,)):
        print(f"== stage {stage}", flush=True)
        globals()[f"stage_{stage}"](work, args.device, args.threads)


if __name__ == "__main__":
    main()
