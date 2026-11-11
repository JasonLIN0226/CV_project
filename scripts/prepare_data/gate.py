"""Generate the clean-future Gate dataset (seed 6301) with the frozen repo generator.

    python scripts/prepare_data/gate.py --work <tree> --device cpu

One stage of scripts/train_predictor/gate.py (see there); it resumes when rerun.
"""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("gate_rebuild", Path(__file__).resolve().parents[1] / "train_predictor/gate.py")
gate_rebuild = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate_rebuild)

if __name__ == "__main__":
    gate_rebuild.main(default_stage="data")
