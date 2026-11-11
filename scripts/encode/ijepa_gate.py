"""Encode a Gate dataset with the frozen I-JEPA encoder and fit the train-only normalisation and state-decoder probes.

    python scripts/encode/ijepa_gate.py --work <tree> --device cpu

One stage of scripts/train_predictor/gate.py (see there); it resumes when rerun.
"""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("gate_rebuild", Path(__file__).resolve().parents[1] / "train_predictor/gate.py")
gate_rebuild = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate_rebuild)

if __name__ == "__main__":
    gate_rebuild.main(default_stage="features")
