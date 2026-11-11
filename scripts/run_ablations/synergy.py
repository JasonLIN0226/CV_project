"""Ablations of our method on Synergy for one reader and student seed: independent scoring,
future-attention and shuffled labels, the appearance-only control and the distilled-sensitivity
students. Same arguments as scripts/evaluate/synergy.py, with --arms ablations as the default.
Writes <out>/students_ablations_<reader>_seed<seed>.json.
"""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("evaluate_synergy", Path(__file__).resolve().parents[1] / "evaluate/synergy.py")
evaluate_synergy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate_synergy)

if __name__ == "__main__":
    evaluate_synergy.main(default_arms="ablations")
