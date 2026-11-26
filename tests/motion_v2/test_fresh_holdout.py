"""scripts/evaluate/motion_v2_fresh_holdout.py: final-pair lookup and the paired scene-pair comparison."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "fresh_holdout", Path(__file__).resolve().parents[2] / "scripts/evaluate/motion_v2_fresh_holdout.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


def test_final_losses_and_paired_difference():
    episodes = [{"rec": {"id": f"e{i}", "pair": f"p{i // 2}"}} for i in range(4)]
    tables = {f"e{i}": np.arange(2 * 4 * 4, dtype=float).reshape(2, 4, 4) + i for i in range(4)}
    kept = [(0, 1), (1, 2), (2, 3), (0, 3)]
    got = script.final_losses(tables, episodes, kept)
    assert got.shape == (4, 2) and got[1, 0] == tables["e1"][0, 1, 2] and got[3, 1] == tables["e3"][1, 0, 3]
    attention = np.ones((4, 2))
    ours = attention.copy()
    ours[:2] -= 0.5                                  # better on pair p0 only
    p = script.paired(ours, attention, episodes, resamples=500)
    assert p["pairs"] == 2 and np.isclose(p["difference"], -0.25) and p["share_of_pairs_lower"] == 0.5
    assert p["ci95"][0] <= -0.25 <= p["ci95"][1]
