"""scripts/aggregate_results/gate_init_seeds.py: unit means, collapse count and the paired difference."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gate_init_seeds_summary", ROOT / "scripts/aggregate_results/gate_init_seeds.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


def _unit(folder, method, init, predictor, routes):
    policy = f"{script.METHODS[method]}_init{init}"
    rows = [{"policy": policy, "episode": f"e{k // 2}", "horizon": 3 + 3 * (k % 2), "route_accuracy": r,
             "normalized_mse": 0.1} for k, r in enumerate(routes)]
    rows.append({"policy": "fifo", "episode": "e0", "horizon": 3, "route_accuracy": 0.0, "normalized_mse": 1.0})
    (folder / f"{method}_init{init}_seed{predictor}.json").write_text(json.dumps({"rows": rows}))


def test_summary_and_pairing(tmp_path):
    _unit(tmp_path, "ours", 101, 17, [1, 1, 1, 0])
    _unit(tmp_path, "ours", 202, 17, [1, 1, 1, 1])
    _unit(tmp_path, "pacc_recurrent", 101, 17, [1, 0, 0, 0])
    _unit(tmp_path, "pacc_recurrent", 202, 17, [1, 1, 1, 1])
    (tmp_path / "unrelated.json").write_text("{}")
    units = script.load_units(tmp_path)
    assert units["ours"][(17, 101)]["route"] == 0.75          # other policies' rows are ignored
    s = script.method_summary(units["pacc_recurrent"])
    assert s["units"] == 2 and s["collapsed_units"] == 1 and s["route_min"] == 0.25
    p = script.paired(units["ours"], units["pacc_recurrent"], resamples=200)
    assert p["units"] == 2 and p["episodes"] == 2
    assert abs(p["difference_ours_minus_pacc"] - 0.25) < 1e-12
    assert p["units_ours_higher"] == 1 and p["units_tied"] == 1
