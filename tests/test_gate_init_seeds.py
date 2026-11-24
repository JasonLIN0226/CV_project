"""The extra-initialisation fit in scripts/train_baselines/gate_init_seeds.py is the registered fit for the registered seed."""
import importlib.util
from pathlib import Path

import numpy as np

from bounded_memory.methods.deletion_cost import training

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gate_init_seeds", ROOT / "scripts/train_baselines/gate_init_seeds.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


def _labels(rng, n=4096):
    x = rng.normal(size=(n, 3, 517)).astype(np.float32)
    y = rng.random((n, 3)).astype(np.float32)
    return x, y / y.sum(1, keepdims=True), np.roll(y, 1, 1) / y.sum(1, keepdims=True)


def test_mirror_equals_registered_fit_for_the_registered_seed():
    x, ya, yd = _labels(np.random.default_rng(0))
    registered = training.fit_pair(x, ya, yd, seed=17, split="train", device="cpu")
    mirror = script.fit_with_init(x, ya, yd, 17, "cpu")
    assert script.max_weight_difference(registered["deletion"], mirror["deletion_cost_student"]) == 0.0
    assert script.max_weight_difference(registered["attention"], mirror["attention_proxy_student"]) == 0.0


def test_another_init_changes_the_weights():
    x, ya, yd = _labels(np.random.default_rng(1))
    a = script.fit_with_init(x, ya, yd, 17, "cpu")["deletion_cost_student"]
    b = script.fit_with_init(x, ya, yd, 101, "cpu")["deletion_cost_student"]
    assert script.max_weight_difference(a, b) > 0
