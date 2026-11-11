"""Registered tests over the actual saved Motion v2.1 train/validation dataset.

Set MOTION_V2_DATASET to the dataset directory and MOTION_V2_AUDIT_OUTPUT to a
new JSON output path. The session performs the audit once; tests consume that
same report. No synthetic fixture generation or test-split access is allowed.
"""
import importlib.util
import os
from pathlib import Path

MOTION = Path(__file__).resolve().parents[2] / "src/bounded_memory/environments/motion_v2"   # moved from experiments/motion_v2

import pytest


@pytest.fixture(scope="session")
def real_dataset_audit():
    dataset = os.environ.get("MOTION_V2_DATASET")
    output = os.environ.get("MOTION_V2_AUDIT_OUTPUT")
    if not dataset or not output:
        pytest.fail("MOTION_V2_DATASET and MOTION_V2_AUDIT_OUTPUT must be explicitly set after preregistration")
    source = MOTION.joinpath("audit_v2.py")
    spec = importlib.util.spec_from_file_location("motion_v2_registered_audit", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.write_audit(dataset, output)


def test_real_train_validation_pairs_and_physics(real_dataset_audit):
    structure = real_dataset_audit["structure"]
    assert structure["status"] == "passed", structure["failures"]
    assert structure["pairs_checked"] == 608
    assert structure["pairs_passed"] == 608


def test_no_test_data_or_hidden_model_input(real_dataset_audit):
    assert real_dataset_audit["raw_test_access"] is False
    assert real_dataset_audit["split_pairs"] == {"train": 512, "validation": 96}
    assert real_dataset_audit["structure"]["status"] == "passed"


def test_online_history_headroom(real_dataset_audit):
    online = real_dataset_audit["online"]
    assert online["status"] == "passed", online
    for split in ("train", "validation"):
        assert online["splits"][split]["passed"], online["splits"][split]


@pytest.mark.parametrize("subset", ["end_only", "one_post_non_end", "stale_plus_end", "two_post"])
def test_train_only_grouped_leakage_thresholds(real_dataset_audit, subset):
    leakage = real_dataset_audit["leakage"]
    assert leakage.get("source_split") == "train", leakage
    assert leakage["scenes"] == 512 and leakage["examples"] == 1024
    assert leakage["group_unit"] == "sibling_pair" and leakage["folds"] == 5
    result = leakage["results"][subset]
    assert result["passed"], {"subset": subset, **result}
