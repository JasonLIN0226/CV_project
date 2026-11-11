"""Eight of the original Motion v2 tests need the frozen dataset or a retained output folder and
fail without them. Skip those when the variables they read are unset; the test files are unchanged.

    MOTION_V2_DATASET, MOTION_V2_AUDIT_OUTPUT     tests that use the real_dataset_audit fixture
    MOTION_V2_SOFTWARE_TEST_OUTPUT                test_average_siblings_and_seeds_before_pair_bootstrap
"""
import os

import pytest


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "real_dataset_audit" in getattr(item, "fixturenames", ()) and not (
                os.environ.get("MOTION_V2_DATASET") and os.environ.get("MOTION_V2_AUDIT_OUTPUT")):
            item.add_marker(pytest.mark.skip(reason="needs the frozen Motion v2 dataset: set MOTION_V2_DATASET and MOTION_V2_AUDIT_OUTPUT"))
        if item.name == "test_average_siblings_and_seeds_before_pair_bootstrap" and not os.environ.get("MOTION_V2_SOFTWARE_TEST_OUTPUT"):
            item.add_marker(pytest.mark.skip(reason="writes retained files: set MOTION_V2_SOFTWARE_TEST_OUTPUT to a new folder"))
