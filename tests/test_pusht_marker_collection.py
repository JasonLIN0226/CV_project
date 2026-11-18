"""Portable collection tests use v5's dedicated software-test seeds."""
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("gym_pusht")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/phase2"))
import pusht_marker_collection as portable


@pytest.mark.parametrize("installed", ["opencv-python", "opencv-python-headless"])
def test_opencv_distribution(installed):
    def version(name):
        if name == installed:
            return "4.11"
        raise portable.metadata.PackageNotFoundError(name)
    assert portable.opencv_packages(version) == {installed: "4.11"}


def test_portable_collection_matches_frozen_simulator(tmp_path, monkeypatch):
    marker = portable.marker
    seed = marker.TEST_ONLY_SEEDS[0]
    monkeypatch.setattr(marker, "SPLIT_COUNTS", {"train": 2, "evaluation": 1})
    monkeypatch.setattr(marker, "SPLIT_BASE_SEEDS", {"train": seed})  # accessing evaluation would fail
    serial, parallel = tmp_path / "serial", tmp_path / "parallel"
    for output, workers in ((serial, 1), (parallel, 2)):
        receipt = portable.run(output, workers, ["train"])
        assert set(receipt["counts"]) == {"train"} and receipt["counts"]["train"]["episodes"] == 6
        assert all("\\" not in key for key in receipt["artifacts"])
    index = json.loads((serial / "observation_index.json").read_text())
    assert index == json.loads((parallel / "observation_index.json").read_text())
    for row in index["records"][::3]:
        original = marker.collect_candidate(("train", row["environment_seed"]))
        assert original["valid"]
        for member, label in enumerate(marker.sibling_classes(original["seed"])):
            name = f"observations/{row['family_id']}_s{member}.npz"
            with np.load(parallel / name) as observed:
                np.testing.assert_array_equal(observed["rgb"], original["runs"][label]["rgb"])
                np.testing.assert_array_equal(observed["actions_absolute"], original["actions"].astype(np.float32))
    with pytest.raises(FileExistsError):
        portable.run(serial, 1, ["train"])
    protocol = json.loads((serial / "protocol.json").read_text())
    with pytest.raises(ValueError, match="one episode per class"):
        marker.audit_dataset(serial, protocol, index["records"][1:])
    (serial / index["records"][0]["observations"]).write_bytes(b"changed observation")
    with pytest.raises(ValueError, match="checksum"):
        marker.audit_dataset(serial, protocol, index["records"])
