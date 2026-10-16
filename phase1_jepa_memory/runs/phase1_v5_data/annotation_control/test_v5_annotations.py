"""Tests for unselected-label exclusion, official-split independence and failures."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from audit_v5_annotations import development_segments, range_compatible, action_catalog


def test_unselected_sealed_labels_never_interpreted():
    class Poison(dict):
        def __getitem__(self, key):
            raise AssertionError("unselected labels accessed")
    rows = [Poison(video="sealed/C10095_rgb.mp4"),
            Poison(video="dev/C10115_rgb.mp4"),
            {"video": "dev/C10095_rgb.mp4", "start_frame": "0", "end_frame": "14",
             "action_id": "0", "verb_id": "2", "noun_id": "3"}]
    selected = {"dev/C10095_rgb.mp4": {"participant": "p1", "split": "validation"}}
    output = list(development_segments(rows, selected, {0: (2, 3)}))
    assert len(output) == 1
    assert output[0]["split"] == "validation"
    assert output[0]["labels"] == (0, 2, 3)


def test_missing_or_challenge_labels_are_unavailable_not_background():
    selected = {"dev/C10095_rgb.mp4": {"participant": "p1", "split": "train"}}
    row = {"video": "dev/C10095_rgb.mp4", "start_frame": "5", "end_frame": "12"}
    assert list(development_segments([row], selected, {0: (2, 3)}))[0]["labels"] is None
    row.update(action_id="-1", verb_id="-1", noun_id="-1")
    assert list(development_segments([row], selected, {0: (2, 3)}))[0]["labels"] is None


def test_selected_inconsistent_labels_and_intervals_rejected():
    selected = {"dev/C10095_rgb.mp4": {"participant": "p1", "split": "train"}}
    row = {"video": "dev/C10095_rgb.mp4", "start_frame": "5", "end_frame": "12",
           "action_id": "0", "verb_id": "3", "noun_id": "3"}
    with pytest.raises(ValueError, match="vocabulary"):
        list(development_segments([row], selected, {0: (2, 3)}))
    row.update(start_frame="13", verb_id="2")
    with pytest.raises(ValueError, match="interval"):
        list(development_segments([row], selected, {0: (2, 3)}))


def test_numeric_range_gate_does_not_decide_origin():
    assert range_compatible(0, 29, 60)
    assert range_compatible(1, 30, 60)
    assert not range_compatible(0, 31, 60)
    assert not range_compatible(-1, 0, 60)
    assert not range_compatible(5, 4, 60)


def test_duplicate_dictionary_ids_rejected():
    row = {"action_id": "0", "verb_id": "2", "noun_id": "3"}
    with pytest.raises(ValueError):
        action_catalog([row, row])
