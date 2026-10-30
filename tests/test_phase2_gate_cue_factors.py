from itertools import product

import numpy as np
import pytest

from experiments.phase2.diagnose_gate_cue_factors import factor_effects, factor_entries


def test_factorial_interactions_and_order_average_allocation():
    values = {(c, t, a): 2*c+3*t+5*a+6*c*t+9*c*t*a for c, t, a in product((0, 1), repeat=3)}
    result = factor_effects(values)
    assert result['interaction_content_age_past_action'] == 9
    assert result['interaction_content_age'] == 10.5
    assert result['main_content'] == 7.25
    assert result['endpoint_allocation_content'] == pytest.approx(8)
    assert result['endpoint_allocation_age'] == pytest.approx(9)
    assert result['endpoint_allocation_past_action'] == pytest.approx(8)
    assert result['endpoint_gap'] == 25
    with pytest.raises(ValueError): factor_effects({(0, 0, 0): 1})


def test_content_age_and_executed_actions_are_independent_and_filler_fixed():
    rec = {'id': 'example', 'query': 8}
    z = np.arange(45, dtype=np.float32).reshape(15, 3)
    sibling = z+1000
    actions = np.arange(28, dtype=np.float32).reshape(14, 2)
    comparison = {'with_cue': [1, 4], 'without_cue': [3, 4]}
    for content, age, action in product((0, 1, 2), (0, 1), (0, 1)):
        recent, history, past, meta = factor_entries(rec, z, sibling, actions, comparison, content, age, action)
        assert [e.timestamp for e in recent] == [5, 6, 7, 8]
        slot = history[meta['slot_index']-4]
        expected_time = (3, 1)[age]
        assert slot.timestamp == expected_time
        assert np.array_equal(slot.representation, (z[3], z[1], sibling[1])[content])
        assert np.array_equal(past[slot.entry_id], actions[(3, 1)[action]:8].sum(0, dtype=np.float32))
        filler = next(e for e in history if e.timestamp == 4)
        assert np.array_equal(filler.representation, z[4])
        assert np.array_equal(past[filler.entry_id], actions[4:8].sum(0, dtype=np.float32))


def test_slot_can_cross_filler_order_without_changing_its_role():
    rec = {'id': 'example', 'query': 8}
    z = np.arange(45, dtype=np.float32).reshape(15, 3)
    actions = np.zeros((14, 2), np.float32)
    comparison = {'with_cue': [1, 3], 'without_cue': [3, 4]}
    for age, expected_index in ((0, 5), (1, 4)):
        _, history, _, meta = factor_entries(rec, z, z+10, actions, comparison, 1, age, 1)
        assert meta['slot_index'] == expected_index
        assert [e.timestamp for e in history] == sorted(e.timestamp for e in history)
        assert np.array_equal(history[expected_index-4].representation, z[1])
