"""Causal and factorial generator contracts, before any encoder experiment."""
from collections import Counter
from itertools import permutations

import numpy as np
import pytest

from experiments.phase2 import terminal_structures as env
from experiments.phase2.memory import MemoryEntry, StreamingMemory


def test_full_factorial_counts_and_distinct_target_classes():
    base = env.family(20261111, 0, "development")
    report = env.validate_family(base)
    assert report["counts"] == {"R": 12, "D": 12, "S": 48}
    assert report["synergy_local_pixel_keys"] == 12
    assert report["counterfactual_groups"] == 12
    records = env.records_for_family(base)
    assert len({r["id"] for r in records}) == 72
    assert Counter(r["outcome_class"] for r in records if r["task_family"] == "S") == {0: 24, 1: 24}
    assert env.target_class_count("S") == 2
    assert env.target_class_count("R") == env.target_class_count("D") == 4


def test_synergy_pixels_are_strictly_local_and_roles_are_exchangeable():
    # Include a salient slot: salience cannot depend on which role it becomes.
    base = env.family(20261111, 2, "development")
    records = [r for r in env.records_for_family(base) if r["task_family"] == "S"]
    lookup, counts, contexts = {}, Counter(), set()
    for r in records:
        ep = env.episode(r)
        q = r["query"]
        contexts.add(ep["rgb"][q-3:q+1].tobytes())
        singleton = next(j for j, tag in enumerate(r["tags"]) if r["tags"].count(tag) == 1)
        assert r["semantic_correct_delete_set"] == [singleton]
        pair = [j for j in range(3) if j != singleton]
        assert r["outcome_class"] == r["slot_payload"][pair[0]] ^ r["slot_payload"][pair[1]]
        for j, t in enumerate(r["evidence"]):
            key = (j, r["tags"][j], r["slot_payload"][j])
            raw = ep["rgb"][t].tobytes()
            assert key not in lookup or lookup[key] == raw
            lookup[key] = raw
            counts[(j, r["tags"][j], r["slot_payload"][j], j == singleton)] += 1
    assert len(contexts) == 1
    assert len(lookup) == 12
    # Each local slot/tag/payload appears as unique in 1/3 and paired in 2/3.
    for j, tag, bit in lookup:
        assert counts[(j, tag, bit, False)] == 2*counts[(j, tag, bit, True)]


def test_style_changes_only_the_declared_cue_and_not_future_or_context():
    base = env.family(20261111, 13, "development")
    records = env.records_for_family(base)
    for task, styles in (("R", env.R_STYLES), ("D", env.D_STYLES)):
        neutral = {tuple(r["bits"]): r for r in records if r["task_family"] == task and r["condition"] == "neutral"}
        for r in records:
            if r["task_family"] != task or r["condition"] == "neutral":
                continue
            left, right = env.episode(neutral[tuple(r["bits"])]), env.episode(r)
            changed = np.flatnonzero(np.any(left["rgb"] != right["rgb"], axis=(1, 2, 3))).tolist()
            assert changed == [r["evidence"][r["salient_slot"]]]
            assert np.array_equal(left["states"], right["states"])
            assert np.array_equal(left["actions"], right["actions"])


def test_legal_fifo_parent_and_single_terminal_boundary():
    base = env.family(20261111, 0, "development")
    stream = StreamingMemory(4, 2, "fifo")
    q = base["query"]
    for t in range(q):
        stream.push(MemoryEntry("terminal-test", t, np.asarray([t], np.float32)))
    before = stream.snapshot
    candidates = before.retained+(before.recent[0],)
    assert [entry.timestamp for entry in candidates] == base["cue_times"]
    final = stream.push(MemoryEntry("terminal-test", q, np.asarray([q], np.float32)))
    assert [entry.timestamp for entry in final.recent] == list(range(q-3, q+1))
    assert base["retention_stop"] == q


def test_future_action_override_has_no_past_or_cue_path():
    records = env.records_for_family(env.family(20261111, 4, "development"))
    for task in env.TASK_FAMILIES:
        r = next(r for r in records if r["task_family"] == task)
        original = env.episode(r)
        changed = env.episode(r, np.full((6, 2), .81, np.float32))
        q = r["query"]
        assert np.array_equal(original["rgb"][:q+1], changed["rgb"][:q+1])
        assert np.array_equal(original["actions"][:q], changed["actions"][:q])
        assert not np.array_equal(original["states"][q+1:], changed["states"][q+1:])
        assert np.all(original["actions"][:q] == 0)
        assert np.array_equal(original["states"][:q+1], np.repeat(original["states"][:1], q+1, axis=0))
        assert original["observation_keys"] == ("rgb", "actions")


def test_target_only_renderer_matches_full_episode_and_future_cache_key():
    records = env.records_for_family(env.family(20261111, 2, "development"))
    plan = np.full((6, 2), .92, np.float32)
    cached = {}
    for r in records:
        target = env.target_frames(r, plan)
        ep = env.episode(r, plan)
        assert np.array_equal(target, ep["rgb"][np.asarray(r["target_times"])])
        assert np.array_equal(env.reference_plan(r), env.episode(r)["actions"][r["query"]:r["query"]+6])
        key = (r["family"], r["task_family"], r["outcome_class"])
        assert key not in cached or np.array_equal(cached[key], target)
        cached[key] = target
    assert len(cached) == 4+4+2


def test_family_design_balances_axis_order_salient_carrier_and_splits():
    cells, s_slots, carriers = Counter(), Counter(), Counter()
    ids = set()
    for index in range(24):
        base = env.family(20261111, index, "development")
        cells[(base["duplicate_axis"], base["order_index"])] += 1
        s_slots[base["synergy_salient_slot"]] += 1
        carriers[(base["salient_redundant_role"], base["salient_useful_role"])] += 1
        ids.add(base["family_id"])
        assert len(env.records_for_family(base)) == 72
    assert len(cells) == 12 and set(cells.values()) == {2}
    assert set(s_slots.values()) == {6}
    assert set(carriers.values()) == {12}
    assert not ids & {env.family(20261112, i, "confirmation")["family_id"] for i in range(24)}


def test_independent_ordering_bound_matches_balanced_six_tag_patterns():
    # For each fixed payload-at-position tuple, any independent score induces
    # an ordering over six (position,tag) entries; even position-aware scores
    # cannot correctly identify more than3/6 singleton tags. No learned model.
    entries = [(j, tag) for j in range(3) for tag in range(2)]
    best = 0
    for ordering in permutations(entries):
        rank = {entry: k for k, entry in enumerate(ordering)}
        success = 0
        for tags in env.TAG_PATTERNS:
            drop = min(range(3), key=lambda j: rank[(j, tags[j])])
            success += tags.count(tags[drop]) == 1
        best = max(best, success)
    assert best == 3


def test_invalid_generator_inputs_fail_before_rendering():
    with pytest.raises(ValueError):
        env.family(12, -1, "train")
    with pytest.raises(ValueError):
        env.family(12, 0, "unlocked_test_name")
    r = env.records_for_family(env.family(12, 0, "train"))[0]
    with pytest.raises(ValueError):
        env.episode(r, np.zeros((6, 2), np.float32))
    with pytest.raises(ValueError):
        env.episode(r, np.ones((5, 2), np.float32))
