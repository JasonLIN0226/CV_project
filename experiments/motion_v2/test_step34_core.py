"""Hand-built causal replay tests, with no real data, model fitting, or GPU use.

The registered software preflight may import this file only after source freeze.
Writers and frozen-model services are replaced by in-memory test doubles.
"""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


_SPEC = importlib.util.spec_from_file_location("motion_step34_core_tested", Path(__file__).with_name("step34_core.py"))
core = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(core)


def manual_episode(query=9):
    length = query + 7
    projected = np.arange(length * 128, dtype=np.float32).reshape(length, 128) / 1000
    features = np.repeat(projected, 40, axis=1)
    actions = np.arange((length - 1) * 2, dtype=np.float32).reshape(length - 1, 2) / 100
    prefix = np.concatenate([np.zeros((1, 2), np.float32), np.cumsum(actions, axis=0)])
    visible = np.zeros(length, dtype=bool)
    visible[[0, 1, 3, 5]] = True
    return {"rec": {"id": "1" * 24, "pair": "a" * 24, "query": query, "frames": length},
            "p": projected, "z": features, "actions": actions, "prefix": prefix,
            "meta": {"states": np.zeros((length, 2), np.float32), "visible": visible,
                     "t_c": np.int64(3), "end": np.int64(5), "sufficient": visible & (np.arange(length) >= 3)}}


@pytest.mark.parametrize("query", [9, 10, 11, 20, 65])
def test_exactly_four_public_overflow_clocks(query):
    clocks = core.sample_clocks(query)
    assert clocks == tuple(6 + j * (query - 6) // 3 for j in range(4))
    assert len(clocks) == len(set(clocks)) == 4
    assert clocks[0] == 6 and clocks[-1] == query
    assert all(a < b for a, b in zip(clocks, clocks[1:]))


@pytest.mark.parametrize("query", [8, -1, 9.0, True])
def test_invalid_sampling_query_is_rejected(query):
    with pytest.raises(ValueError):
        core.sample_clocks(query)


def test_candidate_inputs_have_only_registered_deployable_values():
    ep = manual_episode(13)
    clock, pool = 9, [0, 2, 5]
    items, context = core.causal_features(ep["p"], ep["prefix"], clock, pool)
    assert items.shape == (3, 259) and context.shape == (524,)
    assert items.dtype == context.dtype == np.float32
    assert np.array_equal(items[:, :128], ep["p"][pool])
    assert np.array_equal(items[:, 128:256], ep["p"][pool] - ep["p"][clock])
    assert np.allclose(items[:, 256], np.log1p(clock - np.asarray(pool)))
    assert np.array_equal(items[:, 257:], ep["prefix"][clock] - ep["prefix"][pool])
    assert np.array_equal(context[:512].reshape(4, 128), ep["p"][6:10])
    assert np.array_equal(context[512:516], [3, 2, 1, 0])
    assert np.array_equal(context[516:].reshape(4, 2), ep["prefix"][clock] - ep["prefix"][6:10])


def test_future_hidden_and_discarded_frames_cannot_change_causal_features():
    ep = manual_episode(13)
    clock, pool = 9, [0, 2, 5]
    before = core.causal_features(ep["p"], ep["prefix"], clock, pool)
    changed = copy.deepcopy(ep)
    # Neither discarded observations nor future observations are selector inputs.
    changed["p"][[1, 3, 4]] = -9000
    changed["p"][clock + 1:] = 9000
    changed["actions"][clock:] = -1000
    changed["prefix"] = np.concatenate([np.zeros((1, 2), np.float32), np.cumsum(changed["actions"], axis=0)])
    changed["meta"] = {"states": np.full_like(ep["meta"]["states"], 1000), "visible": ~ep["meta"]["visible"],
                       "t_c": -100, "end": 999, "label": 1, "sufficient": ~ep["meta"]["sufficient"]}
    after = core.causal_features(changed["p"], changed["prefix"], clock, pool)
    assert all(np.array_equal(a, b) for a, b in zip(before, after))


@pytest.mark.parametrize("pool", [[0, 1], [0, 1, 1], [-1, 0, 1], [0, 1, 6], [0.5, 1, 2]])
def test_noncausal_or_noninteger_candidate_pool_is_rejected(pool):
    ep = manual_episode(13)
    with pytest.raises(ValueError):
        core.causal_features(ep["p"], ep["prefix"], 9, pool)


def test_frozen_reader_held_action_input_ignores_future_and_all_hidden_fields():
    ep = manual_episode(13)
    before = core.FrozenAssets.sample(None, ep, 9, [0, 5], held_action=True)
    changed = copy.deepcopy(ep)
    changed["z"][10:] = 8888
    changed["p"][10:] = -8888
    changed["actions"][9:] = 7777
    changed["prefix"] = np.concatenate([np.zeros((1, 2), np.float32), np.cumsum(changed["actions"], axis=0)])
    changed["meta"] = {"arbitrary_hidden_state": object()}
    after = core.FrozenAssets.sample(None, changed, 9, [0, 5], held_action=True)
    assert set(before) == core.MODEL_KEYS
    assert all(np.array_equal(before[k], after[k]) for k in before)
    assert np.array_equal(before["actions"], np.tile(ep["actions"][8], (6, 1)))


@pytest.mark.parametrize("policy", core.BASELINES + ("student",))
def test_every_transition_preserves_B2_and_never_resurrects(policy):
    rng = np.random.default_rng(19)
    kept, discarded = [], set()
    for clock in range(4, 22):
        arrival = clock - 4
        previous = kept.copy()
        pool = previous + [arrival]
        vectors = {t: np.array([t, t * t], np.float32) for t in pool} if policy == "diversity" else {}
        scores = {t: float(t) for t in pool} if policy in ("attention", "surprise", "student") else {}
        kept = core.observable_update(policy, previous, arrival, clock, vectors,
                                      np.array([clock, 0], np.float32), scores, rng)
        assert len(kept) == min(2, arrival + 1)
        assert len(set(kept)) == len(kept) and set(kept) <= set(pool)
        assert not (set(kept) & discarded)
        assert previous == pool[:-1], "The transition mutated its previous-state argument"
        discarded.update(set(pool) - set(kept))


@pytest.mark.parametrize("previous,arrival,clock", [([-1, 0], 2, 6), ([0, 99], 2, 6),
                                                     ([0, 1], 3, 6), ([0, 0], 2, 6),
                                                     ([0, 1, 2], 3, 7), ([0.5, 1], 2, 6)])
def test_illegal_previous_state_or_arrival_is_rejected(previous, arrival, clock):
    with pytest.raises(ValueError):
        core.observable_update("fifo", previous, arrival, clock, {}, None, {}, None)


def test_selectors_reject_archive_wide_inputs_and_scores():
    with pytest.raises(ValueError, match="exactly"):
        core.observable_update("diversity", [0, 1], 2, 6,
                               {t: np.zeros(128) for t in (0, 1, 2, 3)}, np.zeros(128), {}, None)
    with pytest.raises(ValueError, match="exactly"):
        core.observable_update("attention", [0, 1], 2, 6, {}, None, {t: 0. for t in (0, 1, 2, 3)}, None)
    with pytest.raises(ValueError, match="finite"):
        core.observable_update("attention", [0, 1], 2, 6, {}, None, {0: 0., 1: np.nan, 2: 1.}, None)


def test_attention_is_mean_of_reader_softmax_with_correct_keep_direction():
    raw = np.array([[10., 0., 0.], [0., 3., 0.], [0., 3., 0.]])
    exp = np.exp(raw - raw.max(axis=1, keepdims=True))
    expected = (exp / exp.sum(axis=1, keepdims=True)).mean(axis=0)
    actual = core.mean_attention(raw)
    assert np.allclose(actual, expected) and np.isclose(actual.sum(), 1.)
    mean_logits = raw.mean(axis=0)
    wrong = np.exp(mean_logits - mean_logits.max())
    wrong /= wrong.sum()
    assert np.argmax(actual) == 1 and np.argmax(wrong) == 0
    kept = core.observable_update("attention", [0, 1], 2, 6, {}, None, dict(zip([0, 1, 2], actual)), None)
    assert kept == [0, 1], "Low attention is deleted; high attention is retained"


def test_forced_deletion_continuation_cannot_reintroduce_removed_frames():
    # Favor older candidate IDs. The forced-out ID0 is highly salient forever,
    # but it must never re-enter the candidate set.
    tables = np.broadcast_to(-np.arange(16, dtype=np.float64), (3, 16, 16)).copy()
    final = core.continue_attention([0, 1, 2], 0, 6, 9, tables)
    assert final == [1, 2] and 0 not in final
    modified = tables.copy()
    modified[:, :, 0] = 1e9
    modified[:, 10:, :] = -1e9
    for clock in range(7, 10):
        modified[:, clock, clock - 3:] = 1e9
    assert core.continue_attention([0, 1, 2], 0, 6, 9, modified) == final


@pytest.mark.parametrize("pool,drop,clock,query", [([-1, 0, 1], 0, 6, 9), ([0, 1, 99], 2, 6, 9),
                                                  ([0, 1, 1], 0, 6, 9), ([0, 1, 2], -1, 6, 9),
                                                  ([0, 1, 2], 3, 6, 9), ([0, 1, 2], 0, 6, 5)])
def test_invalid_forced_deletion_is_rejected(pool, drop, clock, query):
    tables = np.zeros((3, 16, 16))
    with pytest.raises(ValueError):
        core.continue_attention(pool, drop, clock, query, tables)


@pytest.fixture(scope="module")
def hand_built_behavior_states():
    rows = []
    ep = manual_episode(9)
    for pair_index in range(96):
        pair_id = f"{pair_index:024x}"
        ids = [f"{1000 + 2 * pair_index + slot:024x}" for slot in range(2)]
        for slot, eid in enumerate(ids):
            rows.append({"id": eid, "pair": pair_id, "sibling": ids[1 - slot],
                         "split": "validation", "query": 9})

    class HandAssets:
        store = SimpleNamespace(manifest={"records": rows})

        def episode(self, rec):
            return dict(ep, rec=rec)

        def budget(self):
            return None

    captured, progress_units = {}, []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(core, "write_npz", lambda path, **values: captured.update({str(path): values}))
        patch.setattr(core, "write_json", lambda path, value: captured.update({str(path): value}))
        state_rows, inputs = core.behavior_states(HandAssets(), "validation", Path("unused-in-memory-output"),
                                                 progress=progress_units.append)
    captured["progress_units"] = progress_units
    return state_rows, inputs, captured


def test_four_clocks_four_behaviors_two_siblings_count_and_ids(hand_built_behavior_states):
    rows, inputs, captured = hand_built_behavior_states
    assert len(rows) == len({r["id"] for r in rows}) == 96 * 2 * 4 * 4 == 3072
    assert inputs["items"].shape == (3072, 3, 259)
    assert inputs["context"].shape == (3072, 524)
    for episode_id in {r["episode_id"] for r in rows}:
        selected = [r for r in rows if r["episode_id"] == episode_id]
        for behavior in core.BEHAVIORS:
            assert [r["clock"] for r in selected if r["behavior"] == behavior] == [6, 7, 8, 9]
    assert captured["unused-in-memory-output/states.json"]["multiplicity_preserved"] is True
    assert captured["progress_units"] == [1] * 3072


def test_behavior_identity_retains_duplicates_from_distinct_policies(hand_built_behavior_states):
    rows, _, _ = hand_built_behavior_states
    episode_id = rows[0]["episode_id"]
    initial = [r for r in rows if r["episode_id"] == episode_id and r["clock"] == 6]
    assert len(initial) == 4 and all(r["pool"] == [0, 1, 2] for r in initial)
    assert len({r["id"] for r in initial}) == 4
    for row in initial:
        assert row["id"] == core.digest([row["episode_id"], row["behavior"], row["clock"], row["pool"]])


def test_reservoir_rng_and_sampling_clocks_are_shared_by_siblings(hand_built_behavior_states):
    rows, _, _ = hand_built_behavior_states
    for pair_id in {r["pair_id"] for r in rows}:
        for clock in (6, 7, 8, 9):
            same = [r for r in rows if r["pair_id"] == pair_id and r["clock"] == clock and r["behavior"] == "reservoir"]
            assert len(same) == 2 and same[0]["pool"] == same[1]["pool"]
            assert same[0]["episode_id"] != same[1]["episode_id"]


def test_privileged_reference_sampling_is_shared_and_does_not_use_target_states():
    ep = manual_episode(13)
    assert core.chosen_reference(ep, "a") == []
    assert core.chosen_reference(ep, "b") == [3, 5]
    assert core.chosen_reference(ep, "d") == [1, 5]
    sibling = copy.deepcopy(ep)
    sibling["rec"]["id"] = "2" * 24
    sibling["meta"]["states"][:] = 1000
    sibling["meta"]["label"] = 1
    assert core.chosen_reference(ep, "b_random") == core.chosen_reference(sibling, "b_random") == [3, 5]


def test_raw_teacher_costs_keep_drop_order_and_average_losses_not_features(monkeypatch):
    ep = manual_episode(9)
    written = {}
    costs = {(1, 2): 8., (0, 2): 2., (0, 1): 5.}

    class HandAssets:
        store = SimpleNamespace(records={ep["rec"]["id"]: ep["rec"]})

        def episode(self, rec):
            return ep

        def attention_tables(self, episode):
            return np.broadcast_to(-np.arange(16, dtype=np.float64), (3, 16, 16)).copy()

        def salience(self, candidates, anchor):
            return np.array([[0., 2., 4.], [1., 3., 5.], [2., 4., 6.]])

        def losses(self, requests, reader_index=None, progress=None):
            base = np.array([costs[tuple(kept)] for _, kept in requests])
            offset = np.arange(3)[:, None, None] + np.array([0., 4.])[None, None, :]
            if progress:
                progress(3 * len(requests))
            return {"feature": base[None, :, None] + offset,
                    "position": 10 * base[None, :, None] + offset}

        def budget(self):
            return None

    monkeypatch.setattr(core, "write_npz", lambda path, **values: written.update({Path(path).name: values}))
    monkeypatch.setattr(core, "write_json", lambda path, value: written.update({Path(path).name: value}))
    rows = [{"episode_id": ep["rec"]["id"], "pool": [0, 1, 2], "clock": 6}]
    progress_units = []
    labels = core.deletion_labels(HandAssets(), rows, Path("unused-teacher-output"), progress=progress_units.append)
    assert progress_units == [6, 9, 9]
    assert 1 + sum(progress_units) == 25, "One state unit plus24 teacher units per parent"
    assert labels["local_feature"].tolist() == [[11., 5., 8.]]
    assert labels["use_time_feature"].tolist() == [[11., 5., 8.]]
    assert labels["local_position"].tolist() == [[83., 23., 53.]]
    assert written["continuation_endpoints.npz"]["times"].tolist() == [[[1, 2], [0, 2], [0, 1]]]
    assert written["raw_teacher_losses.npz"]["local_position"].shape == (3, 1, 3, 2)
    assert np.argmin(labels["local_position"][0]) == 1, "Lowest retained-pair cost favors deleting slot1"
    assert np.allclose(labels["future_attention"][0], core.mean_attention(np.array([[0., 2., 4.], [1., 3., 5.], [2., 4., 6.]])))


def test_online_student_deletion_logits_are_negated_before_retaining(monkeypatch):
    ep = manual_episode(6)
    calls = []

    def fixed_student(items, context):
        calls.append((tuple(items.shape), tuple(context.shape)))
        return core.torch.tensor([[0., 3., 1.]], dtype=core.torch.float32)

    normalizer = SimpleNamespace(normalize_inputs=lambda inputs, norm: inputs)
    monkeypatch.setattr(core, "module", lambda name: normalizer if name == "step34_students" else pytest.fail("Unexpected module"))
    assets = SimpleNamespace(device=core.torch.device("cpu"))
    progress_units = []
    kept, trace = core.online_kept(assets, ep, "student", 17, student=(fixed_student, {}, {}), progress=progress_units.append)
    assert kept == [0, 2], "Largest deletion logit must remove slot1, not retain it"
    assert calls == [((1, 3, 259), (1, 524))]
    assert trace[-1] == (6, (0, 1), 2, (0, 2))
    assert progress_units == [1, 1, 60]


def test_plain_online_arrivals_report_one_unit_each():
    progress_units = []
    kept, trace = core.online_kept(SimpleNamespace(), manual_episode(9), "fifo", 17,
                                  progress=progress_units.append)
    assert kept == [4, 5] and len(trace) == 6
    assert progress_units == [1] * 6
