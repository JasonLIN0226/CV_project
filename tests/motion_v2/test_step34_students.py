"""Preregistered synthetic checks only: no dataset, encoder, or real fitting.

Do not execute before the supervising protocol/source commit is pushed.
These tests do not call fit_students and do not create/remove any files.
"""
import importlib.util
from pathlib import Path

MOTION = Path(__file__).resolve().parents[2] / "src/bounded_memory/environments/motion_v2"   # moved from experiments/motion_v2

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def students():
    source = MOTION.joinpath("step34_students.py")
    spec = importlib.util.spec_from_file_location("motion_step34_students_software_only", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def artificial_inputs(rows=4):
    rng = np.random.default_rng(901)
    return {"items": rng.normal(size=(rows, 3, 259)).astype(np.float32),
            "context": rng.normal(size=(rows, 524)).astype(np.float32)}


def artificial_labels(students, rows=4):
    costs = np.tile([1., 2., 5.], (rows, 1))
    result = {name: costs * (index + 1) for index, name in enumerate(students.RAW_LABELS[:-1])}
    result["future_attention"] = np.tile([.1, .3, .6], (rows, 1))
    return result


def artificial_ids(rows=4):
    return [f"synthetic-parent-only-{index}" for index in range(rows)]


def test_exact_fixed_budget_and_recipe(students):
    assert students.SEEDS == (17, 29, 43)
    assert students.STEPS == 1000 and students.BATCH_SIZE == 128
    assert students.VALIDATE_EVERY == 100 and students.VALIDATION_BATCH_SIZE == 512
    assert students.TOTAL_UPDATES == 54000
    assert students.TRAIN_ROWS == 16384 and students.VALIDATION_ROWS == 3072
    assert len(students.LABELS) == 9
    assert students.architecture_plan() == {
        "independent": {"width": 64, "parameters": 54401},
        "set": {"width": 60, "parameters": 54361}}


@pytest.mark.parametrize("architecture", ["independent", "set"])
def test_parameter_counts_and_relu_only(students, architecture):
    model = students.TerminalScorer(architecture)
    assert sum(x.numel() for x in model.parameters()) == students.architecture_plan()[architecture]["parameters"]
    assert sum(isinstance(x, torch.nn.ReLU) for x in model.modules()) == 2
    assert not any(isinstance(x, (torch.nn.Dropout, torch.nn.GELU)) for x in model.modules())


@pytest.mark.parametrize("architecture", ["independent", "set"])
def test_candidate_permutation_equivariance(students, architecture):
    data = artificial_inputs()
    model = students.initialized_student(architecture, 17).eval()
    items, context = torch.from_numpy(data["items"]), torch.from_numpy(data["context"])
    permutation = [2, 0, 1]
    with torch.inference_mode():
        original = model(items, context)
        permuted = model(items[:, permutation], context)
    torch.testing.assert_close(permuted, original[:, permutation], rtol=1e-5, atol=1e-6)


def test_independent_candidate_has_no_other_candidate_input(students):
    data = artificial_inputs()
    model = students.initialized_student("independent", 17).eval()
    items, context = torch.from_numpy(data["items"]), torch.from_numpy(data["context"])
    changed = items.clone()
    changed[:, 1:] = changed[:, 1:] * 100 + 27
    with torch.inference_mode():
        original, mutated = model(items, context), model(changed, context)
    torch.testing.assert_close(original[:, 0], mutated[:, 0], rtol=0, atol=0)


def test_deepsets_candidate_receives_other_candidate_pool(students):
    model = students.TerminalScorer("set").eval()
    # A deliberately simple positive synthetic network makes the dependency exact.
    with torch.no_grad():
        for value in model.parameters():
            value.fill_(.01)
    items, context = torch.zeros(2, 3, 259), torch.zeros(2, 524)
    changed = items.clone()
    changed[:, 2] = 1
    with torch.inference_mode():
        original, mutated = model(items, context), model(changed, context)
    assert bool((mutated[:, 0] > original[:, 0]).all())


@pytest.mark.parametrize("architecture", ["independent", "set"])
@pytest.mark.parametrize("seed", [17, 29, 43])
def test_same_architecture_seed_exact_initialization(students, architecture, seed):
    first = students.initialized_student(architecture, seed)
    second = students.initialized_student(architecture, seed)
    assert students._state_record(first.state_dict()) == students._state_record(second.state_dict())


@pytest.mark.parametrize("extra", ["labels", "t_c", "query", "pair_id"])
def test_extra_metadata_cannot_enter_inputs(students, extra):
    data = artificial_inputs()
    data[extra] = np.zeros(4)
    with pytest.raises(ValueError, match="Only items and context"):
        students.validate_inputs(data)


@pytest.mark.parametrize("field", ["items", "context"])
def test_nonfinite_and_wrong_dimension_rejected(students, field):
    data = artificial_inputs()
    data[field].flat[0] = np.nan
    with pytest.raises(ValueError):
        students.validate_inputs(data)
    data = artificial_inputs()
    data[field] = data[field][..., :-1]
    with pytest.raises(ValueError):
        students.validate_inputs(data)


def test_train_only_normalization_and_floor(students):
    train = {"items": np.ones((2, 3, 259), np.float32), "context": np.full((2, 524), 2., np.float32)}
    normalization = students.fit_normalization(train)
    assert np.array_equal(normalization["items_mean"], np.ones(259, np.float32))
    assert np.array_equal(normalization["context_mean"], np.full(524, 2., np.float32))
    assert np.array_equal(normalization["items_std"], np.full(259, 1e-4, np.float32))
    validation = {"items": np.full((1, 3, 259), 4., np.float32), "context": np.full((1, 524), 5., np.float32)}
    normalized = students.normalize_inputs(validation, normalization)
    np.testing.assert_allclose(normalized["items"], 30000., rtol=1e-6)
    np.testing.assert_allclose(normalized["context"], 30000., rtol=1e-6)
    assert np.array_equal(normalization["items_mean"], np.ones(259, np.float32))


def test_target_sign_and_all_nine_target_names(students):
    raw = artificial_labels(students)
    targets, _ = students.soft_targets(raw, artificial_ids(), students.training_label_scales(raw))
    assert set(targets) == set(students.LABELS)
    for name in students.RAW_LABELS:
        assert np.all(targets[name][:, 0] > targets[name][:, 1])
        assert np.all(targets[name][:, 1] > targets[name][:, 2])
    for value in targets.values():
        np.testing.assert_allclose(value.sum(1), 1., atol=1e-7)


def test_shuffle_correspondence_and_shared_parent_permutation(students):
    raw, ids = artificial_labels(students), artificial_ids()
    scales = students.training_label_scales(raw)
    targets, permutations = students.soft_targets(raw, ids, scales)
    for shuffled, source in students.SHUFFLE_SOURCES.items():
        assert scales[shuffled] == scales[source]
        np.testing.assert_allclose(targets[shuffled], np.take_along_axis(targets[source], permutations, 1), atol=1e-7)
    order = [3, 0, 2, 1]
    reordered = students.stable_parent_permutations([ids[index] for index in order])
    assert np.array_equal(reordered, permutations[order])
    assert np.array_equal(permutations, students.stable_parent_permutations(ids))


def test_global_train_rms_not_per_parent_temperature(students):
    raw = artificial_labels(students, rows=2)
    raw["local_feature"] = np.asarray([[0., 1., 2.], [0., 10., 20.]])
    scales = students.training_label_scales(raw)
    expected = np.sqrt(np.mean(np.asarray([[-1., 0., 1.], [-10., 0., 10.]])**2))
    assert scales["local_feature"] == expected
    targets, _ = students.soft_targets(raw, artificial_ids(2), scales)
    assert targets["local_feature"][1, 0] > targets["local_feature"][0, 0]


def test_uniform_costs_and_attention_have_uniform_targets(students):
    raw = {name: np.full((3, 3), 4.) for name in students.RAW_LABELS[:-1]}
    raw["future_attention"] = np.full((3, 3), 1 / 3)
    scales = students.training_label_scales(raw)
    assert all(value == 1e-8 for value in scales.values())
    targets, _ = students.soft_targets(raw, artificial_ids(3), scales)
    for values in targets.values():
        np.testing.assert_allclose(values, 1 / 3, atol=1e-7)


def test_parent_common_loss_offsets_do_not_change_targets(students):
    raw = artificial_labels(students)
    modified = {name: value.copy() for name, value in raw.items()}
    for name in students.RAW_LABELS[:-1]:
        modified[name] += np.asarray([1., 3., 7., 11.])[:, None]
    original_scales, modified_scales = students.training_label_scales(raw), students.training_label_scales(modified)
    assert original_scales == modified_scales
    a, _ = students.soft_targets(raw, artificial_ids(), original_scales)
    b, _ = students.soft_targets(modified, artificial_ids(), modified_scales)
    for name in students.LABELS:
        np.testing.assert_allclose(a[name], b[name], atol=1e-7)


def test_label_contract_rejects_missing_nonfinite_negative_and_bad_attention(students):
    raw = artificial_labels(students)
    with pytest.raises(ValueError):
        students.validate_labels({key: value for key, value in raw.items() if key != "local_feature"}, 4)
    for bad in (-1., np.nan):
        changed = {key: value.copy() for key, value in raw.items()}
        changed["local_feature"][0, 0] = bad
        with pytest.raises(ValueError):
            students.validate_labels(changed, 4)
    raw["future_attention"][0] = [.1, .1, .1]
    with pytest.raises(ValueError, match="probabilities"):
        students.validate_labels(raw, 4)


def test_step0_is_untrained_and_strict_tie_keeps_earliest(students):
    best, selected, changed = students.checkpoint_selection(float("inf"), None, 1., 0)
    assert (best, selected, changed) == (1., 0, True)
    assert students.checkpoint_flags(selected)["is_trained"] is False
    assert students.checkpoint_selection(best, selected, 1., 100) == (1., 0, False)
    assert students.checkpoint_selection(best, selected, 1.1, 200) == (1., 0, False)
    best, selected, changed = students.checkpoint_selection(best, selected, .9, 300)
    assert (best, selected, changed) == (.9, 300, True)
    assert students.checkpoint_flags(selected)["is_trained"] is True
    assert students.checkpoint_flags(selected)["training_updates_executed"] == 1000


@pytest.mark.parametrize("step", [-1, 1, 99, 1001, True])
def test_checkpoint_markers_reject_nonregistered_steps(students, step):
    with pytest.raises(ValueError):
        students.checkpoint_flags(step)


def test_minibatch_sequence_is_fixed_per_seed_not_label(students):
    first = students.minibatch_indices(17, 7)
    second = students.minibatch_indices(17, 7)
    assert first.shape == (1000, 128)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, students.minibatch_indices(29, 7))
    assert first.min() >= 0 and first.max() < 7


def test_cross_entropy_direction_matches_deletion_logits(students):
    target = torch.tensor([[.8, .1, .1]])
    aligned = students.soft_cross_entropy(torch.tensor([[3., 0., 0.]]), target)
    opposed = students.soft_cross_entropy(torch.tensor([[0., 3., 0.]]), target)
    assert bool((aligned < opposed).all())


def artificial_completed_summary(students):
    arms = {}
    for architecture in students.ARCHITECTURES:
        for label in students.LABELS:
            for seed in students.SEEDS:
                name = f"{architecture}_{label}_seed{seed}"
                arms[name] = {"architecture": architecture, "label": label, "seed": seed,
                    **students.checkpoint_flags(100), "checkpoint": name + ".pt",
                    "checkpoint_sha256": "a" * 64, "provenance_sha256": "b" * 64,
                    "parameters": students.architecture_plan()[architecture]["parameters"]}
    return {"status": "completed", "arms": arms, "completed_updates": 54000,
            "total_updates": 54000, "arm_count": 54, "provenance_sha256": "b" * 64,
            "step0_selected_arms": []}


def test_checkpoint_inventory_complete_and_seed_order_explicit(students):
    summary = artificial_completed_summary(students)
    summary["arms"] = dict(reversed(list(summary["arms"].items())))
    entries = students._checkpoint_metadata_entries(summary)
    assert len(entries) == 54
    for at in range(0, len(entries), 3):
        assert [entry["seed"] for entry in entries[at:at + 3]] == [17, 29, 43]
        assert len({(entry["architecture"], entry["label"]) for entry in entries[at:at + 3]}) == 1


@pytest.mark.parametrize("kind", ["missing_arm", "extra_arm", "wrong_seed", "partial_training", "path_traversal", "absolute_path", "invalid_digest", "wrong_provenance"])
def test_invalid_checkpoint_inventory_rejected_without_file_access(students, kind):
    summary = artificial_completed_summary(students)
    name = next(iter(summary["arms"]))
    record = summary["arms"][name]
    if kind == "missing_arm":
        summary["arms"].pop(name)
    elif kind == "extra_arm":
        summary["arms"]["unregistered"] = record.copy()
    elif kind == "wrong_seed":
        record["seed"] = 43
    elif kind == "partial_training":
        summary["completed_updates"] = 53999
    elif kind == "path_traversal":
        record["checkpoint"] = "../" + name + ".pt"
    elif kind == "absolute_path":
        record["checkpoint"] = "/" + name + ".pt"
    elif kind == "invalid_digest":
        record["checkpoint_sha256"] = "z" * 64
    else:
        record["provenance_sha256"] = "c" * 64
    with pytest.raises(ValueError):
        students._checkpoint_metadata_entries(summary)


def test_checkpoint_inventory_retains_step0_untrained_mark(students):
    summary = artificial_completed_summary(students)
    name = next(iter(summary["arms"]))
    summary["arms"][name].update(students.checkpoint_flags(0))
    summary["step0_selected_arms"] = [name]
    entries = students._checkpoint_metadata_entries(summary)
    assert entries[0]["selected_step"] == 0 and entries[0]["is_trained"] is False
    summary["arms"][name]["is_trained"] = True
    with pytest.raises(ValueError):
        students._checkpoint_metadata_entries(summary)
