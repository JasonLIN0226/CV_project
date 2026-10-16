"""Synthetic helper checks only; these do not support the conditional candidate."""
import copy
import random

import numpy as np
import pytest
import torch
from torch import nn

from phase1.common_anchor import (COMMON_ANCHOR_WEIGHT, HISTORY_CONDITIONS,
                                 TeacherForwardAccounting, common_anchor_loss,
                                 recent_only_teacher_batch, recent_teacher_prediction)
from phase1.model import DynamicsPredictor
from phase1.training import regression_loss


def rows():
    return [{"id": f"e{i}", "sibling": f"e{i ^ 1}", "pair": f"p{i // 2}", "task": "gate", "split": "train"}
            for i in range(4)]


def inputs(episodes=4):
    # Two recent frames followed by one legal historical frame; sibling recent
    # inputs/actions match while historical visual evidence differs.
    z = torch.arange(episodes*3*2, dtype=torch.float32).reshape(episodes, 3, 2)
    for i in range(0, episodes-1, 2):
        z[i+1, :2] = z[i, :2]
    return {"z": z, "age": torch.tensor([[1., 0., 4.]]).repeat(episodes, 1),
            "type": torch.tensor([[0, 0, 1]]).repeat(episodes, 1),
            "past_actions": torch.zeros(episodes, 3, 2), "actions": torch.zeros(episodes, 6, 2),
            "current": z[:, 1].clone(), "mask": torch.ones(episodes, 3, dtype=torch.bool)}


@pytest.mark.parametrize("condition", sorted(HISTORY_CONDITIONS))
def test_common_loss_has_fixed_weight_equal_sibling_gradients_and_no_teacher_gradient(condition):
    prediction = torch.tensor([2., 4., 5., 9.]).reshape(4, 1, 1).requires_grad_()
    teacher = torch.tensor([1., 1., 2., 2.]).reshape(4, 1, 1).requires_grad_()
    loss = common_anchor_loss(prediction, teacher, rows(), condition=condition)
    assert COMMON_ANCHOR_WEIGHT == 1.0 and loss.item() == pytest.approx((2**2+5**2)/2)
    loss.backward()
    assert teacher.grad is None
    assert torch.equal(prediction.grad[::2], prediction.grad[1::2])
    assert torch.equal(prediction.grad.flatten(), torch.tensor([1., 1., 2.5, 2.5]))
    # Equal sibling gradients directly affect common mode, not their difference.
    centered = prediction.detach().clone()
    centered[::2] += 3
    centered[1::2] -= 3
    assert torch.equal(common_anchor_loss(centered, teacher, rows(), condition=condition), loss.detach())


def test_original_supervised_loss_is_unchanged_by_additive_helper():
    prediction = torch.randn(4, 2, 3, requires_grad=True)
    target = torch.randn(4, 2, 3)
    original, mse, pair = regression_loss(prediction, target, 2., .5)
    teacher = torch.tensor([1., 1., 2., 2.]).reshape(4, 1, 1).expand_as(prediction)
    anchor = common_anchor_loss(prediction, teacher, rows(), condition="full_history")
    combined = original + anchor
    unchanged = regression_loss(prediction, target, 2., .5)
    assert torch.equal(original, unchanged[0]) and torch.equal(mse, unchanged[1]) and torch.equal(pair, unchanged[2])
    assert torch.allclose(combined-original, anchor)


@pytest.mark.parametrize("invalid", ["single", "validation", "motion", "wrong_sibling", "repeated_pair", "order"])
def test_common_loss_rejects_invalid_pair_or_training_boundary(invalid):
    meta, prediction, teacher = rows(), torch.zeros(4, 2, 3), torch.zeros(4, 2, 3)
    if invalid == "single":
        meta, prediction, teacher = meta[:1], prediction[:1], teacher[:1]
    elif invalid == "validation":
        meta[0]["split"] = "validation"
    elif invalid == "motion":
        meta[0]["task"] = "motion"
    elif invalid == "wrong_sibling":
        meta[1]["sibling"] = "other"
    elif invalid == "repeated_pair":
        meta[2]["pair"] = meta[3]["pair"] = meta[0]["pair"]
    else:
        meta = [meta[i] for i in [0, 2, 1, 3]]
    with pytest.raises(ValueError):
        common_anchor_loss(prediction, teacher, meta, condition="full_history")


@pytest.mark.parametrize("condition", ["recent_only", "wrong_evidence", "unknown"])
def test_common_loss_rejects_other_conditions(condition):
    with pytest.raises(ValueError, match="history condition"):
        common_anchor_loss(torch.zeros(4, 2, 3), torch.zeros(4, 2, 3), rows(), condition=condition)


def test_common_loss_rejects_different_recent_context_and_nonfinite_tensors():
    prediction, teacher = torch.zeros(4, 2, 3), torch.zeros(4, 2, 3)
    teacher[1, 0, 0] = 1
    with pytest.raises(ValueError, match="identical sibling"):
        common_anchor_loss(prediction, teacher, rows(), condition="full_history")
    teacher.zero_()
    prediction[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        common_anchor_loss(prediction, teacher, rows(), condition="full_history")


def test_recent_transform_removes_history_and_supports_one_trajectory_without_pair_or_future():
    source = inputs(1)
    source["projected_z"] = source["z"] * .5
    source["projected_current"] = source["current"] * .5
    recent = recent_only_teacher_batch(source)
    assert recent["z"].shape == (1, 2, 2) and recent["type"].eq(0).all() and recent["mask"].all()
    assert torch.equal(recent["z"], source["z"][:, :2])
    assert torch.equal(recent["projected_z"], source["projected_z"][:, :2])
    changed = copy.deepcopy(source)
    changed["z"][:, 2] += 1000
    changed["projected_z"][:, 2] += 2000
    assert all(torch.equal(value, recent_only_teacher_batch(changed)[key]) for key, value in recent.items())
    model = DynamicsPredictor(2, {"hidden": 8, "dropout": .25}, [3, 6]).eval()
    # Ordinary inference is unchanged and requires neither siblings nor this loss.
    with torch.no_grad():
        ordinary = model(recent)
    assert torch.equal(ordinary, recent_teacher_prediction(model, source))


@pytest.mark.parametrize("invalid", ["target", "hidden_state", "future_rgb", "future_visual_time", "noncontiguous", "current"])
def test_recent_transform_rejects_future_or_non_predictor_inputs(invalid):
    source = inputs()
    if invalid in {"target", "hidden_state", "future_rgb"}:
        source[invalid] = torch.zeros(4, 1)
    elif invalid == "future_visual_time":
        source["age"][0, 2] = -3
    elif invalid == "noncontiguous":
        source["age"][0, 0] = 2
    else:
        source["current"][0, 0] += 1
    with pytest.raises(ValueError):
        recent_only_teacher_batch(source)


class CountingPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.))
        self.dropout = nn.Dropout(.8)
        self.calls = []
        self.fail = False

    def forward(self, batch):
        self.calls.append((self.training, self.dropout.training, torch.is_grad_enabled(), batch["z"].shape[1]))
        # Deliberately consume all global RNGs to check restoration beyond dropout.
        random.random()
        np.random.random()
        torch.rand(1)
        if self.fail:
            raise RuntimeError("synthetic teacher failure")
        return self.dropout(batch["current"][:, None]) * self.weight


def rng_snapshot():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def assert_rng_equal(a, b):
    assert a[0] == b[0]
    assert a[1][0] == b[1][0] and np.array_equal(a[1][1], b[1][1]) and a[1][2:] == b[1][2:]
    assert torch.equal(a[2], b[2])


def test_teacher_current_weights_no_grad_eval_rng_restoration_and_forward_accounting():
    model, source, accounting = CountingPredictor().train(), inputs(), TeacherForwardAccounting()
    model.dropout.eval()  # Preserve a deliberately mixed original module mode.
    modes = [module.training for module in model.modules()]
    before = rng_snapshot()
    first = recent_teacher_prediction(model, source, accounting=accounting)
    assert not first.requires_grad and first.grad_fn is None
    assert_rng_equal(before, rng_snapshot())
    assert [module.training for module in model.modules()] == modes
    with torch.no_grad():
        model.weight.fill_(3)
    second = recent_teacher_prediction(model, source, accounting=accounting)
    assert torch.equal(second, 3*first)  # Recomputed now, never cached across steps.
    assert model.calls == [(False, False, False, 2), (False, False, False, 2)]
    assert accounting.extra_forward_calls == 2 and accounting.extra_forward_examples == 8
    assert_rng_equal(before, rng_snapshot())
    model.fail = True
    with pytest.raises(RuntimeError, match="synthetic teacher failure"):
        recent_teacher_prediction(model, source, accounting=accounting)
    assert_rng_equal(before, rng_snapshot())
    assert [module.training for module in model.modules()] == modes
    assert accounting.extra_forward_calls == 3  # Attempted failed forward also costs compute.


def test_common_loss_reads_only_pair_metadata_never_future_or_state_payload():
    class BoundaryRow(dict):
        def __getitem__(self, key):
            assert key in {"id", "sibling", "pair", "task", "split"}
            return super().__getitem__(key)
    meta = [BoundaryRow(row, target=object(), future_rgb=object(), state=object()) for row in rows()]
    prediction, teacher = torch.ones(4, 2, 3, requires_grad=True), torch.zeros(4, 2, 3)
    assert common_anchor_loss(prediction, teacher, meta, condition="full_history").item() == 1
