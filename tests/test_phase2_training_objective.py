"""Algebraic and intervention invariants for the reproducible cause diagnostic."""
import copy

import numpy as np
import pytest
import torch

from experiments.phase2.diagnose_training_objective import (
    components, cosine, gradient_vector, inactive_columns, run, unit_column_basis,
)
from phase1.training import regression_loss


def test_common_difference_decomposition_and_original_objective():
    torch.manual_seed(17)
    prediction = torch.randn(8, 2, 11, dtype=torch.float64, requires_grad=True)
    target = torch.randn_like(prediction)
    value = components(prediction, target)
    assert torch.allclose(value["mse"], value["common_mse"] + value["difference_mse"], atol=1e-14, rtol=1e-14)
    assert torch.allclose(value["pair_mse"], 4 * value["difference_mse"])
    expected, mse, pair = regression_loss(prediction, target, 2., .5)
    assert torch.equal(value["objective"], expected)
    assert torch.equal(value["mse"], mse)
    assert torch.equal(value["pair_mse"], pair)


def test_pair_rank_reflects_sibling_assignment_not_global_mean():
    target = torch.tensor([[[2., 0.]], [[-2., 0.]], [[0., 2.]], [[0., -2.]]])
    correct = components(target.clone(), target)
    swapped = components(target.reshape(2, 2, 1, 2).flip(1).reshape_as(target), target)
    assert correct["pair_choice_accuracy"] == 1 and swapped["pair_choice_accuracy"] == 0
    assert correct["rank"] < swapped["rank"]
    assert swapped["common_mse"] == 0 and swapped["difference_mse"] > 0


def test_gradient_decomposition_and_cosine_with_unused_parameter():
    class Probe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(4, 2, 3))
            self.unused = torch.nn.Parameter(torch.ones(7))

    model = Probe()
    target = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3) / 10
    value = components(model.weight, target)
    gm, gp, gr = [gradient_vector(value[key], model) for key in ("mse", "pair_mse", "rank")]
    objective = gradient_vector(value["objective"], model)
    assert torch.allclose(objective, gm + 2 * gp + .5 * gr, atol=1e-6)
    assert torch.equal(objective[-7:], torch.zeros(7))
    assert cosine(gm, gm) == pytest.approx(1.)
    assert cosine(gm, -gm) == pytest.approx(-1.)
    assert cosine(gm, torch.zeros_like(gm)) == 0


def test_whitening_removal_preserves_directions_and_other_basis_buffers():
    original = {"projection": np.array([[2., 0., 0.], [0., -5., 0.], [0., 0., .1]], np.float32),
                "input_mean": np.array([1., 2., 3.], np.float32),
                "output_basis": np.eye(3, dtype=np.float32), "target_mean": np.array([4., 5., 6.], np.float32)}
    before = copy.deepcopy(original)
    transformed = unit_column_basis(original)
    assert np.array_equal(transformed["projection"], np.diag([1., -1., 1.]).astype(np.float32))
    for key in original:
        assert np.array_equal(original[key], before[key])
        if key != "projection":
            assert np.array_equal(transformed[key], before[key])
    transformed["input_mean"][:] = 100
    assert np.array_equal(original["input_mean"], before["input_mean"])


@pytest.mark.parametrize("projection", [np.zeros((3, 2), np.float32), np.full((3, 2), np.nan, np.float32)])
def test_invalid_projection_rejected(projection):
    with pytest.raises(ValueError, match="nonzero"):
        unit_column_basis({"projection": projection})


def test_inactive_mask_is_train_only_and_uses_second_moment():
    # Constant nonzero train columns are active by second moment even if their
    # variance is zero. A validation outlier cannot choose this mask.
    train = np.array([[0., 2., .001], [0., 2., -.001]], np.float32)
    mask = inactive_columns(train)
    assert mask.tolist() == [True, False, True]
    validation = np.array([[1e5, 0., 1e5]], np.float32)
    assert inactive_columns(train).tolist() == [True, False, True]
    assert inactive_columns(np.r_[train, validation]).tolist() != mask.tolist()
    with pytest.raises(ValueError, match="positive"):
        inactive_columns(train, threshold=0)


def test_pair_shape_and_output_overwrite_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="adjacent-pair"):
        components(torch.zeros(3, 2, 4), torch.zeros(3, 2, 4))
    with pytest.raises(ValueError, match="adjacent-pair"):
        components(torch.zeros(2, 2, 4), torch.zeros(2, 2, 5))
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "untouched"
    sentinel.write_text("preserve")
    with pytest.raises(ValueError, match="overwrite"):
        run(None, output)
    assert sentinel.read_text() == "preserve"
