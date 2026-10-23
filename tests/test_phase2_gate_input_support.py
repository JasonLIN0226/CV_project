import numpy as np
import pytest
import torch

from experiments.phase2.diagnose_gate_input_support import (
    masked_batch, masked_readout, training_support,
)
from experiments.phase2.diagnose_training_objective import inactive_columns


def test_support_uses_existing_train_second_moment_rule_not_variance():
    x = np.array([[2., .00001, .02], [2., -.00001, -.02]], np.float32)
    mask, energy = training_support(x)
    assert np.array_equal(mask, inactive_columns(x))
    assert mask.tolist() == [False, True, False]
    assert np.array_equal(energy, np.mean(x ** 2, axis=0))
    with pytest.raises(ValueError):
        training_support(np.empty((0, 3), np.float32))


def test_mask_only_changes_projected_inputs_without_mutation():
    batch = {"projected_z": torch.arange(12.).reshape(1, 4, 3),
             "projected_current": torch.tensor([[7., 8., 9.]]),
             "current": torch.tensor([[1., 2., 3., 4.]]),
             "z": torch.ones(1, 4, 4), "actions": torch.ones(1, 6, 2)}
    before = {key: value.clone() for key, value in batch.items()}
    result = masked_batch(batch, np.array([False, True, False]))
    assert torch.count_nonzero(result["projected_z"][..., 1]) == 0
    assert torch.count_nonzero(result["projected_current"][..., 1]) == 0
    assert result["current"] is batch["current"]
    assert result["z"] is batch["z"] and result["actions"] is batch["actions"]
    for key in before:
        assert torch.equal(before[key], batch[key])
    assert torch.equal(result["projected_z"][..., [0, 2]], batch["projected_z"][..., [0, 2]])


def test_probe_intervention_freezes_standardization_and_supported_weights():
    model = {"mean": np.array([10., 20.]), "std": np.array([2., 1e-6]),
             "target_mean": np.array([.5]), "weight": np.array([[.2], [.8]])}
    result = masked_readout(model, np.array([False, True]))
    assert result["weight"].tolist() == [[.2], [0.]]
    assert model["weight"].tolist() == [[.2], [.8]]
    for key in ("mean", "std", "target_mean"):
        assert np.array_equal(result[key], model[key])
