"""Distinct checkpoint binding rejects provenance/transform substitution."""
import copy
import unittest
import numpy as np
import torch

from experiments.phase2.run_gate_followup_phase2 import verify_selected_binding


class GateBindingTests(unittest.TestCase):
    def fixture(self):
        original = {"input_mean": np.zeros(3, np.float32), "projection": np.diag([2., 3., 4.]).astype(np.float32),
            "target_mean": np.ones(3, np.float32), "output_basis": np.eye(3, dtype=np.float32), "train_ids": np.array(["train"])}
        basis = {k: v.copy() for k, v in original.items()}
        basis["projection"] = np.eye(3, dtype=np.float32)
        config = {"hidden": 4, "dropout": 0.}
        checkpoint = {"task": "gate", "arm": "unit_columns", "epoch": 35, "state_kind": "selected",
            "checkpoint_kind": "exploratory_fixed_projection_followup", "protocol_sha256": "a" * 64,
            "model_config": config, "horizons": (3, 6), "feature_dimension": 3,
            "model": {k: torch.from_numpy(basis[k].copy()) for k in ("input_mean", "projection", "target_mean", "output_basis")}}
        return checkpoint, basis, original, "a" * 64, config

    def test_accepts_exact_new_binding(self):
        verify_selected_binding(*self.fixture())

    def test_refuses_different_checkpoint_or_protocol(self):
        for key, value in (("task", "motion"), ("epoch", 1), ("arm", "original_projection"), ("protocol_sha256", "b" * 64)):
            args = list(self.fixture()); args[0][key] = value
            with self.assertRaises(ValueError):
                verify_selected_binding(*args)

    def test_refuses_silent_basis_or_weight_buffer_changes(self):
        for changed in ("projection", "target_mean", "output_basis"):
            args = list(self.fixture()); args[1][changed].flat[0] += .1
            with self.assertRaises(ValueError):
                verify_selected_binding(*args)
        args = list(self.fixture()); args[0]["model"]["projection"][0, 0] += 1
        with self.assertRaises(ValueError):
            verify_selected_binding(*args)

    def test_training_membership_must_be_preserved(self):
        args = list(self.fixture()); args[1]["train_ids"] = np.array(["validation"])
        with self.assertRaises(ValueError):
            verify_selected_binding(*args)


if __name__ == "__main__":
    unittest.main()
