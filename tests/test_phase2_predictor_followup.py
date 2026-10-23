"""Checks for train-only projection interventions and paired loss reporting."""
import unittest
import numpy as np
import torch

from experiments.phase2.train_predictor_followup import (
    ARMS, PROTOCOL, Samples, loss_components, projection_variant,
)


class FollowupTests(unittest.TestCase):
    def basis(self):
        return {"input_mean": np.zeros(3, np.float32), "projection": np.diag([2., 4., 8.]).astype(np.float32),
                "target_mean": np.ones(3, np.float32), "output_basis": np.eye(3, dtype=np.float32),
                "train_ids": np.array(["only_train"])}

    def test_supported_mask_uses_prefix_support_and_preserves_other_basis(self):
        basis = self.basis()
        train = [np.array([[1., 0., 2.], [-1., 0., -2.]], np.float32)]
        altered, proof = projection_variant(basis, train, "train_support_mask")
        self.assertEqual(proof["masked_dimensions"], 1)
        self.assertTrue(np.array_equal(altered["projection"], np.diag([2., 0., 8.])))
        self.assertTrue(np.array_equal(basis["projection"], np.diag([2., 4., 8.])))
        for key in ("input_mean", "target_mean", "output_basis", "train_ids"):
            self.assertTrue(np.array_equal(altered[key], basis[key]))
        # A novel validation coordinate cannot re-enable a masked train-null direction.
        self.assertEqual(float(np.array([0., 10., 0.]) @ altered["projection"][:, 1]), 0.)

    def test_unit_columns_and_original_are_distinct_bounded_interventions(self):
        basis = self.basis()
        train = [np.ones((3, 3), np.float32)]
        unchanged, _ = projection_variant(basis, train, "original_projection")
        unit, _ = projection_variant(basis, train, "unit_columns")
        self.assertTrue(np.array_equal(unchanged["projection"], basis["projection"]))
        np.testing.assert_allclose(np.linalg.norm(unit["projection"], axis=0), 1.)
        with self.assertRaises(ValueError):
            projection_variant(basis, train, "validation_fitted")

    def test_paired_components_distinguish_common_and_differential_error(self):
        target = torch.tensor([[[1., 2.]], [[-1., -2.]]])
        common_error = torch.tensor([[[3., 3.]], [[3., 3.]]])
        r = loss_components(target + common_error, target)
        self.assertAlmostEqual(float(r["common_mse"]), 9.)
        self.assertEqual(float(r["difference_mse"]), 0.)
        pred = torch.zeros_like(target)
        r = loss_components(pred, target)
        self.assertEqual(float(r["common_mse"]), 0.)
        self.assertAlmostEqual(float(r["mse"]), float(r["common_mse"] + r["difference_mse"]))
        self.assertAlmostEqual(float(r["pair_mse"]), 4 * float(r["difference_mse"]))
        self.assertAlmostEqual(float(r["combined_loss"]),
                               float(r["common_mse"] + 9 * r["difference_mse"] + .5 * r["rank_loss"]), places=5)
        with self.assertRaises(ValueError):
            loss_components(pred[:1], target[:1])

    def test_predeclared_scope_has_exactly_three_arms_and_epoch_zero_selection(self):
        self.assertEqual(len(ARMS), 3)
        self.assertEqual(PROTOCOL["epochs"], 40)
        self.assertIn("epoch zero", PROTOCOL["selection"])
        self.assertFalse(PROTOCOL["test_arrays_permitted"])
        self.assertFalse(PROTOCOL["adopt_best_arm"])

    def test_test_record_refused_before_arrays_opened(self):
        class ForbiddenSnapshot:
            records = [{"task": "gate", "split": "test_id"}]
            def features(self, rec):
                raise AssertionError("Test arrays must not open")
        with self.assertRaises(ValueError):
            Samples(ForbiddenSnapshot(), "gate", self.basis())


if __name__ == "__main__":
    unittest.main()
