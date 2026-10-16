"""Artificial arithmetic and rejection checks; no real FAR scores are inferred."""
import unittest
import numpy as np
from aggregate_reference import interval, validate_record, window_score


def record(steps=30):
    frac = np.arange(1, steps + 1) / steps
    return {"n_steps": np.array(steps), "video_id": np.array("test/ABCA/fixture"),
            "psnr": 10 + 4 * frac, "lpips": 1 - frac, "dreamsim": .5 * (1 - frac),
            "frac": frac, "frame_idx": np.arange(steps) * 4 + 104,
            "sec_from_start": np.arange(1, steps + 1) * .2,
            "sec_to_end": np.arange(steps - 1, -1, -1) * .2,
            "dist_to_closure": np.arange(steps - 1, -1, -1).astype(float)}


class AggregationNumeric(unittest.TestCase):
    def test_producer_timeline_and_pair_are_valid(self):
        a = record()
        axis = validate_record(a, 30, "test/ABCA/fixture")
        validate_record(record(), 30, "test/ABCA/fixture", axis)

    def test_exact_full_score(self):
        self.assertAlmostEqual(window_score(record(), "psnr", "FULL_TRAJECTORY"), 10 + 4 * 31 / 60)

    def test_author_quarter_interpolation(self):
        grid = np.linspace(.02, 1, 25)
        self.assertAlmostEqual(window_score(record(), "psnr", "FINAL_QUARTER"), (10 + 4 * grid[grid >= .75]).mean())

    def test_last_five_seconds(self):
        self.assertAlmostEqual(window_score(record(), "psnr", "LAST_5_SECONDS_ELIGIBLE_ONLY"), 10 + 4 * 18 / 30)
        with self.assertRaises(RuntimeError):
            window_score(record(24), "psnr", "LAST_5_SECONDS_ELIGIBLE_ONLY")

    def test_nonfinite_metric_rejected(self):
        a = record(); a["lpips"][3] = np.nan
        with self.assertRaises(RuntimeError): validate_record(a, 30, "test/ABCA/fixture")

    def test_phase_shift_rejected(self):
        a = record(); a["frac"] = np.arange(30) / 30
        with self.assertRaises(RuntimeError): validate_record(a, 30, "test/ABCA/fixture")

    def test_different_paired_target_rejected(self):
        axis = validate_record(record(), 30, "test/ABCA/fixture")
        a = record(); a["frame_idx"] += 1
        with self.assertRaises(RuntimeError): validate_record(a, 30, "test/ABCA/fixture", axis)

    def test_incomplete_timeline_rejected(self):
        a = record(); a["sec_to_end"] = a["sec_to_end"][:-1]
        with self.assertRaises(RuntimeError): validate_record(a, 30, "test/ABCA/fixture")

    def test_constant_bootstrap_and_determinism(self):
        self.assertEqual(interval(np.ones(10) * 2), [2, 2])
        self.assertEqual(interval(np.arange(8)), interval(np.arange(8)))


if __name__ == "__main__":
    unittest.main()
