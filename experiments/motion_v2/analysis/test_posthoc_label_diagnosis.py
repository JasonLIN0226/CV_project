"""事后探索性分析，未预登记：synthetic checks, no experimental data access."""
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import unittest

import numpy as np


SPEC = importlib.util.spec_from_file_location("diagnosis", Path(__file__).with_name("posthoc_label_diagnosis.py"))
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)


class PosthocArithmeticTests(unittest.TestCase):
    def test_types_and_history_boundary(self):
        visible = np.ones(21, dtype=bool)
        visible[0] = visible[7] = False
        s = {"visible": visible, "query": 20, "t_c": 5, "end": 10}
        self.assertEqual([D.frame_type(s, i) for i in [0, 2, 5, 7, 10]], ["O", "S", "P", "O", "E"])
        with self.assertRaises(ValueError):
            D.frame_type(s, 17)

    def test_rank_is_timestamp_rank_not_slot(self):
        keys = D.candidate_keys(["O", "P", "E"], [9, 3, 6])
        self.assertEqual([k[1] for k in keys], [2, 0, 1])
        self.assertTrue(all(k[2] == (1, 0, 1, 1) for k in keys))

    def test_gap_sign_and_tie_rule(self):
        rows = [["O", "P", "E"], ["P", "E", "S"]]
        value = D.two_new_alignment(np.array([[.1, .5, .8], [.1, .2, .4]]), rows)
        self.assertEqual(value["argmin_deletes_nonnew_rate"], .5)
        self.assertAlmostEqual(value["median_best_competing_new_drop_minus_nonnew_drop"], .05)
        tied = D.two_new_alignment(np.zeros((2, 3)), rows)
        self.assertEqual(tied["argmin_deletes_nonnew_rate"], .5)
        self.assertAlmostEqual(tied["tie_averaged_deletes_nonnew_rate"], 1/3)

    def test_train_reference_normalizes_and_never_fits_validation(self):
        a,b,c,d,e = [("O", i, (3,0,0,0)) for i in range(5)]
        train_keys = [[a,b,c], [a,d,e]]
        train = np.array([[.8,.1,.1], [.2,.3,.5]])
        q = np.array([[.2,.3,.5]])
        value = D.type_reference(train_keys, [[a,b,c]], train, q)
        expected = -np.sum(q * np.log(np.array([[.5,.1,.1]])/.7))
        self.assertAlmostEqual(value["validation_soft_ce"], expected)
        changed = D.type_reference(train_keys, [[a,b,c]], train, np.array([[.9,.05,.05]]))
        self.assertEqual(value["key_means"], changed["key_means"])
        unseen = D.type_reference(train_keys, [[("S",i,(0,3,0,0)) for i in range(3)]], train, q)
        self.assertEqual(unseen["unseen_validation_candidate_keys"], 3)
        self.assertAlmostEqual(unseen["validation_soft_ce"], math.log(3))

    def test_targets_match_frozen_training_arithmetic(self):
        # Execute only the frozen source's pure NumPy label functions, never
        # import torch, a student, or any training/inference entry point.
        path = Path(__file__).parents[1] / "step34_students.py"
        parsed = ast.parse(path.read_text())
        constants = {"RAW_LABELS", "SHUFFLE_SOURCES", "LABELS", "LABEL_SCALE_FLOOR", "SHUFFLE_SEED"}
        functions = {"validate_labels", "_validate_ids", "training_label_scales", "stable_parent_permutations", "soft_targets"}
        nodes = [n for n in parsed.body if (isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id in constants for t in n.targets)) or (isinstance(n, ast.FunctionDef) and n.name in functions)]
        namespace = {"np":np,"math":math,"hashlib":hashlib,"json":json}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),"exec"), namespace)
        rows = np.array([[.1,.3,.8],[.4,.1,.6],[.1,.1,.1]])
        raw = {k: rows*(i+1) for i,k in enumerate(D.LABELS[:-1])}
        raw["future_attention"] = rows/rows.sum(1,keepdims=True)
        scales = namespace["training_label_scales"](raw)
        expected,_ = namespace["soft_targets"](raw,["a","b","c"],scales)
        actual, fitted = D.targets(raw)
        for label in D.LABELS:
            self.assertEqual(fitted[label], scales[label])
            np.testing.assert_array_equal(actual[label], expected[label])
        shifted = {k:v*3 for k,v in raw.items()}
        heldout,_ = D.targets(shifted,fitted)
        logits = -(shifted["local_feature"]-shifted["local_feature"].mean(1,keepdims=True))/fitted["local_feature"]
        logits -= logits.max(1,keepdims=True)
        expected_heldout = np.exp(logits)/np.exp(logits).sum(1,keepdims=True)
        np.testing.assert_array_equal(heldout["local_feature"], expected_heldout.astype(np.float32))


if __name__ == "__main__":
    unittest.main(verbosity=2)
