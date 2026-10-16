"""Synthetic software fixtures only; no research data or model training."""
import copy
import unittest
import numpy as np
import torch

from phase1.model import DynamicsPredictor
from phase1.anchored_memory import AnchoredMemoryPredictor
from phase1.common_anchor import recent_only_teacher_batch


class InputBoundaryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(1)
        d, rank, n = 16, 4, 3
        self.basis = {'input_mean': np.zeros(d, np.float32),
                      'projection': np.eye(d, rank, dtype=np.float32),
                      'target_mean': np.zeros(d, np.float32),
                      'output_basis': np.eye(d, rank, dtype=np.float32)}
        reference = DynamicsPredictor(d, {'hidden': 16, 'dropout': 0.}, [3, 6], self.basis)
        self.model = AnchoredMemoryPredictor(reference, self.basis, hidden=32, layers=1, heads=4)
        z = torch.randn(n, 6, d)
        self.batch = {'z': z, 'age': torch.tensor([[3., 2., 1., 0., 8., 5.]]).repeat(n, 1),
                      'type': torch.tensor([[0, 0, 0, 0, 1, 1]]).repeat(n, 1),
                      'past_actions': torch.randn(n, 6, 2), 'actions': torch.randn(n, 6, 2),
                      'current': z[:, 3].clone(), 'mask': torch.ones(n, 6, dtype=torch.bool)}

    def make_nonzero_side(self):
        with torch.no_grad():
            self.model.side.output.weight.normal_(std=.1)

    def test_zero_initialization_preserves_reference_with_any_history(self):
        self.model.eval()
        with torch.no_grad():
            expected = self.model.reference(recent_only_teacher_batch(self.batch))
            self.assertTrue(torch.equal(self.model(self.batch), expected))

    def test_reference_is_frozen_and_recent_anchor_survives_update(self):
        before = copy.deepcopy(self.model.reference.state_dict())
        self.model.train()
        opt = torch.optim.AdamW(self.model.side.parameters(), lr=.001)
        loss = (self.model(self.batch) - torch.randn(3, 2, 16)).square().mean()
        loss.backward()
        self.assertTrue(all(p.grad is None for p in self.model.reference.parameters()))
        self.assertGreater(float(self.model.side.output.weight.grad.abs().sum()), 0.)
        opt.step()
        self.assertTrue(all(torch.equal(before[k], v) for k, v in self.model.reference.state_dict().items()))
        recent = recent_only_teacher_batch(self.batch)
        self.assertTrue(torch.equal(self.model(recent), self.model.reference(recent)))
        self.assertFalse(self.model.reference.training)

    def test_h3_cannot_read_h4_to_h6_controls(self):
        self.make_nonzero_side()
        changed = copy.deepcopy(self.batch)
        changed['actions'][:, 3:] += 1000.
        a, b = self.model(self.batch), self.model(changed)
        self.assertTrue(torch.equal(a[:, 0], b[:, 0]))
        self.assertFalse(torch.allclose(a[:, 1], b[:, 1]))

    def test_padding_poison_has_no_effect(self):
        self.make_nonzero_side()
        changed = copy.deepcopy(self.batch)
        for key in ['z', 'age', 'past_actions']:
            shape = (len(changed[key]), 2) + changed[key].shape[2:]
            changed[key] = torch.cat((changed[key], torch.full(shape, float('nan'))), 1)
        changed['type'] = torch.cat((changed['type'], torch.full((3, 2), 99)), 1)
        changed['mask'] = torch.cat((changed['mask'], torch.zeros(3, 2, dtype=torch.bool)), 1)
        self.assertTrue(torch.allclose(self.model(self.batch), self.model(changed), atol=1e-6, rtol=1e-6))

    def test_inference_does_not_require_siblings_or_labels(self):
        self.make_nonzero_side()
        full = self.model(self.batch)
        single = self.model({k: v[1:2] for k, v in self.batch.items()})
        self.assertTrue(torch.allclose(single, full[1:2], atol=1e-6, rtol=1e-6))
        illegal = dict(self.batch, target=torch.zeros(3, 2, 16))
        with self.assertRaises(ValueError):
            self.model(illegal)

    def test_future_visual_token_rejected(self):
        self.batch['age'][0, 4] = -1
        with self.assertRaises(ValueError):
            self.model(self.batch)


if __name__ == '__main__':
    unittest.main()
