"""Random-tensor checkpoint interleave fixture; not scientific CPU training."""
import copy
import unittest
import numpy as np
import torch
from test_anchored_memory import InputBoundaryTests
from phase1.utils import seed_all, rng_state, restore_rng
from phase1.proposal_trial import cpu_tree

class ResumeTest(InputBoundaryTests):
    def test_exact_checkpoint_interleave(self):
        initial = copy.deepcopy(self.model)
        def create():
            model = copy.deepcopy(initial).train()
            opt = torch.optim.AdamW(model.side.parameters(), lr=.001, weight_decay=.0001)
            sch = torch.optim.lr_scheduler.StepLR(opt, 1, .5)
            return model, opt, sch
        def advance(model, opt, sch):
            opt.zero_grad(set_to_none=True)
            target = torch.randn(3, 2, 16) + float(np.random.default_rng(37).random())
            (model(self.batch)-target).square().mean().backward()
            torch.nn.utils.clip_grad_norm_(model.side.parameters(), 1.)
            opt.step(); sch.step()
        seed_all(31, 1)
        continuous, co, cs = create()
        advance(continuous, co, cs)
        checkpoint = {'model': cpu_tree(continuous.state_dict()), 'optimizer': cpu_tree(co.state_dict()),
                      'scheduler': copy.deepcopy(cs.state_dict()), 'rng': rng_state()}
        advance(continuous, co, cs)
        seed_all(997, 1)
        torch.randn(43)
        restored, ro, rs = create()
        restored.load_state_dict(checkpoint['model'])
        ro.load_state_dict(checkpoint['optimizer']); rs.load_state_dict(checkpoint['scheduler'])
        restore_rng(checkpoint['rng'])
        advance(restored, ro, rs)
        for key, value in continuous.state_dict().items():
            self.assertTrue(torch.equal(value, restored.state_dict()[key]), key)
        self.assertEqual(cs.state_dict(), rs.state_dict())
        for key, value in co.state_dict()['state'].items():
            for field, tensor in value.items():
                self.assertTrue(torch.equal(tensor, ro.state_dict()['state'][key][field]))

if __name__ == '__main__':
    unittest.main()
