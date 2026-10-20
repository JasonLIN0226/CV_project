"""Synthetic serialization fixtures only; no research data or models."""
import copy
import random
import numpy as np
import pytest
import torch
from phase1.utils import rng_state
from resume_checkpoint import restore_exact, equal


@pytest.mark.parametrize('kind', ['reference', 'candidate'])
def test_resume_matches_uninterrupted_optimizer_schedule_and_random_stream(kind):
    torch.set_num_threads(1)
    random.seed(7); np.random.seed(7); torch.manual_seed(7)
    def construct():
        m = torch.nn.Linear(3, 2)
        o = torch.optim.AdamW(m.parameters(), lr=.001)
        s = (torch.optim.lr_scheduler.StepLR(o, 2, .5) if kind == 'reference'
             else torch.optim.lr_scheduler.LambdaLR(o, lambda e: min(1., (e+1)/5)))
        return m, o, s
    def update(m, o, s):
        o.zero_grad()
        x = torch.randn(4, 3) * (random.random() + np.random.random())
        m(x).square().mean().backward(); o.step(); s.step()
    model, opt, sched = construct()
    for _ in range(3): update(model, opt, sched)
    checkpoint = copy.deepcopy(dict(model=model.state_dict(), optimizer=opt.state_dict(),
                                   scheduler=sched.state_dict(), rng=rng_state()))
    for _ in range(2): update(model, opt, sched)
    expected = copy.deepcopy(dict(model=model.state_dict(), optimizer=opt.state_dict(),
                                 scheduler=sched.state_dict(), rng=rng_state()))
    resumed, new_opt, new_sched = construct()
    restore_exact(resumed, new_opt, new_sched, checkpoint)
    for _ in range(2): update(resumed, new_opt, new_sched)
    actual = dict(model=resumed.state_dict(), optimizer=new_opt.state_dict(),
                  scheduler=new_sched.state_dict(), rng=rng_state())
    assert equal(expected, actual)
