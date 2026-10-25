"""Regression checks for the proper conditional-mean reader objective."""
import torch
from experiments.phase2.run_terminal_structures import group_loss, fifo_parent_trace


def test_balanced_missing_xor_has_zero_mean_gradient():
    shared_prediction=torch.tensor(0.,requires_grad=True)
    targets=torch.tensor([-1.,1.]).reshape(2,1,1)
    group_loss(shared_prediction.expand_as(targets),targets).backward()
    assert shared_prediction.grad.item()==0.


def test_complete_and_equal_xor_targets_have_no_repulsion():
    targets=torch.tensor([[-1.,1.],[-1.,1.],[1.,-1.],[1.,-1.]])
    prediction=targets.clone().requires_grad_(True)
    group_loss(prediction,targets).backward()
    assert torch.equal(prediction.grad,torch.zeros_like(targets))


def test_actual_fifo_prefix_forms_all_three_candidates():
    for q in range(12,19):
        trace=fifo_parent_trace(q)
        assert trace[-1]['before']==[q-6,q-5]
        assert trace[-1]['incoming']==q-4
        assert trace[-1]['after']==[q-6,q-5,q-4]
        assert trace[-1]['prefix_deleted'] is None
        assert all(len(t['after'])<=2 for t in trace[:-1])
