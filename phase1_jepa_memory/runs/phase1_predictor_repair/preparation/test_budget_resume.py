"""Small software fixtures for serialization; no scientific data or HPC training."""
import copy
import io
import numpy as np
import torch


def fixture():
    torch.manual_seed(17)
    model=torch.nn.Linear(3,2)
    opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda n:(n+1)/5 if n<5 else 1.)
    return model,opt,scheduler


def step(model,opt,scheduler):
    x=torch.randn(4,3); y=torch.randn(4,2)
    opt.zero_grad(set_to_none=True)
    ((model(x)-y)**2).mean().backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    opt.step();scheduler.step()


def test_exact_epoch_boundary_state_without_reexecuting_epoch1():
    model,opt,schedule=fixture();step(model,opt,schedule)
    origin={'model':copy.deepcopy(model.state_dict()),'optimizer':copy.deepcopy(opt.state_dict()),
            'scheduler':schedule.state_dict(),'rng':torch.get_rng_state().clone(),'epoch':1,'step':1,
            'signature':{'original_source':'old'},'logs':[{'epoch':1}]}
    step(model,opt,schedule)
    expected=copy.deepcopy(model.state_dict());expected_opt=copy.deepcopy(opt.state_dict())
    expected_rng=torch.get_rng_state().clone()
    other,other_opt,other_schedule=fixture()
    stream=io.BytesIO();torch.save(origin,stream);stream.seek(0)
    actual=torch.load(stream,weights_only=False)
    other.load_state_dict(actual['model']);other_opt.load_state_dict(actual['optimizer'])
    other_schedule.load_state_dict(actual['scheduler']);torch.set_rng_state(actual['rng'])
    new_signature={'new_source':'new','epoch1_original_source':actual['signature'],
                   'epoch1_original_step':actual['step']}
    assert actual['signature']=={'original_source':'old'}
    step(other,other_opt,other_schedule)
    assert all(torch.equal(v,other.state_dict()[k]) for k,v in expected.items())
    assert torch.equal(expected_rng,torch.get_rng_state())
    for pid,entry in expected_opt['state'].items():
        for key,value in entry.items():assert torch.equal(value,other_opt.state_dict()['state'][pid][key])
    assert other_schedule.state_dict()==schedule.state_dict()
    assert new_signature['epoch1_original_source']=={'original_source':'old'}


def test_cumulative_budget_extension_includes_actual_failed_attempt():
    used={7477:100,7480:126,7485:6021,7524:163}
    assert 28800-sum(used.values()) == 22390
    assert 18000+3*3600==28800
    assert sum(used.values())+22330 < 28800
