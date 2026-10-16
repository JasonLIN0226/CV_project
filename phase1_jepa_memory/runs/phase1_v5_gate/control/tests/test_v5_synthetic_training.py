import copy
import numpy as np
import torch
from phase1.v5_synthetic_reader import SyntheticCausalReader
from phase1.v5_synthetic_training import train_epoch, cpu_tree, finite_tree
from phase1.training import TRAIN_CONDITIONS, regression_loss, validation_loss, validation_score


class ArtificialStore:
    def __init__(self):
        self.calls = []
    def sample(self, row, condition, rng=None):
        draw = int(rng.integers(100000)) if rng is not None else 0
        self.calls.append((row['pair'],row['id'],condition,draw))
        r = np.random.default_rng(row['id'])
        data = {'z':r.normal(size=(4,8)).astype('float32'), 'age':np.arange(3,-1,-1,dtype='float32'),
                'type':np.zeros(4,dtype='int64'), 'past_actions':np.zeros((4,2),dtype='float32'),
                'actions':np.zeros((6,2),dtype='float32'), 'current':r.normal(size=8).astype('float32')}
        return data,r.normal(size=(2,8)).astype('float32'),{}


def test_balanced_original_pair_loop_and_validation_selection():
    torch.set_num_threads(1)
    basis = {'input_mean':np.zeros(8,np.float32),'projection':np.eye(8,4,dtype=np.float32),
             'target_mean':np.zeros(8,np.float32),'output_basis':np.eye(8,4,dtype=np.float32)}
    model = SyntheticCausalReader(basis)
    pairs = [[{'id':i,'pair':i//2,'task':'gate'},{'id':i+1,'pair':i//2,'task':'gate'}] for i in (0,2)]
    c = {'training':{'batch_size':4,'clip':1.,'pair_weight':2.,'rank_weight':.5},
         'data':{'tasks':['gate']},'experiment':{'arm':'reference','stage':'development','tasks':['gate']}}
    opt = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    store = ArtificialStore()
    result = train_epoch(c,store,model,opt,pairs,17,0,{'requested':False})
    assert result['steps'] == 4
    assert result['counts'] == {k:4 for k in TRAIN_CONDITIONS}
    for start in range(0,len(store.calls),2):
        a,b = store.calls[start:start+2]
        assert a[0] == b[0] and a[2:] == b[2:]
    val = validation_loss(c,store,model,pairs,'cpu','gate')
    assert validation_score(c,val) == np.mean([val[k]['mse'] for k in TRAIN_CONDITIONS])
    assert finite_tree(model.state_dict()) and finite_tree(opt.state_dict())
    frozen = cpu_tree(model.state_dict())
    with torch.no_grad(): next(model.parameters()).add_(1)
    assert any(not torch.equal(frozen[k],v) for k,v in model.state_dict().items())


def test_literal_original_loss_and_stop_discard_path():
    pred = torch.tensor([[[0.,1.]],[[2.,3.]]],requires_grad=True)
    target = torch.tensor([[[1.,0.]],[[3.,2.]]])
    loss,mse,pair = regression_loss(pred,target,2.,.5)
    other = target.flip(0)
    expected = mse+2*pair+.5*torch.nn.functional.softplus(-4*((pred-other).square().mean(-1)-(pred-target).square().mean(-1))/(target-other).square().mean(-1).clamp_min(1e-6)).mean()
    torch.testing.assert_close(loss,expected)
    assert not finite_tree({'x':torch.tensor(float('nan'))})
