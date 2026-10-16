"""Synthetic specification checks only; no real model or feature inputs."""
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
import torch

ROOT=Path(os.environ.get('SPACE_RELEASE_ROOT',str(Path(__file__).parents[1]/'SPACE')))
sys.path.insert(0,str(ROOT/'src'))
from memory_jepa.memory.policies.counter_based import CounterBasedReservoirSamplingPolicy

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

corrected=module('corrected',Path(__file__).with_name('training_utils.py'))
released=module('released',ROOT/'scripts/training_utils.py')


class Predictor:
    memory_slots=4
    def __init__(self):self.calls=[]
    def __call__(self,local,memory,valid,local_age,memory_age):
        self.calls.append({'memory':memory.clone(),'valid':valid.clone(),'local':local.clone(),'age':memory_age.clone()})
        v=(memory*valid.unsqueeze(-1)).sum(1)+local[:,-1]
        return types.SimpleNamespace(predictions=torch.stack([v,v],dim=1))
    def project_inputs(self,x,**kwargs):return x


def run_rollout(lib,seed):
    latents=torch.randn(180,5,generator=torch.Generator().manual_seed(60))
    candidate=latents[89:93].clone();times=torch.arange(89,93)
    predictor=Predictor()
    losses=lib.proposal_rollout_losses(predictor,latents,100,candidate,times,
        torch.arange(4).view(1,4),(1,4),torch.tensor([.5,.5]),'cosine',8,seed)
    return predictor,latents,candidate,times,losses


class PaperSemantics(unittest.TestCase):
    def test_padding_mask_matches_real_retained_capacity(self):
        predictor=Predictor();candidate=torch.randn(4,5);local=torch.randn(8,5)
        corrected.online_features(predictor,candidate,torch.arange(4).float(),local,torch.device('cpu'),'fp32')
        self.assertTrue(predictor.calls[0]['valid'][:,:3].all())
        self.assertFalse(predictor.calls[0]['valid'][:,3].any())
        baseline=Predictor()
        released.online_features(baseline,candidate,torch.arange(4).float(),local,torch.device('cpu'),'fp32')
        self.assertTrue(baseline.calls[0]['valid'][:,3].all())
    def test_every_continuation_branch_matches_common_reservoir(self):
        for seed in (0,17,20260813):
            pred,latents,candidate,times,losses=run_rollout(corrected,seed)
            branches=torch.stack([candidate[torch.arange(4)!=a] for a in range(4)])
            bt=torch.stack([times[torch.arange(4)!=a] for a in range(4)])
            sampler=CounterBasedReservoirSamplingPolicy(3,seed,keep_stream=100,replacement_stream=101)
            for offset,call in enumerate(pred.calls):
                t=100+offset
                if offset:
                    ct=torch.cat([bt,torch.full((4,1),t-8)],1)
                    action=int(sampler.reservoir_proposal(torch.ones(1,4,dtype=torch.bool),torch.tensor([t]),ct[:1]))
                    keep=torch.arange(4)!=action
                    branches=torch.cat([branches,latents[t-8].view(1,1,5).expand(4,1,5)],1)[:,keep]
                    bt=ct[:,keep]
                self.assertTrue(torch.equal(call['memory'][:,:3],branches))
                self.assertTrue(torch.equal(call['local'],latents[t-7:t+1].unsqueeze(0).expand(4,-1,-1)))
                self.assertFalse(call['valid'][:,3].any())
            self.assertEqual(len(pred.calls),32)
            self.assertEqual(set(losses[0]),{8,32})
    def test_released_uniform_continuation_is_detectably_different(self):
        a,*_=run_rollout(corrected,20260813);b,*_=run_rollout(released,20260813)
        self.assertTrue(any(not torch.equal(x['memory'],y['memory']) for x,y in zip(a.calls,b.calls)))
    def test_common_draws_do_not_depend_on_candidate_payload(self):
        a,*_=run_rollout(corrected,7);b,*_=run_rollout(corrected,7)
        self.assertTrue(all(torch.equal(x['age'],y['age']) for x,y in zip(a.calls,b.calls)))


if __name__=='__main__':unittest.main()
