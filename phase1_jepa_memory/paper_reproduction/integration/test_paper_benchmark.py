"""Artificial software invariants only; no research observations or scores."""
import unittest
from dataclasses import replace

import numpy as np
import torch

from space.config import SpaceConfig
from space.model import TemporalJEPAPredictor
from space.pipeline import SPACE
from space.slow_basis import SlowPredictiveBasis
from space.utility import UtilityBank
from paper_benchmark import (CachedRetrieval,OnlineMethods,METHODS,ranking_scores,
    recovery_anchors,memory_distance,normalized_recovery,participant_pair_inference)


def fixture():
    torch.manual_seed(13)
    config=replace(SpaceConfig(),capacity=2)
    model=TemporalJEPAPredictor(d_in=8,d_model=8,n_layers=1,n_heads=2,d_ff=16,
        dropout=0.,memory_slots=3,local_context_len=2,horizons=config.horizons).eval()
    basis=SlowPredictiveBasis(torch.eye(8)[:,:2],torch.ones(2),1,32,.001)
    bank=UtilityBank(torch.randn(7,3,49),torch.randn(7,3))
    return config,model,basis,bank


class RetrievalChecks(unittest.TestCase):
    def test_cached_distribution_matches_release(self):
        torch.manual_seed(23);bank=UtilityBank(torch.randn(13,5,11),torch.randn(13,5));cached=CachedRetrieval(bank,4,7)
        for _ in range(4):
            q=torch.randn(5,11);a=bank.retrieve(q,state_neighbors=4,action_neighbors=7);b=cached.retrieve(q)
            for key in ('mean_cost','standard_error','mean_similarity'):
                torch.testing.assert_close(getattr(a,key),getattr(b,key),rtol=0,atol=0)

    def test_single_neighbor_standard_error_zero(self):
        bank=UtilityBank(torch.ones(1,1,3),torch.zeros(1,1));result=CachedRetrieval(bank,1,1).retrieve(torch.ones(1,3))
        self.assertEqual(float(result.standard_error[0]),0.)

    def test_normalization_does_not_modify_frozen_bank(self):
        _,_,_,bank=fixture();copy=bank.features.clone();CachedRetrieval(bank)
        self.assertTrue(torch.equal(copy,bank.features))


class CausalPolicyChecks(unittest.TestCase):
    def test_space_adapter_matches_public_stream(self):
        config,model,basis,bank=fixture();ours=OnlineMethods(model,basis,CachedRetrieval(bank),config,seed=0,video_id='synthetic')
        reference=SPACE(config=config,predictor=model,slow_predictive_basis=basis,utility_bank=bank,policy_seed=0)
        for t in range(8):
            event=torch.randn(8);local=torch.randn(2,8)
            ours.step(event,t,local,t+2,t)
            result=reference.step(event,event_time=t,current_time=t+2,local_context=local,video_id='synthetic')
            torch.testing.assert_close(ours.memories['space'].values,result.memory.values,rtol=0,atol=0)
            self.assertTrue(torch.equal(ours.memories['space'].timestamps,result.memory.timestamps))

    def test_deleted_payload_no_reentry(self):
        config,model,basis,bank=fixture();engine=OnlineMethods(model,basis,CachedRetrieval(bank),config,seed=1,video_id='s')
        previous={m:set() for m in METHODS[1:]};deleted={m:set() for m in previous}
        for t in range(12):
            engine.step(torch.randn(8),t,torch.randn(2,8),t+2,t)
            for m in previous:
                current=set(x for x in engine.origins[m] if x>=0)
                self.assertFalse(current & deleted[m]);deleted[m] |= previous[m]-current;previous[m]=current

    def test_clone_shares_frozen_models_but_not_memory(self):
        config,model,basis,bank=fixture();a=OnlineMethods(model,basis,CachedRetrieval(bank),config,seed=0,video_id='s');b=a.clone()
        self.assertIs(a.predictor,b.predictor);b.memories['fifo'].values.fill_(9)
        self.assertFalse(torch.equal(a.memories['fifo'].values,b.memories['fifo'].values))

    def test_corruption_retains_metadata_and_changes_origins(self):
        config,model,basis,bank=fixture();a=OnlineMethods(model,basis,CachedRetrieval(bank),config,seed=0,video_id='s');b=a.clone()
        for t in range(2):
            local=torch.randn(2,8);a.step(torch.ones(8)*t,t,local,t+2,t);b.step(torch.ones(8)*(t+8),t-9,local,t+2,t+8)
        old=a.memories['fifo'].timestamps.clone();seen=a.seen;a.corrupt(b,[0])
        self.assertTrue(torch.equal(old,a.memories['fifo'].timestamps));self.assertEqual(a.seen,seen)
        self.assertEqual(a.origins['fifo'][0],8);self.assertEqual(memory_distance(a,b,'fifo'),.5)

    def test_recent_never_retains_archive(self):
        config,model,basis,bank=fixture();a=OnlineMethods(model,basis,CachedRetrieval(bank),config,seed=0,video_id='s')
        for t in range(4):a.step(torch.randn(8),t,torch.randn(2,8),t+2,t)
        self.assertEqual(a.memories['recent'].active_count,0)


class EvaluationChecks(unittest.TestCase):
    def test_perfect_ranking_and_absent_class_convention(self):
        y=np.zeros((3,24),np.uint8);y[0,0]=1;y[1,1]=1;y[2,0]=1
        p=y*.9+.05;r=ranking_scores(y,p)
        self.assertAlmostEqual(r['mAP'],2/24);self.assertEqual(r['LRAP'],1.);self.assertEqual(r['Recall_at_5'],1.)

    def test_background_disclosed(self):
        y=np.zeros((2,24),np.uint8);y[1,0]=1;r=ranking_scores(y,y+.01)
        self.assertEqual(r['background_fraction'],.5);self.assertEqual(r['known_targets'],2)

    def test_ineligible_recovery_never_fabricates_donor(self):
        y=np.zeros((300,24),np.uint8);y[:,0]=1;self.assertEqual(recovery_anchors(y),[])

    def test_recovery_eligibility_and_only_earlier_donors(self):
        y=np.zeros((300,24),np.uint8);y[:80,0]=1;y[80:,1]=1;a=recovery_anchors(y)
        self.assertEqual(len(a),4)
        for t,d in a:
            self.assertLess(d,t);self.assertGreaterEqual(d,23);self.assertLess(t+16+64,len(y))
            self.assertTrue((y[t:t+17]==y[t]).all());self.assertFalse(np.array_equal(y[d],y[t]))

    def test_recovery_auc_offset_one_to_sixteen(self):
        r=normalized_recovery([.25,0,0,0,0,0]);self.assertEqual(r['normalized_auc'],1.)
        self.assertEqual(r['initial_distance'],.25);self.assertEqual(r['final_distance'],0.)

    def test_zero_initial_distance_is_disclosed(self):
        r=normalized_recovery([0]*6);self.assertTrue(r['denominator_floored']);self.assertEqual(r['normalized_auc'],0.)

    def test_negative_recovery_is_not_clipped(self):
        r=normalized_recovery([.1,.2,.2,.2,.2,.2]);self.assertLess(r['normalized_auc'],0.)

    def test_participant_paired_inference(self):
        r=participant_pair_inference([.01]*10);self.assertAlmostEqual(r['exact_one_sided_sign_flip_p'],1/1024)
        self.assertAlmostEqual(r['CI95'][0],.01);self.assertAlmostEqual(r['CI95'][1],.01)


if __name__=='__main__':
    torch.set_num_threads(1);unittest.main()
