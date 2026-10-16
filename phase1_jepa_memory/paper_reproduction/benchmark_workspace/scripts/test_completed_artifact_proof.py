import unittest
import torch
from completed_artifact_proof import check_payloads


def fake():
    basis={'basis':torch.eye(512)[:,:32],'eigenvalues':torch.arange(32,0,-1).float(),'participants':[str(i) for i in range(29)],
           'fit_split':'train','validation_features_accessed':False,'test_features_accessed':False,'predictor_hash':'m',
           'short_lag':1,'long_lag':32,'regularization':.001}
    meta={'predictor_hash':'m','manifest_hash':'f','split':'train','validation_features_accessed':False,
          'test_features_accessed':False,'states':2,'memory_budget':16,'local_context_len':8,'states_per_participant':64}
    tensors={'utility_scalar':torch.zeros(2,17),'proposal_rollout_delta':torch.zeros(2,1,17),'reference_evict_index':torch.zeros(2,dtype=torch.long)}
    return basis,{'metadata':dict(meta),'features':torch.zeros(2,17,3073),'targets':torch.zeros(2,17)}, {'metadata':dict(meta),'tensors':tensors}


class Checks(unittest.TestCase):
    def test_complete_synthetic_payload(self):
        a,b,c=fake();self.assertTrue(check_payloads(a,b,c,model_hash='m',manifest_hash='f',expected_states=2)['all_finite'])
    def test_nonfinite_rejected(self):
        a,b,c=fake();b['features'][0,0,0]=float('nan')
        with self.assertRaises(RuntimeError):check_payloads(a,b,c,model_hash='m',manifest_hash='f',expected_states=2)
    def test_validation_fit_rejected(self):
        a,b,c=fake();a['validation_features_accessed']=True
        with self.assertRaises(RuntimeError):check_payloads(a,b,c,model_hash='m',manifest_hash='f',expected_states=2)
    def test_cross_model_bank_rejected(self):
        a,b,c=fake();c['metadata']['predictor_hash']='other'
        with self.assertRaises(RuntimeError):check_payloads(a,b,c,model_hash='m',manifest_hash='f',expected_states=2)
    def test_bank_target_mismatch_rejected(self):
        a,b,c=fake();b['targets'][0,0]=1
        with self.assertRaises(RuntimeError):check_payloads(a,b,c,model_hash='m',manifest_hash='f',expected_states=2)


if __name__=='__main__':unittest.main()
