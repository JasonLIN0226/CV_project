"""Artificial end-to-end scoring and aggregate checks; no real fit or data."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from space.config import SpaceConfig
from space.model import TemporalJEPAPredictor
from space.slow_basis import SlowPredictiveBasis
from space.utility import UtilityBank
from paper_benchmark import METHODS,CachedRetrieval
from run_paper_benchmark import evaluate_record,summaries,verify


class DriverChecks(unittest.TestCase):
    def test_invalid_template_rejected_before_input_loading(self):
        with self.assertRaises(RuntimeError):verify(Path('/no-such-artifact'),{'actual_binding_complete':False})

    def test_full_artificial_record_all_methods_queries_and_recovery(self):
        torch.manual_seed(12);torch.set_num_threads(1)
        model=TemporalJEPAPredictor(d_in=2048,d_model=8,n_layers=1,n_heads=2,d_ff=16,dropout=0.,
            memory_slots=17,local_context_len=8,horizons=(1,4,16,64)).eval()
        basis=SlowPredictiveBasis(torch.eye(8)[:,:2],torch.ones(2),1,32,.001)
        bank=CachedRetrieval(UtilityBank(torch.randn(5,17,49),torch.randn(5,17)))
        decoder=torch.nn.Linear(2048,24).eval();labels=torch.zeros(120,24,dtype=torch.uint8)
        labels[:27,0]=1;labels[27:,1]=1
        record={'features':torch.randn(120,2048),'labels':labels,'valid':torch.ones(120,dtype=torch.bool),
                'video_id':'artificial','participant':'artificial','valid_length':120,'split':'val'}
        with tempfile.TemporaryDirectory() as folder,patch.object(torch.Tensor,'cuda',lambda s:s):
            rows,cases,raw=evaluate_record(record,model,basis,bank,decoder,SpaceConfig(),0,lambda:None,Path(folder))
            self.assertEqual(len(rows),6);self.assertEqual(sum(r['queries'] for r in rows),6*33)
            self.assertEqual(len(cases),4*5*2*3)
            self.assertTrue(all(len(r['recovery'])==6 for r in cases))
            with np.load(raw['path']) as z:
                self.assertEqual(z['nll'].shape,(33,6,4));self.assertEqual(z['probabilities'].shape,(33,6,4,24))
                self.assertTrue(np.array_equal(z['query'],np.arange(23,56)))
                self.assertTrue(np.isfinite(z['signature']).all())
            self.assertTrue(all(np.isfinite(r['Action_NLL']) for r in rows))

    def test_participant_then_record_aggregation_not_frame_weighting(self):
        rows=[]
        for seed in (0,1,2):
            for pid in range(10):
                for method in METHODS:
                    for replica,value in enumerate((1.,9.) if pid==0 else (5.,)):
                        gain=.1 if method=='space' else 0
                        rows.append({'seed':seed,'video_id':f'{pid}/{replica}','participant':str(pid),'method':method,
                                     'queries':1 if replica==0 else 1000,'Action_NLL':value-gain,
                                     'horizon_NLL':[value-gain]*4,'within_regime_drift':.1,'boundary_displacement':.2,
                                     'log_boundary_within_ratio':float(np.log(2)),'within_transitions':1,'boundary_transitions':1,
                                     'true_future_feature_decoder_NLL':2.,'ranking':{'mAP':.2,'LRAP':.5,'Recall_at_5':.7,
                                     'background_fraction':.1,'foreground_LRAP':.4,'foreground_Recall_at_5':.6}})
        scores,recovery=summaries(rows,[])
        self.assertAlmostEqual(scores['methods']['reservoir']['mean']['Action_NLL'],5.)
        self.assertAlmostEqual(scores['methods']['space']['mean']['Action_NLL'],4.9)
        self.assertAlmostEqual(scores['paired_NLL_against_Reservoir']['space']['mean_gain'],.1)
        self.assertEqual(recovery['space/25/state']['status'],'NO_ELIGIBLE_DONOR_CASES')


if __name__=='__main__':unittest.main()
