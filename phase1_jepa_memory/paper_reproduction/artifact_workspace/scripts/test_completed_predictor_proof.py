"""Tiny synthetic checkpoints; mocks scheduler, never submits or fits models."""
from pathlib import Path
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch

SOFTWARE=Path(os.environ.get('SPACE_RELEASE_ROOT',str(Path(__file__).parents[1]/'SPACE')))
sys.path.insert(0,str(SOFTWARE/'src'))
from memory_jepa.utils.modeling import build_predictor
from memory_jepa.utils.hashing import canonical_hash,state_dict_hash
from completed_predictor_proof import completed_proof,sha


def dump(path,data):path.write_text(json.dumps(data,sort_keys=True,indent=2)+'\n')


class CompletionBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.train=self.root/'train';self.train.mkdir()
        self.out=self.root/'output';self.out.mkdir();self.feature=self.root/'fake.pt';torch.save({'synthetic':True},self.feature)
        manifest=self.root/'manifest.json';dump(manifest,{'videos':[{'video_id':'fake'}]})
        data=self.root/'data.json';dump(data,{'status':'OFFICIAL_TSM_TRAIN_VAL_PREPARED_AND_AUDITED','test_payload_deserialized':False,
            'manifest':str(manifest),'features':[{'video_id':'fake','bytes':self.feature.stat().st_size,'sha256':sha(self.feature)}]})
        dump(self.train/'paper_training_protocol.json',{'data_report':str(data),'data_report_sha256':sha(data),'manifest_sha256':sha(manifest)})
        dump(self.train/'source_lock.json',{'files':{'paper_training_protocol.json':sha(self.train/'paper_training_protocol.json')}})
        self.config={'predictor':{'d_model':8,'n_layers':1,'n_heads':2,'d_ff':16,'dropout':0.,'memory_budget':16,'local_context_len':8,'horizons':[1,4,16,64]}}
        proofs=[]
        for seed in (0,1,2):
            folder=self.out/f'seed_{seed}';folder.mkdir()
            history=[{'epoch':i,'steps':2*i,'validation':{'loss':float(i)}} for i in range(1,31)]
            dump(folder/'predictor_history.json',history)
            model=build_predictor(self.config,5)
            optimizer={'param_groups':[{'weight_decay':.05}],'state':{0:{'step':torch.tensor(60.),'exp_avg':torch.zeros(8)}}}
            for tag,epoch in (('best',1),('latest',30)):
                ck={'seed':seed,'epoch':epoch,'global_step':epoch*2,'complete':epoch==30,
                    'source_lock_sha256':sha(self.train/'source_lock.json'),'protocol_sha256':sha(self.train/'paper_training_protocol.json'),
                    'manifest_sha256':sha(manifest),'config':self.config,'config_hash':canonical_hash(self.config),'d_in':5,
                    'model':model.state_dict(),'predictor_hash':state_dict_hash(model),'optimizer':optimizer,
                    'scheduler':{'last_epoch':epoch*2},'rng':{'python':(),'numpy':[],'torch_cpu':torch.get_rng_state(),'torch_cuda':[]}}
                torch.save(ck,folder/f'predictor_{tag}.pt')
            proofs.append({'seed':seed,'epochs':30,'steps':60,'expected_steps':60,'best_epoch':1,
                           'history_sha256':sha(folder/'predictor_history.json'),
                           'best_sha256':sha(folder/'predictor_best.pt'),'latest_sha256':sha(folder/'predictor_latest.pt')})
        self.report={'resources':{'job_id':888,'visible_gpus':1,'gpu':'synthetic A100'},
                     'elapsed_seconds':.1,
                     'protocol_sha256':sha(self.train/'paper_training_protocol.json'),'source_lock_sha256':sha(self.train/'source_lock.json'),
                     'completed_models':proofs}
        dump(self.out/'training_report.json',self.report);self.sub=self.root/'submission.json';dump(self.sub,{'job_id':888,'output':str(self.out)})
    def tearDown(self):self.temp.cleanup()
    def call(self):
        with patch('completed_predictor_proof.subprocess.check_output',return_value='888|COMPLETED|0:0|csci_ga_2271-2026fa\n'):
            return completed_proof(self.sub,self.train)
    def test_six_serialized_boundary_models_prove_without_inference(self):
        result=self.call();self.assertEqual(len(result['models']),6)
        self.assertFalse(result['heldout_model_inference_before_proof'])
    def test_report_cannot_choose_a_later_checkpoint_than_earliest_best(self):
        self.report['completed_models'][0]['best_epoch']=30;dump(self.out/'training_report.json',self.report)
        with self.assertRaisesRegex(RuntimeError,'earliest'):self.call()
    def test_hash_consistent_nonfinite_weights_still_rejected(self):
        file=self.out/'seed_0/predictor_best.pt';ck=torch.load(file,weights_only=True)
        key=next(iter(ck['model']));ck['model'][key].view(-1)[0]=float('nan')
        model=build_predictor(self.config,5);model.load_state_dict(ck['model']);ck['predictor_hash']=state_dict_hash(model)
        torch.save(ck,file);self.report['completed_models'][0]['best_sha256']=sha(file);dump(self.out/'training_report.json',self.report)
        with self.assertRaisesRegex(RuntimeError,'nonfinite'):self.call()
    def test_failed_native_parent_is_not_completed_by_file_presence(self):
        with patch('completed_predictor_proof.subprocess.check_output',return_value='888|FAILED|2:0|csci_ga_2271-2026fa\n'):
            with self.assertRaisesRegex(RuntimeError,'parent'):completed_proof(self.sub,self.train)
    def test_complete_file_counts_do_not_override_frozen_budget(self):
        self.report['elapsed_seconds']=13801;dump(self.out/'training_report.json',self.report)
        with self.assertRaisesRegex(RuntimeError,'budget'):self.call()


if __name__=='__main__':unittest.main()
