import pathlib,json,os,sys,subprocess,time
import torch
from phase1.proposal_trial import verify_control
from phase1.exact_runtime_cache import CachedFeatureStore
from phase1.model import fit_basis
from phase1.training import paired_rows
from phase1.utils import file_hash,write_json
r=pathlib.Path('/scratch/xl4358/phase1_proposal_validation/fast_execution');control=r/'control'
if (r/'training_submission.json').exists():
 print('Actual scientific receipt already exists; no duplicate');raise SystemExit(0)
if (r/'training_submission_attempt.json').exists():raise ValueError('Unknown prior submission attempt; no retry')
torch.set_num_threads(1);p,lock,configs=verify_control(control)
pre=json.loads((r/'deployment_preflight.json').read_text());assert pre['contract_exit']==pre['software_exit']==0
path=r/'native_fast_input_audit.json';proof=json.loads(path.read_text()) if path.exists() else {'source_lock_sha256':file_hash(control/'source_lock.json'),'tasks':{},'all_exact':False,'no_model_forward':True,'no_held_out_inputs':True}
assert proof['source_lock_sha256']==file_hash(control/'source_lock.json')
for task,c in configs.items():
 if proof.get('all_exact'):break
 store=CachedFeatureStore(c,max_cached=64)
 if task in proof['tasks']:
  assert store.metadata_fingerprint==proof['tasks'][task]['metadata_fingerprint'];del store;continue
 basis=fit_basis(c,store,task);store.bases={task:basis}
 result=store.verify_all_inputs();pairs=paired_rows([x for x in store.manifest['records'] if x['task']==task and x['split']=='validation']);store.prepare_validation(pairs,'cpu');result['original_validation_batches_exact']=store.verify_validation_inputs(pairs)
 assert result['exact_samples_targets_info_rng']==4864 and result['original_validation_batches_exact']==12
 proof['tasks'][task]=result;write_json(path,proof);del store
assert set(proof['tasks'])=={'gate','motion'};proof.update(all_exact=True,CPU_continuation_job=os.environ['SLURM_JOB_ID']);write_json(path,proof)
pre.update(all_inputs_exit=0,actual_independent_CPU_completion_job=os.environ['SLURM_JOB_ID']);write_json(r/'final_preflight.json',pre)
subprocess.run(['/scratch/xl4358/phase1_env/bin/python',str(r/'launch_control/submit.py')],check=True)
