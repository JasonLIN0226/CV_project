"""Explicit new8h authorization resumes existing data and812features without refitting."""
import shutil,math
from pathlib import Path
import numpy as np
from phase1.utils import read_json,write_json,file_hash
from budget_guard import native_allocations,CAP,PRIOR
ALLOWED={'config.json','stage.sh','watchdog.sh','cleanup.sh','orchestrator.py','budget_guard.py','protocol.json','restart_authorization.json','authorization.json','guarded_encode.py','fresh_training.py','terminal_evidence.py','startup_client.py','watchdog.py','guardian_peer.py','guardian_preflight.py','test_startup_integration.py'}

def initialize(root,protocol,c):
 root=Path(root)
 stop=read_json(root/'v8_CPU_startup_stop_evidence.json')
 assert stop['GPU_receipts']==[] and stop['GPU_jobs']==[] and stop['prior_GPU_seconds']==40499
 repair=protocol['CPU_first_batch_startup_repair'];failed=Path(repair['origin_root'])
 assert file_hash(failed/'native_CPU_startup_stop_evidence.json')==file_hash(root/'v8_CPU_startup_stop_evidence.json')==repair['native_zero_GPU_settlement_sha256']
 assert file_hash(failed/'source_lock.json')==repair['origin_source_lock_sha256']
 for name,h in read_json(failed/'source_lock.json')['files'].items():assert file_hash(failed/name)==h,name
 rows=native_allocations([8908,8927,8928])
 assert all(x['state'] in ['COMPLETED','FAILED','CANCELLED','TIMEOUT'] and 'gres/gpu' not in x['tres'] for x in rows)
 r=protocol['metadata_resume'];old=Path(r['origin_root']);oldlock=read_json(old/'source_lock.json')
 assert file_hash(old/'source_lock.json')==r['origin_source_lock_sha256']
 for name,h in oldlock['files'].items():
  assert file_hash(old/name)==h,name
  if name not in ALLOWED:assert file_hash(root/name)==h,name
 for name,h in r['origin_proofs'].items():assert file_hash(old/name)==h,name
 proof=read_json(old/'native_budget_stop_evidence.json');row=native_allocations([8886])[0]
 assert row['state'] in ['COMPLETED','FAILED','CANCELLED','TIMEOUT'] and row['elapsed']==proof['actual_GPU']['elapsed']
 from watchdog import epoch
 wh=read_json(old/'watchdog_terminal.json')['GPU_jobs']['8886']
 charge=max(row['elapsed'],wh['charged_seconds'],math.ceil(epoch(row['end'])-wh['first_allocated_start']))
 assert charge==216 and 40283+charge==PRIOR==40499 and CAP==69299
 assert proof['models_trained']==proof['maps_fitted']==0 and proof['fresh_test_unopened'] and proof['committed_encoded_frames']==812
 prev=read_json(old/'config.json');a=dict(prev);b=dict(c);a.pop('output');b.pop('output');assert a==b
 D=Path(c['output']);D.mkdir(exist_ok=False)
 shutil.copytree(Path(prev['output'])/'dataset',D/'dataset')
 shutil.copytree(Path(prev['output'])/'features',D/'features')
 from staged_data import require_stage
 m=require_stage(c);assert {x['split'] for x in m['records']}=={'train','validation'}
 ix=read_json(D/'features/index.json');frames=0
 for k,e in ix['episodes'].items():
  p=D/'features'/e['path'];assert file_hash(p)==e['sha256']
  with np.load(p,allow_pickle=False) as f:
   z=f['z'];assert z.dtype==np.float32 and list(z.shape)==e['shape'] and np.isfinite(z).all()
  frames+=e['shape'][0]
 assert frames==812
 for name in ['preflight_cpu.json','preflight_gpu.json']:
  shutil.copy2(old/name,root/name);assert read_json(root/name)['all_exact']
 write_json(root/'resumed_execution_provenance.json',dict(origin_root=str(old),origin_proofs=r['origin_proofs'],
  original_GPU_preflight_job=8840,original_GPU_preflight_inherited_without_reexecution=True,
  native_budget_stop_GPU=row,charged_configuration_seconds_included=charge,
  train_validation_dataset_copied_byte_exact=True,committed_encoded_frames_reused=812,
  new_models=0,new_maps=0,test_unopened=True,scientific_thresholds_samples_seed_epochs_unchanged=True))
 write_json(D/'sequence_budget.json',dict(prior_settled_seconds=PRIOR,cap_seconds=CAP,native_prior_verified=True,
  total_frames=198656,GPU_job_ids=[],epochs={},encoded_frames=812,native_GPU_equivalence_done=True,
  encoder_frames_per_second=proof['encoder_frames_per_second_used'],origin_settlement=proof['cumulative_GPU_seconds']))
 from metadata_cpu_check import verify
 verify(root,c,protocol)
 # Native terminal proof needed by the final evidence manifest, even in a resumed chain.
 release=native_allocations([8523])[0];assert release['state'] not in ['RUNNING','PENDING','CONFIGURING','COMPLETING']
 write_json(root/'resource_release8523.json',release)
 write_json(root/'resume_cache_native_audit.json',dict(passed=True,frames=812,feature_episodes=len(ix['episodes']),
  original_dataset_manifest_sha256=file_hash(Path(prev['output'])/'dataset/manifest.json'),
  copied_dataset_manifest_sha256=file_hash(D/'dataset/manifest.json'),
  source_files=len(oldlock['files']),all_source_SHA_exact=True,no_model_forward_or_training=True))
