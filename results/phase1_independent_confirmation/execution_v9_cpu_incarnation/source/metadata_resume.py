"""Resume before the first encoder frame, preserving V6 and all raw train/val data."""
import json,math,shutil,hashlib
from pathlib import Path
from phase1.utils import read_json,write_json,file_hash
from budget_guard import native_allocations,CAP,PRIOR

ALLOWED={'config.json','stage.sh','watchdog.sh','cleanup.sh','orchestrator.py','budget_guard.py','protocol.json','restart_authorization.json','guarded_encode.py','test_submission.py'}

def initialize(root,protocol,c):
 root=Path(root);resume=protocol['metadata_resume'];old=Path(resume['origin_root'])
 assert file_hash(old/'source_lock.json')==resume['origin_source_lock_sha256']
 lock=read_json(old/'source_lock.json')
 for n,h in lock['files'].items():
  assert file_hash(old/n)==h,n
  if n not in ALLOWED:assert file_hash(root/n)==h,n
 native=native_allocations([8840,8849]);watch=read_json(old/'watchdog_terminal.json');charges={}
 for row in native:
  assert row['state'] in ('COMPLETED','FAILED','CANCELLED','TIMEOUT')
  h=watch['GPU_jobs'][str(row['job'])]
  from watchdog import epoch
  charges[str(row['job'])]=max(row['elapsed'],h['charged_seconds'],math.ceil(epoch(row['end'])-h['first_allocated_start']))
 assert charges==resume['settled_origin_GPU_charges'] and 39897+sum(charges.values())==PRIOR
 for name,sha in resume['origin_proofs'].items():assert file_hash(old/name)==sha,name
 proof=read_json(old/'metadata_failure_native_evidence.json')
 assert proof['JSON_wire_metadata_equal'] and proof['encoded_frames']==0 and proof['new_models']==0 and proof['fresh_test_unopened']
 previous=read_json(old/'config.json');a=dict(previous);b=dict(c);a.pop('output');b.pop('output');assert a==b
 D=Path(c['output']);D.mkdir(exist_ok=False)
 shutil.copytree(Path(previous['output'])/'dataset',D/'dataset')
 from staged_data import require_stage
 m=require_stage(c);assert file_hash(D/'dataset/manifest.json')==proof['train_validation_dataset_manifest_sha256']
 assert {r['split'] for r in m['records']}=={'train','validation'}
 for name in ['preflight_cpu.json','preflight_gpu.json']:
  shutil.copy2(old/name,root/name);assert read_json(root/name)['all_exact']
 assert next(r for r in native if r['job']==8840)['state']=='COMPLETED'
 # Bind the original native proof origin in a separate receipt, not a fabricated new run.
 write_json(root/'resumed_execution_provenance.json',dict(origin_root=str(old),origin_proofs=resume['origin_proofs'],
  original_GPU_preflight_job=8840,origin_GPU_native=native,origin_configuration_charges=charges,
  original_train_val_dataset_copied_byte_exact=True,test_unopened=True,new_models=0,
  only_encoder_metadata_JSON_representation_check_changed=True))
 write_json(D/'sequence_budget.json',dict(prior_settled_seconds=PRIOR,cap_seconds=CAP,native_prior_verified=True,
  total_frames=198656,GPU_job_ids=[],epochs={},encoded_frames=0,native_GPU_equivalence_done=True,
  origin_settlement=resume['settled_origin_GPU_charges']))
 from metadata_cpu_check import verify
 verify(root,c,protocol)
