"""Course-CPU metadata only: no model construction, forward, training or fresh test."""
import sys,os,json,pathlib,hashlib,math,datetime
R=pathlib.Path('/scratch/xl4358/phase1_independent_confirmation_execution_v6');sys.path[:0]=[str(R),str(R/'original_src')]
from watchdog import native_job,write,epoch
from budget_guard import native_allocations
me=native_job(int(sys.argv[1]));assert me['JobState']=='RUNNING' and me['Account']=='csci_ga_2271-2026fa' and 'gres/gpu=' not in me['AllocTRES']
lock=json.loads((R/'source_lock.json').read_text());assert all(hashlib.sha256((R/n).read_bytes()).hexdigest()==h for n,h in lock['files'].items())
prior_ids=[7477,7480,7485,7524,7535,7574,7582,8496,8511,8512,8518,8536,8792,8799]
prior_rows=native_allocations(prior_ids);assert sum(r['elapsed'] for r in prior_rows)==39897
rows=native_allocations([8840,8849]);watch=json.loads((R/'watchdog_terminal.json').read_text());charges={}
for row in rows:
 assert row['state'] not in ['RUNNING','CONFIGURING','PENDING','COMPLETING'] and row['tres'].get('gres/gpu')=='1'
 h=watch['GPU_jobs'][str(row['job'])];charges[str(row['job'])]=max(row['elapsed'],h['charged_seconds'],math.ceil(epoch(row['end'])-h['first_allocated_start']))
cpus=native_allocations([8830,8833,int(json.loads((R/'watchdog_cleanup_submission.json').read_text())['job'])]);assert all(x['state'] not in ['RUNNING','PENDING','CONFIGURING','COMPLETING'] for x in cpus)
c=json.loads((R/'config.json').read_text());D=pathlib.Path(c['output']);manifest=json.loads((D/'dataset/manifest.json').read_text());splits={r['split'] for r in manifest['records']};assert splits=={'train','validation'}
assert not list((D/'models').glob('*/*/seed_*')) and not (D/'observation_maps/fitted_manifest.json').exists()
ix=D/'features/index.json';assert not ix.exists() or not json.loads(ix.read_text())['episodes']
# Read checkpoint CONFIG and processor only; never instantiate a model on CPU.
from phase1.encoders import checkpoint_components
import torch
mc,processor,revision,_=checkpoint_components(c['encoder']);mc.dtype=getattr(torch,c['encoder']['dtype'])
expected=json.loads((R/'protocol.json').read_text())['encoder_metadata'];actual=dict(expected,model_config=mc.to_dict(),processor=processor.to_dict(),resolved_revision=revision)
canonical=lambda x:json.loads(json.dumps(x,sort_keys=True,allow_nan=False))
raw_equal=actual==expected;serialized_equal=canonical(actual)==expected
proof=dict(status='NATIVE_V6_ENCODER_METADATA_DIAGNOSIS',native_GPU=rows,native_prior=prior_rows,charged_including_configuration=charges,prior_GPU_seconds=39897,cumulative_GPU_seconds=39897+sum(charges.values()),cap_GPU_seconds=54051,remaining_GPU_seconds=54051-39897-sum(charges.values()),CPU_terminals=cpus,source_files_match=True,train_validation_dataset_manifest_sha256=hashlib.sha256((D/'dataset/manifest.json').read_bytes()).hexdigest(),fresh_test_unopened=True,new_models=0,new_maps=0,encoded_frames=0,raw_metadata_equal=raw_equal,JSON_wire_metadata_equal=serialized_equal,expected_metadata=expected,config_and_processor_metadata=canonical(actual),parameter_bytes_copied_from_frozen_contract_not_CPU_model_load=True,no_CPU_model_construct_forward_or_training=True,actual_CPU=me)
write(R/'metadata_failure_native_evidence.json',proof)
print(json.dumps({k:proof[k] for k in ['status','charged_including_configuration','cumulative_GPU_seconds','remaining_GPU_seconds','raw_metadata_equal','JSON_wire_metadata_equal','fresh_test_unopened']}))
assert not raw_equal and serialized_equal,'Not solely a JSON wire-format mismatch; stop before any GPU retry'
