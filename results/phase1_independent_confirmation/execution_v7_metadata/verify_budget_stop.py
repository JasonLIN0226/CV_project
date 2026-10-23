"""Read-only terminal evidence on one actual course CPU. No model forward or fitting."""
import pathlib,sys,json,hashlib,math,socket
R=pathlib.Path('/scratch/xl4358/phase1_independent_confirmation_execution_v7_metadata');sys.path[:0]=[str(R),str(R/'original_src')]
from watchdog import native_job,epoch,write
from budget_guard import native_allocations,PRIOR,CAP
me=native_job(int(sys.argv[1]));assert me['JobState']=='RUNNING' and me['Account']=='csci_ga_2271-2026fa' and me['NodeList']==socket.gethostname().split('.')[0] and 'gres/gpu=' not in me['AllocTRES']
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
lock=read(R/'source_lock.json');assert all(sha(R/n)==h for n,h in lock['files'].items())
row=native_allocations([8886])[0];assert row['state'] in ['COMPLETED','FAILED','CANCELLED','TIMEOUT'] and row['tres'].get('gres/gpu')=='1'
w=read(R/'watchdog_terminal.json');h=w['GPU_jobs']['8886'];charged=max(row['elapsed'],h['charged_seconds'],math.ceil(epoch(row['end'])-h['first_allocated_start']))
c=read(R/'config.json');D=pathlib.Path(c['output']);s=read(D/'sequence_budget.json');b=read(D/'budget_stop.json');ix=read(D/'features/index.json');loaded=read(D/'encoder_loaded_train.json')
assert loaded['all_finite'] and loaded['metadata_exact'] and loaded['actual_float_tensors_verified']>0
import numpy as np
for name,e in ix['episodes'].items():
 p=D/'features'/e['path'];assert sha(p)==e['sha256']
 with np.load(p,allow_pickle=False) as f:
  z=f['z'];assert z.dtype==np.float32 and list(z.shape)==e['shape'] and np.isfinite(z).all()
frames=sum(e['shape'][0] for e in ix['episodes'].values());assert frames==s['encoded_frames'] and frames>0
m=read(D/'dataset/manifest.json');assert {r['split'] for r in m['records']}=={'train','validation'}
assert not s.get('epochs') and not list((D/'models').glob('*/*/seed_*')) and not (D/'observation_maps/fitted_manifest.json').exists()
assert b['fits'] is False and b['projected_cumulative_seconds']>CAP
assert 'BudgetStop' in read(R/'stage_failures/encode_train.json')['error']
cpus=native_allocations([8881,8883,8885]);assert all(r['state'] not in ['RUNNING','PENDING','CONFIGURING','COMPLETING'] for r in cpus)
proof=dict(status='INCOMPLETE_INDEPENDENT_CONFIRMATION_BUDGET_STOP',phase1_go=False,scientific_standard_failure=False,actual_GPU=row,charged_GPU_seconds_including_configuration=charged,prior_GPU_seconds=PRIOR,cumulative_GPU_seconds=PRIOR+charged,remaining_GPU_seconds=CAP-PRIOR-charged,GPU_cap_seconds=CAP,CPU_terminal=cpus,actual_encoder_loaded=loaded,committed_encoded_frames=frames,committed_episodes=len(ix['episodes']),all_committed_feature_SHA_float32_finite_verified=True,train_validation_required_frames=sum(r['frames'] for r in m['records']),source_files=len(lock['files']),source_all_current_SHA_match=True,source_lock_sha256=sha(R/'source_lock.json'),encoder_frames_per_second_used=s.get('encoder_frames_per_second'),projected_cumulative_seconds_at_stop=b['projected_cumulative_seconds'],forecast_excess_seconds=b['projected_cumulative_seconds']-CAP,full_budget_forecast=b,models_trained=0,required_new_models=12,maps_fitted=0,fresh_test_unopened=True,no_GPU_restart_or_new_science_submission=True,actual_CPU=me,no_CPU_model_construct_forward_fitting=True)
write(R/'native_budget_stop_evidence.json',proof)
print(json.dumps({k:proof[k] for k in ['status','committed_encoded_frames','charged_GPU_seconds_including_configuration','cumulative_GPU_seconds','remaining_GPU_seconds','encoder_frames_per_second_used','models_trained','fresh_test_unopened']}))
