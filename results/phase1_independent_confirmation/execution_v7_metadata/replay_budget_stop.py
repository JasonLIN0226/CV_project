from pathlib import Path
import json,hashlib,datetime,math
B=Path(__file__).resolve().parent;E=B/'native_budget_stop'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
t=read(E/'transfer_integrity.json');assert t['all_match'] and t['native_before']==t['local']==t['native_after']
for n,v in t['local'].items():assert sha(E/n)==v['sha256'] and (E/n).stat().st_size==v['bytes']
x=read(E/'native_budget_stop_evidence.json');r=x['actual_GPU'];w=read(E/'watchdog_terminal.json');h=w['GPU_jobs']['8886']
assert r['state'] in ('FAILED','CANCELLED','COMPLETED','TIMEOUT') and r['tres']['gres/gpu']=='1' and r['account']=='csci_ga_2271-2026fa'
epoch=lambda s:datetime.datetime.fromisoformat(s.removesuffix('Z')).replace(tzinfo=datetime.timezone.utc).timestamp()
raw=sum(a['elapsed'] for a in r['attempts']);wall=math.ceil(epoch(r['end'])-h['first_allocated_start']);charge=max(raw,wall,h['charged_seconds'])
assert charge==x['charged_GPU_seconds_including_configuration']==216
assert 40283+charge==x['cumulative_GPU_seconds']==40499
assert 54051-40499==x['remaining_GPU_seconds']==13552
s=read(E/'fresh/sequence_budget.json');b=read(E/'fresh/budget_stop.json');ix=read(E/'fresh/features/index.json')
frames=sum(e['shape'][0] for e in ix['episodes'].values());assert frames==s['encoded_frames']==x['committed_encoded_frames']==812
assert not s['epochs'] and s['GPU_job_ids']==[8886] and x['models_trained']==x['maps_fitted']==0 and x['fresh_test_unopened']
training=sum((120-s['epochs'].get(f'{role}/{task}/{seed}',0))*rate for task,role,rate in [('gate','reference',2.572098377),('motion','reference',2.622685901),('gate','candidate',5.667439204999994),('motion','candidate',3.5664218639999774)] for seed in [17,29,43])
encoding=(198656-812)/s['encoder_frames_per_second'];forecast=40283+(b['actual_GPU_seconds']-40283)+1.15*(training+encoding+27)+1268.295233886
assert abs(forecast-b['projected_cumulative_seconds'])<1e-6 and forecast>54051 and not b['fits']
assert x['actual_encoder_loaded']['all_finite'] and x['actual_encoder_loaded']['metadata_exact']
assert x['source_all_current_SHA_match'] and x['source_files']==100 and x['all_committed_feature_SHA_float32_finite_verified']
result=dict(passed=True,status='PASS_INDEPENDENT_LOCAL_EXECUTION_BOUNDARY_AUDIT',phase1_go=False,scientific_standard_failed=False,evidence_incomplete_due_to_budget_forecast=True,transfer_files=len(t['local']),all_native_local_native_SHA_match=True,actual_GPU_total=40499,actual_new_GPU_including_configuration=216,remaining_GPU_seconds=13552,forecast_cumulative_replayed=forecast,forecast_excess_seconds=forecast-54051,actual_frames=812,encoded_required_frames=x['train_validation_required_frames'],new_models_trained=0,required_new_models=12,fresh_test_unopened=True,CPU_terminals=x['CPU_terminal'])
(B/'local_budget_stop_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='CPU_terminals'}))
