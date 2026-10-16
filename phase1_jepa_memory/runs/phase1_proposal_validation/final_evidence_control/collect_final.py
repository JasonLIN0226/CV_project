from pathlib import Path
import json,subprocess,hashlib,datetime,csv,collections,math
r=Path('/scratch/xl4358/phase1_proposal_validation/fast_execution')
sac=subprocess.run(['/opt/slurm/bin/sacct','-j','7477,7480,7485,7486,7487,7506','--parsable2','--noheader','--format=JobID,JobName,Account,Partition,State,ExitCode,Start,End,ElapsedRaw,AllocTRES,NodeList'],capture_output=True,text=True,check=True).stdout
lines=[x.split('|') for x in sac.strip().splitlines()]; jobs={x[0]:x for x in lines}
for j in ['7485','7486','7487']:
 assert jobs[j][4]=='COMPLETED' and jobs[j][5]=='0:0',(j,jobs[j])
assert jobs['7506'][4]=='RUNNING'
locks={}
for d,n in [('control',48),('independent_audit',3)]:
 lp=r/d/'source_lock.json'; lock=json.loads(lp.read_text()); assert len(lock['files'])==n
 actual={k:hashlib.sha256((r/d/k).read_bytes()).hexdigest() for k in lock['files']};assert actual==lock['files']
 locks[d]={'sha256':hashlib.sha256(lp.read_bytes()).hexdigest(),'files':actual,'all_match':True}
e=r/'evaluation/job_7486';o=json.loads((e/'outcome.json').read_text());a=json.loads((e/'independent_scalar_audit.json').read_text())
assert o['status']=='DEVELOPMENT_REJECTED' and not o['phase1_go'] and not o['confirmation_opened']
assert a['status']=='INDEPENDENT_RAW_AXIS_FINITE_AND_64_FIXED_COMPARISONS_PASS'
train=json.loads((r/'training/job_7485/all_models_completion.json').read_text());ev=json.loads((e/'all_models_completion.json').read_text());assert train['models']==ev['models']
cp={}
for k,v in train['models'].items():
 task,seed=k.split('/');q=r/'training/job_7485/predictor'/task/('seed_'+seed)
 cp[k]={}
 for key,file in [('best_sha256','best.pt'),('latest_sha256','latest.pt')]:
  if key in v:
   h=hashlib.sha256((q/file).read_bytes()).hexdigest();assert h==v[key];cp[k][key]=h
files={str(p.relative_to(r)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in e.rglob('*') if p.is_file() and p.suffix in ['.json','.csv']}
summary=[]
for task in ['gate','motion']:
 rows=list(csv.DictReader((e/task/'evaluation/raw_records.csv').open()));assert len(rows)==34560
 groups=collections.defaultdict(list)
 for x in rows:
  for key in ['normalized_mse','latent_mse','pair_accuracy','state_accuracy','state_position_mse','evidence_hit']:
   assert math.isfinite(float(x[key]))
  groups[(task,x['seed'],x['split'],x['condition'],x['horizon'])].append(x)
 for key,rr in sorted(groups.items()):
  d=dict(zip(['task','seed','split','condition','horizon'],key));d['records']=len(rr)
  for m in ['normalized_mse','latent_mse','pair_accuracy','state_accuracy','state_position_mse','evidence_hit']: d[m]=sum(float(x[m]) for x in rr)/len(rr)
  summary.append(d)
(e/'final_condition_means.csv').write_text('')
with (e/'final_condition_means.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
proof={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sacct':sac,'science_source_guards':locks,'all_six_GPU_CPU_proofs_equal':True,'current_checkpoints':cp,'files':files,'scientific_status':o['status'],'phase1_go':False,'fresh_confirmation_opened':False,'independent_audit_status':a['status'],'no_model_forward':True,'GPU_used_seconds':6247,'GPU_remaining_seconds':11753}
(r/'final_native_verification_7506.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps({'native_jobs':jobs,'files':{k:v['bytes'] for k,v in files.items()},'status':o['status'],'all_source_locks_pass':True}))
