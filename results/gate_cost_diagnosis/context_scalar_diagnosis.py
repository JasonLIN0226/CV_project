"""Post-hoc context interaction counts from old8275 scalars; no model or fitting."""
from pathlib import Path
import collections,csv,hashlib,json,sys,datetime
P=Path('/scratch/xl4358/phase1_age_control_a1/output/job_8275');OUT=Path('/scratch/xl4358/phase1_gate_cost_diagnosis/output/job_8282')
raw=P/'raw_deletions.csv';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(raw)=='cc53f6cd73f9fcaed5e6a25f50edf46257fb652eafd5625b06d000297ab15a45'
cells=collections.defaultdict(lambda:collections.defaultdict(list))
with raw.open() as f:
 for r in csv.DictReader(f):
  if r['task']!='gate':continue
  k=(int(r['seed']),r['split'],r['pool'],r['pair'],int(r['horizon']),int(r['evidence']))
  cells[k][int(r['deleted'])].append((float(r['feature_mse']),float(r['task_error'])))
results=[]
for (seed,split,group,pair,h,cue),forks in cells.items():
 avg={d:(sum(x[0] for x in values)/2,sum(x[1] for x in values)/2) for d,values in forks.items()};assert len(avg)==3 and all(len(v)==2 for v in forks.values())
 cf,ct=avg[cue];other=[v for d,v in avg.items() if d!=cue];helpful=[v for v in other if v[1]<ct-1e-8]
 results.append(dict(seed=seed,split=split,pool_group=group,pair=pair,horizon=h,task_help_available=int(bool(helpful)),joint_help_available=int(any(f<cf-1e-8 for f,t in helpful)),all_task_help_increases_feature=int(bool(helpful) and all(f>cf+1e-8 for f,t in helpful)),
  feature_context_sign_flip=int(min(v[0] for v in other)<cf-1e-8 and max(v[0] for v in other)>cf+1e-8),
  task_context_sign_flip=int(min(v[1] for v in other)<ct-1e-8 and max(v[1] for v in other)>ct+1e-8),
  helpful_vs_unhelpful_context=int(min(v[1] for v in other)<ct-1e-8 and max(v[1] for v in other)>=ct-1e-8),
  lower_feature_other_also_task_help=int(min(other,key=lambda x:x[0])[1]<ct-1e-8),
  cue_is_unique_feature_min=int(all(cf<v[0]-1e-8 for v in other)),
  critical_feature=cf,mean_other_feature=sum(v[0] for v in other)/2,critical_task=ct,mean_other_task=sum(v[1] for v in other)/2,
  other_feature_range=abs(other[0][0]-other[1][0]),other_task_range=abs(other[0][1]-other[1][1])))
assert len(results)==6912
COUNT_FIELDS=('task_help_available','joint_help_available','all_task_help_increases_feature','feature_context_sign_flip','task_context_sign_flip','helpful_vs_unhelpful_context','lower_feature_other_also_task_help','cue_is_unique_feature_min')
def summarize(rr):return dict(rows=len(rr),**{k:sum(r[k] for r in rr) for k in COUNT_FIELDS},mean_cue_retaining_feature_range=sum(r['other_feature_range'] for r in rr)/len(rr),mean_cue_retaining_task_range=sum(r['other_task_range'] for r in rr)/len(rr))
proof=dict(status='POSTHOC_EXISTING_SCALAR_CONTEXT_DIAGNOSIS_COMPLETE',scope='Pair/seed/pool/horizon records; correlated checks, not independent sample count',overall=summarize(results),by_condition={sp:summarize([r for r in results if r['split']==sp]) for sp in ('test_id','test_long_gap','test_distractor_shift')},by_position={g:summarize([r for r in results if r['pool_group']==g]) for g in ('cue_oldest','cue_middle','cue_newest')},source_sha256=sha(__file__),raw_sha256=sha(raw),new_model_forward=False,training=False,GPU=0,original_gates_unchanged=True,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
old=json.loads((OUT/'summary.json').read_text())
assert all(proof['overall'][k]==v for k,v in old['pareto_tradeoff'].items())
(OUT/'context_scalar_diagnosis.json').write_text(json.dumps(proof,indent=2)+'\n')
with (OUT/'context_pairs.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(results[0]),lineterminator='\n');w.writeheader();w.writerows(results)
print(json.dumps(proof),flush=True)
