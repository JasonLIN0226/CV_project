from pathlib import Path
import csv,json,hashlib,datetime
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
root=ROOT/'runs/phase1_v5_gate/evaluation_results'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
refpath=ROOT/'runs/phase1_v3_dev/gate/reference/evaluation/raw_records.csv'
assert sha(refpath)=='2dace4c092e3189f9fae4c544719850f0a4a379f0e90fe8f21382911a4599ecd'
key=lambda r:tuple(r[k] for k in ('seed','task','split','horizon','condition','pair','episode'))
def read(path):
 with Path(path).open() as f:rows=list(csv.DictReader(f))
 values={key(r):r for r in rows}
 assert len(rows)==len(values)==34560
 assert all(np.isfinite(float(r['normalized_mse'])) for r in rows)
 return rows,values
rr,reference=read(refpath);raw,candidate=read(root/'evaluation/raw_records.csv');assert reference.keys()==candidate.keys()
comparison=json.loads((root/'development_comparison.json').read_text())
results=[];maximum=0.
for saved in comparison['fixed_reference_checks']:
 task,split,h,seed=saved['slice'].split('/');h=h[1:];seed=seed[4:]
 selected=[r for r in raw if r['split']==split and r['horizon']==h and r['condition']=='privileged_small_budget' and (seed=='all_seeds_mean' or r['seed']==seed)]
 grouped={}
 for r in selected:
  k=key(r);recent=list(k);recent[4]='recent_only'
  grouped.setdefault((r['pair'],r['episode']),[]).append([float(reference[k]['normalized_mse']),float(r['normalized_mse']),float(reference[tuple(recent)]['normalized_mse'])])
 episodes=sorted(grouped);episode_values=np.array([np.mean(grouped[k],axis=0) for k in episodes])
 pair_values={}
 for k,value in zip(episodes,episode_values):pair_values.setdefault(k[0],[]).append(value)
 matrix=np.array([np.mean(v,axis=0) for v in pair_values.values()]);assert len(matrix)==128
 draws=np.random.default_rng(0).integers(0,len(matrix),(2000,len(matrix)))
 if saved['kind']=='candidate_vs_reference_recent':
  base,alt=matrix[:,2],matrix[:,1];absolute=base.mean()<.02
  def effect(a,b):return a-b if absolute else (a-b)/np.maximum(a,.02)
  mean=float(effect(base.mean(),alt.mean()));boot=effect(base[draws].mean(1),alt[draws].mean(1))
  source=saved['improvement'];passed=mean>=(.01 if absolute else .1) and np.quantile(boot,.025)>0
 else:
  values=matrix[:,0]-matrix[:,1];mean=float(values.mean());boot=values[draws].mean(1)
  source=saved['absolute_mse_reduction'];passed=mean>0 and np.quantile(boot,.025)>0
 actual={'mean':mean,'low':float(np.quantile(boot,.025)),'high':float(np.quantile(boot,.975))}
 diff=max(abs(actual[k]-source[k]) for k in actual);maximum=max(maximum,diff)
 assert diff<1e-14 and bool(passed)==saved['passed']
 results.append({'slice':saved['slice'],'kind':saved['kind'],'passed':bool(passed),'replayed':actual,'max_scalar_difference':diff})
assert len(results)==32
proof={'status':'PASS_SCALAR_REPLAY_NOT_NEW_SCIENCE','root_audit_not_delegated_review':True,
 'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'candidate_rows':len(raw),'reference_rows':len(rr),
 'complete_unique_matching_keys':True,'paired_base_pair_resampling_not_individual_siblings':True,
 'original_2000_bootstrap_repeats':True,'original_floors_and_thresholds':True,'replayed_saved_checks':len(results),
 'max_scalar_difference':maximum,'candidate_raw_sha256':sha(root/'evaluation/raw_records.csv'),
 'reference_raw_sha256':sha(refpath),'checks':results,'development_adoption':comparison['status'],'phase1_go':False}
(root/'fixed_reference_scalar_replay.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps({'replayed':len(results),'max_difference':maximum,'adoption':comparison['status']}))
