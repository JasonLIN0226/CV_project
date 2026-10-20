"""Independent scalar/vector audit. No model forward, fitting or selection."""
import collections,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
from verify_fixed import audit as fixed_audit
from deletion_stats import compute
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def csvread(p):
 with Path(p).open() as f:return list(csv.DictReader(f))
def bounds(v,n=2000):
 a=np.asarray(v,float);ix=np.random.default_rng(0).integers(0,len(a),(n,len(a)));return np.quantile(a[ix].mean(1),[.025,.975])
def improvement(base,alt,pairs,rule,floor,absolute):
 gg=collections.defaultdict(list)
 for x,y,p in zip(base,alt,pairs):gg[p].append((x,y))
 a=np.asarray([np.mean(v,0) for v in gg.values()]);useabs=a[:,0].mean()<floor
 def effect(v):
  x,y=v[...,0].mean(-1),v[...,1].mean(-1);return x-y if useabs else (x-y)/np.maximum(x,floor)
 ix=np.random.default_rng(0).integers(0,len(a),(2000,len(a)));lo,hi=np.quantile(effect(a[ix]),[.025,.975]);return float(effect(a)),float(lo),float(hi),useabs

def main(folder):
 O=Path(folder);R=Path(__file__).parent;s=read(O/'summary.json');prediction=s['prediction']
 assert sha(O/'original/evaluation/raw_records.csv')==s['raw_prediction_sha256'] and sha(O/'raw_deletions.csv')==s['raw_deletion_sha256'] and sha(O/'paired_deletions.csv')==s['paired_deletion_sha256']
 # Existing independent32fixed-reference implementation, byte-preserved formula.
 f=fixed_audit(O/'original');assert len(f['tasks']['gate']['paired_checks'])==32
 raw=csvread(O/'original/evaluation/raw_records.csv');assert len(raw)==34560
 config=read('/scratch/xl4358/phase1_v3_dev/gate/reference/resolved_config.json');rule=config['decision'];assert config['evaluation']['bootstrap']==2000
 metrics=read(O/'original/evaluation/metrics.json');probes=read('/scratch/xl4358/phase1_v3_dev/gate/reference/probes/metrics.json');dataaudit=read('/scratch/xl4358/phase1_v3_dev/gate/reference/dataset/split_audit.json');assert dataaudit['passed']
 # Rebuild all original comparative statistics from raw episode rows.
 index={(int(r['seed']),r['split'],r['condition'],int(r['horizon']),r['episode']):r for r in raw};assert len(index)==34560
 allchecks=[];maxdiff=0.;full_ok=True;structural=True
 for probe in probes['results']:
  if probe['task']=='gate' and probe['split'].startswith('test_'):
   if probe['variant']=='gate_cue':allchecks.append(dict(kind='evidence_probe',task='gate',split=probe['split'],passed=bool(probe['accuracy']['mean']>=rule['probe_accuracy'] and probe['wilson_effective_pairs'][0]>.5)))
   elif probe['variant']=='recent_only':assert abs(probe['accuracy']['mean']-.5)<1e-7
   elif probe['variant']=='state_decoder_on_true_target':assert probe['accuracy']['mean']>=rule['readout_accuracy']
 for comparison in metrics['comparisons']:
  sp,h,seed,alt=comparison['split'],comparison['horizon'],comparison['seed'],comparison['alternative']
  if not sp.startswith('test_'):continue
  seeds=(17,29,43) if seed=='all_seeds_mean' else (int(seed),)
  episodes=sorted({r['episode'] for r in raw if r['split']==sp});assert len(episodes)==256
  def vals(condition,field):return np.mean([[float(index[(se,sp,condition,h,ep)][field]) for ep in episodes] for se in seeds],0)
  pairs=[index[(17,sp,'recent_only',h,ep)]['pair'] for ep in episodes]
  name=f'gate/{sp}/h{h}/seed{seed}'
  if alt=='wrong_evidence':
   delta=vals('privileged_small_budget','pair_margin')-vals('wrong_evidence','pair_margin');gp=collections.defaultdict(list)
   for x,pa in zip(delta,pairs):gp[pa].append(x)
   v=np.array([np.mean(z) for z in gp.values()]);lo,hi=bounds(v);expected=comparison['toward_wrong_future'];maxdiff=max(maxdiff,abs(v.mean()-expected['mean']),abs(lo-expected['low']),abs(hi-expected['high']))
   allchecks.append(dict(kind='content_intervention',slice=name,passed=bool(v.mean()>=rule['wrong_evidence_gain'] and lo>0)))
  else:
   got={}
   for field in ('normalized_mse','latent_mse','state_position_mse','state_accuracy','pair_accuracy'):
    a,b=vals('recent_only',field),vals(alt,field)
    if not field.endswith('mse'):a,b=1-a,1-b
    floor=rule['position_baseline_floor'] if field=='state_position_mse' else 1e-8 if field=='latent_mse' else rule['baseline_floor']
    absolute=rule['absolute_position_mse_gain'] if field=='state_position_mse' else rule['absolute_latent_gain'] if field.endswith('mse') else rule['absolute_state_gain']
    value=improvement(a,b,pairs,rule,floor,absolute);expected=comparison['improvement'][field];maxdiff=max(maxdiff,abs(value[0]-expected['mean']),abs(value[1]-expected['low']),abs(value[2]-expected['high']));got[field]=value
   if alt=='privileged_small_budget':
    l,t=got['normalized_mse'],got['state_accuracy'];lt=rule['absolute_latent_gain'] if l[3] else rule['latent_relative_improvement'];tt=rule['absolute_state_gain'] if t[3] else rule['state_relative_improvement']
    allchecks.append(dict(kind='privileged_benefit',slice=name,passed=bool(l[0]>=lt and l[1]>0 and t[0]>=tt and t[1]>0)))
   elif alt=='full_history':full_ok=full_ok and got['normalized_mse'][0]>=-rule['full_history_tolerance']
 assert len(allchecks)==51 and maxdiff<1e-12
 def ckkey(x):return x['kind'],x.get('slice'),x.get('split')
 assert {ckkey(x):bool(x['passed']) for x in allchecks}=={ckkey(x):bool(x['passed']) for x in prediction['original_checks']}
 for r in metrics['per_seed']:
  if not r['split'].startswith('test_'):continue
  rr=[v for v in raw if int(v['seed'])==r['seed'] and v['split']==r['split'] and int(v['horizon'])==r['horizon'] and v['condition']==r['condition']];assert len(rr)==256
  fraction=np.mean([float(v['target_pair_distance'])<rule['pair_distance_floor'] for v in rr]);assert abs(fraction-r['almost_indistinguishable_fraction'])<1e-12
  if r['condition']=='recent_only':
   acc=np.mean([float(v['pair_accuracy']) for v in rr]);assert abs(acc-r['metrics']['pair_accuracy']['mean'])<1e-12;structural=structural and abs(acc-.5)<=1e-7
  structural=structural and r['almost_indistinguishable_fraction']<=.1
 possible_go=all(x['passed'] for x in allchecks) and full_ok and structural
 assert possible_go==(prediction['original_task_status']=='GO')
 # Raw deletion axes and complete causal equal-capacity forks.
 rows=csvread(O/'raw_deletions.csv');assert len(rows)==41472
 identities={(r['seed'],r['split'],r['episode'],r['pool_group'],r['deleted'],r['horizon']) for r in rows};assert len(identities)==41472
 for r in rows:
  parent=tuple(map(int,r['parent'].split(';')));kept=tuple(map(int,r['kept'].split(';')));assert len(parent)==3 and tuple(sorted(set(parent)))==parent and set(kept)==set(parent)-{int(r['deleted'])} and len(kept)==2 and min(parent)>=0 and max(parent)<int(r['query'])-3
  assert int(r['cue']) in parent and int(r['history_count'])==2 and float(r['task_error']) in (0.,1.)
  assert np.isfinite([float(r[k]) for k in ('feature_mse','task_error','old_feature_mse','old_task_error')]).all()
 pairrows,cells=compute(rows);expected={(str(x['seed']),x['split'],x['pool_group'],x['horizon']):x for x in s['deletion']['cells']};assert len(cells)==72 and len(pairrows)==6912
 for cell in cells:
  target=expected[(str(cell['seed']),cell['split'],cell['pool_group'],cell['horizon'])];assert cell['passed']==target['passed']
  for field,value in cell['metrics'].items():
   e=target['metrics'][field];assert value['passed']==e['passed'] and abs(value['low']-e['difference']['low'])<1e-12 and abs(value['high']-e['difference']['high'])<1e-12
 # Replay parent deletion raw, including all failures. No test-driven intervention.
 prior=Path('/scratch/xl4358/phase1_age_control_a1/output/job_8275/raw_deletions.csv');baseline={(r['seed'],r['split'],r['episode'],r['pool'],r['deleted'],r['horizon']):r for r in csvread(prior) if r['task']=='gate'};assert len(baseline)==41472
 for r in rows:
  old=baseline[(r['seed'],r['split'],r['episode'],r['pool_group'],r['deleted'],r['horizon'])];assert abs(float(r['old_feature_mse'])-float(old['feature_mse']))<=1e-7 and r['old_task_error']==old['task_error']
 with np.load(O/'deletion_vector_witnesses.npz',allow_pickle=False) as v:
  P,T,M=v['prediction'],v['target'],v['feature_mse'];assert P.shape==T.shape==(324,5120) and M.shape==(324,);assert np.max(np.abs(np.mean((P-T)**2,1)-M))<1e-7
 assert sha(O/'deletion_vector_witnesses.npz')==s['witness_sha256'] and s['max_branch_subtraction_residual']<3e-5 and s['content_reference_unchanged']
 assert s['original_prediction_state_proofs']==s['deletion_state_proofs'] and len(s['deletion_state_proofs'])==3
 assert all(sha(Path(path))==value for path,value in read(O/'asset_ledger.json').items()) and all(sha(R/k)==v for k,v in read(R/'source_lock.json')['files'].items())
 repaired=possible_go and all(x['passed'] for x in prediction['fixed_reference_checks']) and all(x['passed'] for x in cells)
 assert repaired==(s['status']=='GATE_DEVELOPMENT_REPAIR_SUPPORTED')
 result=dict(status='PASS_INDEPENDENT_GATE_LOCKED_TEST_AUDIT',scientific_status=s['status'],summary_sha256=sha(O/'summary.json'),raw_prediction_rows=34560,raw_deletion_rows=41472,deletion_pair_rows=6912,original_checks_independently_rebuilt=51,fixed_checks_independently_rebuilt=32,deletion_cells_independently_rebuilt=72,max_original_statistic_difference=maxdiff,primitive_prediction_vectors=324,all41472baseline_scores_replayed=True,weights_unchanged=True,full_history_rule_met=bool(full_ok),structural_rules_met=bool(structural),GPU=0,training=False,original_Phase1_GO=False)
 (O/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main(sys.argv[1])
