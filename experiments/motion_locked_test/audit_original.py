"""Independent scalar/vector audit. No model forward, fitting or selection."""
import collections,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
from verify_fixed import audit as fixed_audit
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

def audit_original(folder):
 O=Path(folder);R=Path(__file__).parent;s=read(O/'summary.json');prediction=s['prediction']
 assert sha(O/'original/evaluation/raw_records.csv')==s['raw_prediction_sha256'] and sha(O/'raw_deletions.csv')==s['raw_deletion_sha256'] and sha(O/'paired_deletions.csv')==s['paired_deletion_sha256']
 # Existing independent32fixed-reference implementation, byte-preserved formula.
 f=fixed_audit(O/'original');assert len(f['tasks']['motion']['paired_checks'])==32
 raw=csvread(O/'original/evaluation/raw_records.csv');assert len(raw)==34560
 config=read('/scratch/xl4358/phase1_v3_dev/motion/reference/resolved_config.json');rule=config['decision'];assert config['evaluation']['bootstrap']==2000
 metrics=read(O/'original/evaluation/metrics.json');probes=read('/scratch/xl4358/phase1_v3_dev/motion/reference/probes/metrics.json');dataaudit=read('/scratch/xl4358/phase1_v3_dev/motion/reference/dataset/split_audit.json');assert dataaudit['passed']
 # Rebuild all original comparative statistics from raw episode rows.
 index={(int(r['seed']),r['split'],r['condition'],int(r['horizon']),r['episode']):r for r in raw};assert len(index)==34560
 allchecks=[];maxdiff=0.;full_ok=True;structural=True
 for probe in probes['results']:
  if probe['task']=='motion' and probe['split'].startswith('test_'):
   if probe['variant']=='motion_two_frame':allchecks.append(dict(kind='evidence_probe',task='motion',split=probe['split'],passed=bool(probe['accuracy']['mean']>=rule['probe_accuracy'] and probe['wilson_effective_pairs'][0]>.5)))
   elif probe['variant']=='recent_only':assert abs(probe['accuracy']['mean']-.5)<1e-7
   elif probe['variant']=='state_decoder_on_true_target':assert np.sqrt(probe['position_mse']['mean'])<=rule['readout_position_rmse']
 for comparison in metrics['comparisons']:
  sp,h,seed,alt=comparison['split'],comparison['horizon'],comparison['seed'],comparison['alternative']
  if not sp.startswith('test_'):continue
  seeds=(17,29,43) if seed=='all_seeds_mean' else (int(seed),)
  episodes=sorted({r['episode'] for r in raw if r['split']==sp});assert len(episodes)==256
  def vals(condition,field):return np.mean([[float(index[(se,sp,condition,h,ep)][field]) for ep in episodes] for se in seeds],0)
  pairs=[index[(17,sp,'recent_only',h,ep)]['pair'] for ep in episodes]
  name=f'motion/{sp}/h{h}/seed{seed}'
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
    l,t=got['normalized_mse'],got['state_position_mse'];lt=rule['absolute_latent_gain'] if l[3] else rule['latent_relative_improvement'];tt=rule['absolute_position_mse_gain'] if t[3] else rule['state_relative_improvement']
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
 return dict(original_checks=51,fixed_checks=32,max_original_statistic_difference=maxdiff,full_history_rule_met=bool(full_ok),structural_rules_met=bool(structural),possible_go=bool(possible_go))
