"""Independent raw pairing, scalar intervals and primitive witness audit."""
import collections,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(folder):
 out=Path(folder);root=Path(__file__).parent;s=json.loads((out/'summary.json').read_text());assert s['raw_sha256']==sha(out/'raw_validation.csv') and s['paired_sha256']==sha(out/'paired_validation.csv') and s['witness_sha256']==sha(out/'vector_witnesses.npz')
 with (out/'raw_validation.csv').open() as f:raw=list(csv.DictReader(f))
 with (out/'paired_validation.csv').open() as f:paired=list(csv.DictReader(f))
 assert len(raw)==14976 and len(paired)==7488 and {r['split'] for r in raw}=={'validation'}
 keys={(r['seed'],r['pair'],r['episode'],r['variant'],r['horizon']) for r in raw};assert len(keys)==14976
 groups=collections.defaultdict(list)
 for r in raw:
  assert np.isfinite([float(r['feature_mse']),float(r['task_error'])]).all() and float(r['feature_mse'])>=0 and float(r['task_error']) in (0.,1.)
  kept=tuple(map(int,r['kept'].split(';'))) if r['kept'] else ();assert len(set(kept))==int(r['history_count']) and tuple(sorted(kept))==kept
  assert all(0<=t<int(r['query'])-3 for t in kept) and int(r['cue_present'])==int(int(r['cue']) in kept)
  if r['family']=='deletion':assert len(kept)==2 and int(r['deleted']) not in kept and set(kept)|{int(r['deleted'])}==set(map(int,r['parent'].split(';')))
  groups[(int(r['seed']),r['pair'],r['variant'],int(r['horizon']))].append(r)
 pi={(int(r['seed']),r['pair'],r['variant'],int(r['horizon'])):r for r in paired};assert len(pi)==7488
 maxmean=0.
 for k,rr in groups.items():
  assert len(rr)==2 and len({r['episode'] for r in rr})==2
  pp=pi[k]
  for m in ('feature_mse','task_error'):maxmean=max(maxmean,abs(float(pp[m])-sum(float(r[m]) for r in rr)/2))
  assert abs(float(pp['total'])-float(pp['common'])-float(pp['contrast']))<1e-12 and abs(float(pp['total'])-float(pp['feature_mse']))<1e-7
 assert maxmean==0 and s['physical_delete_path_input_equal']==5184 and s['native_privileged_anchor_input_equal']==576 and s['max_path_forward_difference']==0 and s['max_anchor_forward_difference']<3e-5
 assert s['max_prior_validation_MSE_difference']<=1e-7 and s['validation_pairs']==96
 with np.load(out/'vector_witnesses.npz',allow_pickle=False) as f:
  P,T,M=f['prediction'].astype(np.float64),f['target'].astype(np.float64),f['metrics'];assert P.shape==T.shape==(78,2,5120) and M.shape==(78,4)
 err=P-T;v=np.stack([np.mean(err**2,axis=(1,2)),np.mean(err.mean(1)**2,axis=1),np.mean(((err[:,0]-err[:,1])/2)**2,axis=1),np.max(np.abs(P[:,0]-P[:,1]),axis=1)],axis=1)
 witness_diff=float(np.max(np.abs(v-M)));assert witness_diff<1e-12
 ids=sorted({r['pair'] for r in paired});assert len(ids)==96
 rngix=np.random.default_rng(0).choice(96,size=(10000,96),replace=True);checks=0;maxci=0.
 def vector(seed,h,f,choose):
  seeds=(17,29,43) if seed=='mean' else (seed,)
  return np.mean([[choose(si,p,h,f) for p in ids] for si in seeds],axis=0)
 def interval(v):return np.percentile(v[rngix].mean(1),[2.5,97.5])
 for cell in s['cells']:
  seed,h,g=cell['seed'],cell['horizon'],cell['pool_group']
  for m,sv in cell['metrics'].items():
   def side(si,p,hh,f,crit):
    rr=[pi[(si,p,f'{g}/delete_{j}',hh)] for j in range(3)];rr=[r for r in rr if bool(int(r['cue_present']))!=crit];assert len(rr)==(1 if crit else 2);return np.mean([float(r[f]) for r in rr])
   ca=vector(seed,h,m,lambda si,p,hh,f:side(si,p,hh,f,True));ot=vector(seed,h,m,lambda si,p,hh,f:side(si,p,hh,f,False));lo,hi=interval(ca-ot);rel=(ca.mean()-ot.mean())/max(ca.mean(),1e-8);flag=bool(ca.mean()>=1e-8 and rel>=.1 and lo>0)
   assert flag==sv['meets_original_numeric_conditions'];maxci=max(maxci,abs(float(lo)-sv['difference']['low']),abs(float(hi)-sv['difference']['high']));checks+=1
  assert cell['both_numeric_conditions']==all(cell['metrics'][f]['meets_original_numeric_conditions'] for f in ('feature_mse','task_error'))
 for cell in s['comparisons']:
  seed,h,g=cell['seed'],cell['horizon'],cell['pool_group']
  for m,v in cell['nonnative_minus_native_penalty'].items():
   def value(si,p,hh,f):
    rr=[pi[(si,p,f'{g}/delete_{j}',hh)] for j in range(3)];rr=[r for r in rr if int(r['cue_present']) and not int(r['native_privileged'])];assert len(rr)==(1 if g=='cue_oldest' else 2)
    return np.mean([float(r[f]) for r in rr])-float(pi[(si,p,'privileged_small_budget',hh)][f])
   a=vector(seed,h,m,value);lo,hi=interval(a);maxci=max(maxci,abs(float(lo)-v['low']),abs(float(hi)-v['high']));checks+=1
 # Original validation means must exactly reproduce previous scores up to execution tolerance.
 prior=Path('/scratch/xl4358/phase1_gate_calibration/output/job_8289/validation_scores.csv')
 with prior.open() as f:gold={(int(r['seed']),r['episode'],r['condition'],int(r['horizon'])):float(r['feature_mse']) for r in csv.DictReader(f) if float(r['alpha'])==1.}
 maxreplay=max(abs(float(r['feature_mse'])-gold[(int(r['seed']),r['episode'],r['variant'],int(r['horizon']))]) for r in raw if r['family']=='original');assert maxreplay<=1e-7 and maxci<1e-12
 assert all(sha(p)==v for p,v in json.loads((out/'asset_ledger.json').read_text()).items()) and all(sha(root/k)==v for k,v in json.loads((root/'source_lock.json').read_text())['files'].items())
 a=dict(status='PASS_INDEPENDENT_FROZEN_VALIDATION_AUDIT',summary_sha256=sha(out/'summary.json'),raw_sha256=s['raw_sha256'],paired_sha256=s['paired_sha256'],raw_rows=14976,paired_rows=7488,independent_intervals=checks,max_CI_difference=maxci,primitive_witness_pairs=78,max_witness_difference=witness_diff,max_original_validation_MSE_difference=maxreplay,exact_physical_delete_input_paths=5184,native_anchor_input_paths=576,no_training=True,no_fitting=True,GPU=0,original_gates_changed=False,original_Phase1_GO=False)
 (out/'independent_audit.json').write_text(json.dumps(a,indent=2)+'\n');print(json.dumps(a),flush=True)
if __name__=='__main__':main(sys.argv[1])
