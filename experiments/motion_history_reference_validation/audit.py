"""Independent QR output reconstruction and paired bootstrap; no fitted parameters."""
import csv,json,hashlib,sys
from pathlib import Path
import numpy as np
from core import maps
REF=Path('/scratch/xl4358/phase1_v3_dev/motion/reference')
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(folder):
 out=Path(folder);s=json.loads((out/'summary.json').read_text());assert sha(out/'scalar_rows.csv')==s['scalar_sha256'] and sha(out/'vectors.npz')==s['vectors_sha256']
 with np.load(out/'vectors.npz',allow_pickle=False) as f:recent=f['recent'];current=f['current'];candidate=f['candidate'];reference=f['reference_history'];target=f['target'];truth=f['truth']
 with (Path('/scratch/xl4358/phase1_motion_geometry_validation/output/job_8336')/'validation_geometry.csv').open() as f:rows=list(csv.DictReader(f))
 with np.load(REF/'predictor/motion/basis.npz',allow_pickle=False) as f:B=f['output_basis']
 with np.load(REF/'probes/motion_state_decoder.npz',allow_pickle=False) as f:d={k:f[k] for k in f.files}
 B,A,b,_,_=maps(B,d['weight'],d['std'],d['mean'],d['bias']);assets=json.loads((out/'asset_ledger.json').read_text())
 with (REF/'evaluation/raw_records.csv').open() as f:gold={(int(r['seed']),r['episode'],int(r['horizon'])):r for r in csv.DictReader(f) if r['split']=='validation' and r['condition']=='privileged_small_budget'}
 for i,r in enumerate(rows):
  v=gold[int(r['seed']),r['episode'],int(r['horizon'])];assert abs(np.mean((reference[i]-target[i])**2)-float(v['normalized_mse']))<2e-6
 # Cached old_history vectors are read directly from the audited parent.
 with np.load('/scratch/xl4358/phase1_motion_geometry_validation/output/job_8336/prediction_witnesses.npz',allow_pickle=False) as f:hist=f['predictions'][:,1].astype(float);assert np.array_equal(target,f['targets'].astype(float))
 Q=np.linalg.qr(B)[0];F=Q.T@A;dual=np.linalg.solve(F.T@F,F.T)
 independent=reference+((current-reference)@A)@dual@Q.T
 diff=float(np.max(abs(independent-candidate)));assert diff<1e-10
 with np.load(out/'vectors.npz',allow_pickle=False) as f:
  assert np.array_equal(f['candidate'],candidate) and np.array_equal(f['target'],target);truth=f['truth']
 with (out/'scalar_rows.csv').open() as f:raw=list(csv.DictReader(f))
 assert len(raw)==5760
 for arm,z in [('recent',recent),('old_history',hist),('current',current),('reference_history',reference),('candidate',independent)]:
  rr=[r for r in raw if r['arm']==arm];assert len(rr)==1152
  assert np.max(abs(np.array([float(r['feature_mse']) for r in rr])-((z-target)**2).mean(1)))<1e-12
  assert np.max(abs(np.array([float(r['position_mse']) for r in rr])-((z@A+b-truth)**2).mean(1)))<1e-12
 maximum=0.;checks=0
 for cell in s['cells']:
  h,se=cell['horizon'],cell['seed'];ids=[i for i,r in enumerate(rows) if int(r['horizon'])==h and (se=='mean' or int(r['seed'])==se)];pairs=sorted({rows[i]['pair'] for i in ids})
  assert len(pairs)==96
  ix=np.random.default_rng(0).integers(0,96,(2000,96))
  for name,base,metric,floor,absolute in [('feature_vs_recent',recent,'f',.02,.01),('position_vs_recent',recent,'p',1e-5,1e-6),('feature_vs_current',current,'f',.02,.01),('position_vs_current',current,'p',1e-5,1e-6)]:
   def loss(z):return ((z-target)**2).mean(1) if metric=='f' else ((z@A+b-truth)**2).mean(1)
   L,R=loss(base),loss(independent);a=np.array([[np.mean([L[i] for i in ids if rows[i]['pair']==p]),np.mean([R[i] for i in ids if rows[i]['pair']==p])] for p in pairs]);isabs=a[:,0].mean()<floor
   def effect(v):
    x=v[...,0].mean(-1);y=v[...,1].mean(-1);return x-y if isabs else (x-y)/np.maximum(x,floor)
   lo,hi=np.quantile(effect(a[ix]),[.025,.975]);value=float(effect(a));w=cell[name];d=max(abs(value-w['mean']),abs(lo-w['low']),abs(hi-w['high']));maximum=max(maximum,d);assert d<1e-11
   assert w['numeric_rule_met']==bool(value>=(absolute if isabs else .1) and lo>0);checks+=1
 assert checks==32 and all(sha(k)==v for k,v in assets.items()) and all(sha(ROOT/k)==v for k,v in json.loads((ROOT/'source_lock.json').read_text())['files'].items())
 result=dict(status='PASS_INDEPENDENT_QR_AND_SCALAR_AUDIT',source_rows=1152,scalar_rows=5760,intervals=32,vector_max_difference=diff,interval_max_difference=maximum,validation_eligible=s['validation_eligible'],GPU=0,training=False,original_phase1_go=False)
 (out/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main(sys.argv[1])
