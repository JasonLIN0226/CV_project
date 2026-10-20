"""Independent original/fixed gates, fork axes, observed inputs and vector replay."""
import csv,hashlib,json,sys,collections,functools
from pathlib import Path
import numpy as np
from audit_original import audit_original
from deletion_independent import audit_rows
ROOT=Path(__file__).resolve().parent;AGE=Path('/scratch/xl4358/phase1_age_control_a1');sys.path.insert(0,str(AGE));import run as old
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def rows(p):
 with Path(p).open() as f:return list(csv.DictReader(f))
def main(folder):
 out=Path(folder);s=read(out/'summary.json');original=audit_original(out)
 raw=rows(out/'raw_deletions.csv');paired=rows(out/'paired_deletions.csv');assert len(raw)==82944 and len(paired)==2304
 assert sha(out/'paired_deletions.csv')==s['paired_deletion_sha256'];deletion=audit_rows(raw,paired,s['deletion'],pairs=128,repeats=10000)
 baseline={(r['seed'],r['split'],r['episode'],r['pool'],r['deleted'],r['horizon']):r for r in rows(AGE/'output/job_8275/raw_deletions.csv') if r['task']=='motion'}
 for r in raw:
  b=baseline[r['seed'],r['split'],r['episode'],r['pool'],r['deleted'],r['horizon']];assert float(r['old_feature_mse'])==float(b['feature_mse']) and float(r['old_task_error'])==float(b['task_error'])
  # Current geometry guard fallback is preserved in position, while reference-history visual output remains the fixed base.
 originalraw=rows(out/'original/evaluation/raw_records.csv');parentraw=rows(old.OLD/'evaluation/job_7575/motion/evaluation/raw_records.csv');parentindex={tuple(r[k] for k in ['seed','split','condition','episode','horizon']):r for r in parentraw}
 for r in originalraw:
  if r['condition'] in ('recent_only','full_history','persistence'):
   b=parentindex[tuple(r[k] for k in ['seed','split','condition','episode','horizon'])]
   assert set(r)==set(b) and all(r[k]==b[k] if k in ('task','split','condition','episode','pair','sibling') else float(r[k])==float(b[k]) for k in r)
 _,_,cc=old.verify_control(old.OLD/'control');c=cc['motion'];store=old.FeatureStore(c,max_cached=8);recs=store.records
 trace=rows(out/'input_trace.csv');assert sha(out/'input_trace.csv')==s['trace_sha256'];traceby=collections.defaultdict(list);lookup={}
 for r in trace:traceby[r['episode']].append(r);lookup[r['episode'],r['condition'],r['kept']]=r
 @functools.lru_cache(maxsize=4)
 def obs(eid):return old.load_episode(c,recs[eid],keys=('rgb','actions'))
 for eid,rr in traceby.items():
  q=int(recs[eid]['query']);own=obs(eid);n=own['rgb'].shape[1];ys=slice(int(.30*n),int(.73*n));xs=slice(int(.10*n),int(.90*n));cur=own['rgb'][q,ys,xs]
  for r in rr:
   kept=tuple(map(int,r['kept'].split(';')));sources=r['source_ids'].split(';');assert len(kept)==len(sources)==2 and max(kept)<q-3
   if r['condition']!='wrong_evidence':assert sources==[eid,eid]
   else:
    ev=set(map(int,old.load_episode(c,recs[eid],True,keys=('evidence',))['evidence']));assert sources==[recs[eid]['sibling'] if t in ev else eid for t in kept]
   enabled=all(np.any(obs(src)['rgb'][t,ys,xs]!=cur) for src,t in zip(sources,kept));assert enabled==(r['enabled']=='True')
 decoder=old.LinearReadout.load(Path(c['output'])/'probes/motion_state_decoder.npz');A=decoder.weight[:,1:]/decoder.std[:,None];b=decoder.bias[1:]-decoder.mean@A
 with np.load(Path(c['output'])/'predictor/motion/basis.npz',allow_pickle=False) as f:Q,_=np.linalg.qr(f['output_basis'].astype(np.float64),mode='reduced')
 R=np.linalg.pinv(Q.T@A)@Q.T
 assert sha(out/'witness_vectors.npz')==s['witness_sha256'] and sha(out/'witness_metadata.csv')==s['witness_metadata_sha256'];meta=rows(out/'witness_metadata.csv')
 with np.load(out/'witness_vectors.npz',allow_pickle=False) as f:P=f['baseline'];C=f['prediction'];T=f['target'];Y=f['position'];H=f['reference_history'];assert np.array_equal(H,.5*(f['reference_best'].astype(float)+f['reference_latest'].astype(float)))
 assert P.shape==C.shape==T.shape==(468,5120) and Y.shape==(468,2) and np.isfinite(C).all();maxvector=0.;maxscalar=0.;maxgeometry=0.
 def roll(xy,v,actions):
  xy=np.array(xy,dtype=float)
  for a in actions:
   x=xy[0]+v+float(np.dot(a,[.0015,.001]))
   if x<.08 or x>.92:
    band=int(np.floor((x-.08)/.84));z=(x-.08)%1.68;x=.08+(z if z<=.84 else 1.68-z)
    if band%2:v=-v
   xy[0]=x
  return xy
 for i,m in enumerate(meta):
  eid=m['episode'];q=int(recs[eid]['query']);h=int(m['horizon']);tr=lookup[eid,m['condition'],m['kept']];times=tuple(map(int,m['kept'].split(';')));src=tr['source_ids'].split(';');z=np.stack([store.features(recs[e])[t] for e,t in zip(src,times)]);xy=decoder.predict(z)[:,1:];v=(xy[1,0]-xy[0,0])/(times[1]-times[0]);geo=roll(xy[1],v,obs(eid)['actions'][times[1]:q+h]);maxgeometry=max(maxgeometry,float(np.max(np.abs(geo-[float(m['geometry_x']),float(m['geometry_y'])]))));assert maxgeometry<1e-8
  enabled=m['enabled']=='True';desired=geo if enabled else P[i].astype(np.float64)@A+b;want=H[i].astype(np.float64)+(desired-(H[i].astype(np.float64)@A+b))@R;maxvector=max(maxvector,float(np.max(np.abs(want-C[i]))));assert maxvector<1e-7
  # Baseline/fallback use parent float32 feature arithmetic; active correction is float64.
  pred=C[i];loss=float(np.mean((pred-T[i])**2));task=float(np.mean((pred@A+b-Y[i])**2));maxscalar=max(maxscalar,abs(loss-float(m['feature_mse'])),abs(task-float(m['task_error'])));assert maxscalar<2e-6
  assert np.array_equal(T[i],store.features(recs[eid])[q+h])
  true=old.load_episode(c,recs[eid],True,keys=('states',))['states'][q+h];assert np.array_equal(true,Y[i])
 assert all(sha(k)==v for k,v in read(out/'asset_ledger.json').items()) and all(sha(ROOT/k)==v for k,v in read(ROOT/'source_lock.json')['files'].items())
 assert s['state_proofs']==read(ROOT/'candidate_lock.json')['state_digest_by_seed']
 assert s['reference_final_state_proofs']==read(ROOT/'candidate_lock.json')['reference_final_state_digest_by_seed']
 repaired=original['possible_go'] and all(x['passed'] for x in s['prediction']['fixed_reference_checks']) and all(x['positive_both_metrics'] for x in s['deletion']['oracle_baseline_comparisons'] if x['choice']=='oracle_transfer')
 assert repaired==(s['status']=='MOTION_DEVELOPMENT_REPAIR_SUPPORTED')
 result=dict(status='PASS_INDEPENDENT_LOCKED_MOTION_AUDIT',scientific_status=s['status'],original=original,deletion=deletion,all_baseline_deletion_rows_replayed=82944,unchanged_prediction_rows_copied_exactly=17280,observed_guard_decisions_replayed=len(trace),primitive_vectors=468,vector_max_difference=maxvector,geometry_max_difference=maxgeometry,scalar_max_difference=maxscalar,assets_current=True,GPU=0,training=False,fitting=False,original_gates_changed=False,original_Phase1_GO=False,fresh9137_opened=False)
 (out/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main(sys.argv[1])
