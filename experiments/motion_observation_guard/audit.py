"""Separate gate replay, complete-vector QR projection and raw-fork statistics."""
import csv, hashlib, json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import legacy_independent as legacy
ROOT=Path(__file__).resolve().parent;OLD=Path('/scratch/xl4358/phase1_motion_feature_deletion_validation/output/job_8339');AGE=Path('/scratch/xl4358/phase1_age_control_a1')
sys.path.insert(0,str(AGE));import run as old
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def rows(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def main(out):
    out=Path(out);s=read(out/'summary.json');raw=rows(out/'raw_guard.csv');paired=rows(out/'paired_guard.csv');decisions=rows(out/'observed_decisions.csv')
    assert len(raw)==62208 and len(paired)==1728 and sha(out/'raw_guard.csv')==s['raw_sha256'] and sha(out/'paired_guard.csv')==s['paired_sha256'] and sha(out/'observed_decisions.csv')==s['decision_sha256']
    proofs={arm:legacy.audit_rows([r for r in raw if r['arm']==arm],[r for r in paired if r['arm']==arm],s['summaries'][arm]['statistics'],pairs=96,repeats=10000) for arm in ('baseline','unconditional','observed_guard')}
    _,_,configs=old.verify_control(old.OLD/'control');c=configs['motion'];store=old.FeatureStore(c,max_cached=192)
    recs={r['id']:r for r in store.manifest['records'] if r['task']=='motion' and r['split']=='validation'}
    decoded={};gate={}
    for d in decisions:decoded.setdefault(d['episode'],[]).append(d)
    for eid,rr in decoded.items():
        rec=recs[eid];q=int(rec['query']);rgb=old.load_episode(c,rec,keys=('rgb',))['rgb'];n=rgb.shape[1]
        cur=rgb[q,int(.30*n):int(.73*n),int(.10*n):int(.90*n)]
        for d in rr:
            kept=tuple(map(int,d['kept'].split(';')));assert len(kept)==2 and max(kept)<q-3
            allow=all(np.count_nonzero(rgb[t,int(.30*n):int(.73*n),int(.10*n):int(.90*n)]!=cur)>0 for t in kept)
            assert allow==(d['allow_update']=='True');gate[eid,kept]=allow
    decoder=old.LinearReadout.load(Path(c['output'])/'probes/motion_state_decoder.npz')
    with np.load(Path(c['output'])/'predictor/motion/basis.npz',allow_pickle=False) as f:B=f['output_basis'].astype(np.float64)
    A=decoder.weight[:,1:]/decoder.std[:,None];b=decoder.bias[1:]-decoder.mean@A;Q,_=np.linalg.qr(B,mode='reduced');R=np.linalg.pinv(Q.T@A)@Q.T
    parent=read(OLD/'summary.json');assert sha(OLD/'baseline_vectors.npz')==parent['vectors_sha256']==s['parent_vector_sha256']
    with np.load(OLD/'baseline_vectors.npz',allow_pickle=False) as f:pred=f['predictions']
    by=defaultdict(dict)
    for r in raw:
        ix=int(r['witness_index']);assert r['arm'] not in by[ix];by[ix][r['arm']]=r
    assert set(by)==set(range(20736)) and pred.shape==(20736,5120)
    truthcache={};maximum=0.;fallbacks=0;applied=0
    for start in range(0,20736,64):
        rr=[by[i]['baseline'] for i in range(start,min(start+64,20736))];gg=np.array([[float(r['geometry_x']),float(r['geometry_y'])] for r in rr]);p=pred[start:start+len(rr)].astype(np.float64)
        corrected=p+(gg-(p@A+b))@R;enabled=np.array([gate[r['episode'],tuple(map(int,r['kept'].split(';')))] for r in rr]);guarded=np.where(enabled[:,None],corrected,p)
        assert np.array_equal(guarded[~enabled],p[~enabled]);fallbacks+=int((~enabled).sum());applied+=int(enabled.sum());targets=[];positions=[]
        for r in rr:
            rec=recs[r['episode']];q=int(rec['query']);h=int(r['horizon']);targets.append(store.features(rec)[q+h])
            if rec['id'] not in truthcache:truthcache[rec['id']]=old.load_episode(c,rec,True,keys=('states',))['states']
            positions.append(truthcache[rec['id']][q+h])
        for arm,v in [('baseline',p),('unconditional',corrected),('observed_guard',guarded)]:
            loss=np.mean((v-np.asarray(targets))**2,axis=1);task=np.mean((v@A+b-np.asarray(positions))**2,axis=1)
            assert np.isfinite(v).all()
            for j in range(len(rr)):
                r=by[start+j][arm];assert (r['allow_update']=='True')==enabled[j]
                maximum=max(maximum,abs(float(loss[j])-float(r['feature_mse'])),abs(float(task[j])-float(r['task_error'])))
                assert maximum<1e-7
    # Independently reconstruct paired mean differences and every incremental CI.
    statmax=0.
    for cell in s['changes']:
        X=[]
        for arm in ('baseline',cell['arm']):
            selected=[r for r in paired if r['arm']==arm and int(r['horizon'])==cell['horizon'] and (cell['seed']=='mean' or int(r['seed'])==cell['seed'])]
            X.append(np.array([np.mean([float(r['uniform_'+cell['metric']]) for r in selected if r['pair']==p]) for p in sorted({r['pair'] for r in selected})]))
        X=np.stack(X);ix=np.random.default_rng(0).integers(0,96,size=(10000,96));m=X[:,ix].mean(2);lo,hi=np.quantile((m[0]-m[1])/m[0],[.025,.975]);effect=(X[0].mean()-X[1].mean())/X[0].mean()
        statmax=max(statmax,abs(effect-cell['relative_reduction']),abs(lo-cell['low']),abs(hi-cell['high']));assert statmax<1e-12
    assert all(sha(k)==v for k,v in read(out/'asset_ledger.json').items())
    assert all(sha(ROOT/k)==v for k,v in read(ROOT/'source_lock.json')['files'].items())
    result=dict(status='PASS',raw_rows=len(raw),paired_rows=len(paired),observed_decisions_replayed=len(gate),complete_vectors_recomputed=20736,independent_QR_projection=True,maximum_scalar_difference=maximum,incremental_statistics=len(s['changes']),incremental_statistic_max_difference=statmax,independent_scalar_statistics=sum(x['independent_scalar_statistics'] for x in proofs.values()),per_arm=proofs,guard_applied_witnesses=applied,fallback_exact_witnesses=fallbacks,no_predictor_or_encoder_forward=True,training=False,fitting=False,GPU=0,original_gates_changed=False,original_Phase1_GO=False)
    (out/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main(sys.argv[1])
