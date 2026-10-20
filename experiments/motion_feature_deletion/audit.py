"""Independent complete-vector and deletion-choice replay, no model forward."""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
import legacy_independent as legacy
ROOT=Path(__file__).resolve().parent;AGE=Path('/scratch/xl4358/phase1_age_control_a1');sys.path.insert(0,str(AGE));import run as old
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def csvread(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def roll(xy,v,actions):
    xy=np.asarray(xy,dtype=np.float64).copy()
    for a in actions:
        x=xy[0]+v+float(np.dot(a,[.0015,.001]))
        if x<.08 or x>.92:
            band=int(np.floor((x-.08)/.84));z=(x-.08)%1.68;x=.08+(z if z<=.84 else 1.68-z)
            if band%2:v=-v
        xy[0]=x
    return xy

def main(out):
    out=Path(out);s=read(out/'summary.json');assert sha(out/'raw_deletion_constraint.csv')==s['raw_sha256'] and sha(out/'baseline_vectors.npz')==s['vectors_sha256']
    assert sha(out/'paired_deletion_constraint.csv')==s['paired_sha256'] and sha(out/'asset_ledger.json')==s['asset_ledger_sha256']
    rows=csvread(out/'raw_deletion_constraint.csv');paired=csvread(out/'paired_deletion_constraint.csv');assert len(rows)==41472 and len(paired)==1152
    scalar={}
    for arm in ('baseline','geometry_constraint'):scalar[arm]=legacy.audit_rows([r for r in rows if r['arm']==arm],[r for r in paired if r['arm']==arm],s['summaries'][arm]['statistics'],pairs=96,repeats=10000)
    _,_,configs=old.verify_control(old.OLD/'control');c=configs['motion'];store=old.FeatureStore(c,max_cached=192)
    recs={r['id']:r for r in store.manifest['records'] if r['task']=='motion' and r['split']=='validation'}
    dp=Path(c['output'])/'probes/motion_state_decoder.npz';decoder=old.LinearReadout.load(dp)
    with np.load(Path(c['output'])/'predictor/motion/basis.npz',allow_pickle=False) as f:B=f['output_basis'].astype(np.float64)
    A=decoder.weight[:,1:]/decoder.std[:,None];b=decoder.bias[1:]-decoder.mean@A;Q,_=np.linalg.qr(B,mode='reduced');right=np.linalg.pinv(Q.T@A)@Q.T
    with np.load(out/'baseline_vectors.npz',allow_pickle=False) as f:pred=f['predictions']
    assert pred.shape==(20736,5120) and np.isfinite(pred).all()
    by={};truthcache={};geometrycache={}
    for r in rows:
        i=int(r['witness_index']);assert r['arm'] not in by.setdefault(i,{});by[i][r['arm']]=r
    assert set(by)==set(range(20736)) and all(set(x)=={'baseline','geometry_constraint'} for x in by.values())
    maximum=0.;maxgeom=0.
    for start in range(0,20736,64):
        gg=[];yy=[];tt=[]
        for i in range(start,min(start+64,20736)):
            r=by[i]['baseline'];ep=r['episode'];h=int(r['horizon']);rec=recs[ep];q=int(rec['query']);kept=tuple(map(int,r['kept'].split(';')))
            if ep not in truthcache:truthcache[ep]=old.load_episode(c,rec,True,keys=('states','visible','evidence'));truthcache[ep]['actions']=old.load_episode(c,rec,keys=('actions',))['actions']
            z=store.features(rec);truth=truthcache[ep];key=(ep,kept,h)
            if key not in geometrycache:
                visible_positions=decoder.predict(z[list(kept)])[:,1:];v=(visible_positions[1,0]-visible_positions[0,0])/(kept[1]-kept[0]);geometrycache[key]=roll(visible_positions[1],v,truth['actions'][kept[1]:q+h])
            g=geometrycache[key];gg.append(g);yy.append(z[q+h]);tt.append(truth['states'][q+h])
            maxgeom=max(maxgeom,float(np.max(np.abs(g-[float(r['geometry_x']),float(r['geometry_y'])]))));assert maxgeom<1e-8
            assert int(r['retained_visible_count'])==int(truth['visible'][list(kept)].sum())
        pp=pred[start:start+len(gg)];yy=np.asarray(yy);tt=np.asarray(tt);updated=pp.astype(np.float64)+(np.asarray(gg)-(pp@A+b))@right
        for arm,v in [('baseline',pp),('geometry_constraint',updated)]:
            feature=np.mean((v-yy)**2,axis=1);pos=np.mean((v@A+b-tt)**2,axis=1)
            for j in range(len(gg)):
                r=by[start+j][arm];maximum=max(maximum,abs(float(feature[j])-float(r['feature_mse'])),abs(float(pos[j])-float(r['task_error'])));assert maximum<1e-7
    assert all(sha(k)==v for k,v in read(out/'asset_ledger.json').items())
    result=dict(status='PASS',raw_rows=41472,paired_rows=1152,complete_vectors_recomputed=20736,feature_and_position_max_difference=maximum,independent_geometry_max_difference=maxgeom,independent_scalar_statistics=sum(x['independent_scalar_statistics'] for x in scalar.values()),per_arm=scalar,causal_equal_capacity_axes=True,no_predictor_forward=True,training=False,fitting=False,GPU=0)
    (out/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main(sys.argv[1])
