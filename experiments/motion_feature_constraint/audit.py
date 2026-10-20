"""Independent QR/SVD reconstruction and paired-statistic audit; no model calls."""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def csvread(p):
    with Path(p).open() as f:return list(csv.DictReader(f))

def main(out):
    out=Path(out);s=read(out/'summary.json');assert sha(out/'validation_feature_constraint.csv')==s['raw_sha256'] and sha(out/'constraint_coefficients.npz')==s['coefficients_sha256']
    parent=Path('/scratch/xl4358/phase1_motion_geometry_validation/output/job_8336');ps=read(parent/'summary.json')
    assert sha(parent/'prediction_witnesses.npz')==ps['witness_sha256']==s['parent_vector_sha256']
    assert sha(parent/'validation_geometry.csv')==ps['raw_sha256']
    meta=csvread(parent/'validation_geometry.csv');raw=csvread(out/'validation_feature_constraint.csv');assert len(meta)==1152 and len(raw)==5760
    with np.load(parent/'prediction_witnesses.npz',allow_pickle=False) as f:p=f['predictions'][:,1].astype(np.float64);recent=f['predictions'][:,0].astype(np.float64);y=f['targets'].astype(np.float64)
    ref=Path('/scratch/xl4358/phase1_v3_dev/motion/reference')
    with np.load(ref/'predictor/motion/basis.npz',allow_pickle=False) as f:B=f['output_basis'].astype(np.float64)
    with np.load(ref/'probes/motion_state_decoder.npz',allow_pickle=False) as f:w=f['weight'];std=f['std'];mean=f['mean'];bias=f['bias']
    A=w[:,1:]/std[:,None];b=bias[1:]-mean@A;Q,_=np.linalg.qr(B,mode='reduced');F=Q.T@A;right=np.linalg.pinv(F)@Q.T
    g=np.array([[float(r['kinematic_actions_x']),float(r['kinematic_actions_y'])] for r in meta]);truth=np.array([[float(r['target_x']),float(r['target_y'])] for r in meta])
    swap=np.empty_like(g);pairs=sorted({r['pair'] for r in meta})
    for seed in (17,29,43):
        for h in (3,6):
            bypair={pair:sorted([i for i,r in enumerate(meta) if int(r['seed'])==seed and int(r['horizon'])==h and r['pair']==pair],key=lambda i:meta[i]['episode']) for pair in pairs}
            for j,pair in enumerate(pairs):swap[bypair[pair]]=g[bypair[pairs[(j+1)%len(pairs)]]]
    arms={'baseline':p,'geometry_constraint':p+(g-(p@A+b))@right,'shuffled_geometry_control':p+(swap-(p@A+b))@right,'oracle_position_constraint':p+(truth-(p@A+b))@right,'oracle_full_basis':p+((y-p)@Q)@Q.T}
    index={(r['arm'],int(r['seed']),r['episode'],int(r['horizon'])):r for r in raw};assert len(index)==5760
    errs={};maximum=0.
    for arm,v in arms.items():
        assert np.isfinite(v).all();decoded=v@A+b;feature=np.mean((v-y)**2,axis=1);position=np.mean((decoded-truth)**2,axis=1);delta=np.mean((v-p)**2,axis=1);errs[arm]=(feature,position)
        for i,r in enumerate(meta):
            x=index[arm,int(r['seed']),r['episode'],int(r['horizon'])]
            observed=np.array([float(x[k]) for k in ['feature_mse','position_mse','delta_feature_mse','prediction_x','prediction_y']]);expected=np.array([feature[i],position[i],delta[i],*decoded[i]])
            maximum=max(maximum,float(np.max(np.abs(observed-expected))));assert maximum<1e-8
    def effect(base,alt,ids,floor,absolute):
        pg={pair:[i for i in ids if meta[i]['pair']==pair] for pair in pairs};P=np.array([[np.mean(base[v]),np.mean(alt[v])] for v in pg.values()]);isabs=P[:,0].mean()<floor
        rng=np.random.default_rng(0);sample=P[rng.integers(0,96,size=(2000,96))];av=sample[:,:,0].mean(1);bv=sample[:,:,1].mean(1)
        boot=av-bv if isabs else (av-bv)/np.maximum(av,floor);v=P[:,0].mean()-P[:,1].mean();v=v if isabs else v/max(P[:,0].mean(),floor);low,high=np.percentile(boot,[2.5,97.5]);th=absolute if isabs else .1
        return dict(mean=float(v),low=float(low),high=float(high),numeric_rule_met=bool(v>=th and low>0))
    recent_feature=np.mean((recent-y)**2,axis=1);recent_position=np.mean((recent@A+b-truth)**2,axis=1);intervals=0;maxstat=0.
    for cell in s['cells']:
        ids=[i for i,r in enumerate(meta) if int(r['horizon'])==cell['horizon'] and (cell['seed']=='mean' or int(r['seed'])==cell['seed'])]
        ft,st=errs[cell['arm']]
        for name,base,alt,floor,absolute in [('feature_vs_recent',recent_feature,ft,.02,.01),('position_vs_recent',recent_position,st,1e-5,1e-6),('feature_vs_baseline',errs['baseline'][0],ft,.02,.01)]:
            computed=effect(base,alt,ids,floor,absolute);expected=cell[name]
            maxstat=max(maxstat,max(abs(computed[k]-expected[k]) for k in ['mean','low','high']));assert maxstat<1e-12 and computed['numeric_rule_met']==expected['numeric_rule_met'];intervals+=1
    assert all(sha(k)==v for k,v in read(out/'asset_ledger.json').items())
    if s['conditional_deletion_eligible']:
        assert s['feature_numeric_checks']==s['position_numeric_checks']==s['incremental_feature_positive_cells']==8
    else:assert s['status']=='VALIDATION_FEATURE_CONSTRAINT_REJECTED_STOP_BEFORE_DELETION'
    result=dict(status='PASS',rows=5760,source_rows=1152,basepairs=96,independent_QR_SVD_vector_scalar_max_difference=maximum,independent_intervals=intervals,max_statistic_difference=maxstat,all_original_numeric_flags_recomputed=True,model_forward=False,training=False,fitting=False,GPU=0)
    (out/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main(sys.argv[1])
