"""SVD/vector audit independent of the primary QR/energy decomposition."""
import collections,csv,json,sys
from pathlib import Path
import numpy as np
from cache import Cache,sha,ROOT
def readcsv(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def main(out):
    out=Path(out);c=Cache();d=c.decoder;A=d['weight'][:,1:]/d['std'][:,None];b=d['bias'][1:]-d['mean']@A
    Ua,sa,_=np.linalg.svd(A,full_matrices=False);Ub,sb,_=np.linalg.svd(c.B.astype(float),full_matrices=False)
    Uc,sc,_=np.linalg.svd(Ub.T@A,full_matrices=False);Us=Ub@Uc
    R=np.linalg.pinv(Ub.T@A)@Ub.T
    rows=readcsv(out/'raw_components.csv');assert len(rows)==20736
    component_max=0.;null_leak=0.;regret_max=0.
    for start in range(0,len(rows),64):
        rr=rows[start:start+64];meta=c.raw[start:start+len(rr)];p=c.pred[start:start+len(rr)].astype(float)
        geometry=np.array([[float(r['geometry_x']),float(r['geometry_y'])] for r in meta]);enabled=np.array([r['allow_update']=='True' for r in meta])
        p=p+np.where(enabled[:,None],(geometry-(p@A+b))@R,0.)
        target,truth=map(np.array,zip(*(c.target(r) for r in meta)));e=p-target
        visible=(e@Ua)@Ua.T;null=e-visible;inbasis=(e@Ub)@Ub.T;sensitive=(e@Us)@Us.T;bn=inbasis-sensitive;outside=e-inbasis
        posdiff=e@A;calibration=target@A+b-truth
        vals={name:np.mean(v*v,1) for name,v in [('total',e),('visible',visible),('null',null),('basis_sensitive',sensitive),('basis_null',bn),('outside_basis',outside)]}
        vals.update(readout_consistency=np.sum(posdiff**2,1)/2,target_readout_error=np.sum(calibration**2,1)/2,task_cross=np.sum(posdiff*calibration,1),task_reconstructed=np.sum((p@A+b-truth)**2,1)/2)
        null_leak=max(null_leak,float(abs(null@A).max()),float(abs(bn@A).max()))
        for j,r in enumerate(rr):
            assert int(r['witness_index'])==start+j and r['episode']==meta[j]['episode']
            component_max=max(component_max,max(abs(float(r[k])-v[j]) for k,v in vals.items()))
    assert component_max<1e-9 and null_leak<1e-9
    groups=collections.defaultdict(list)
    for r in rows:groups[int(r['seed']),int(r['horizon']),r['pair'],r['episode'],r['pool']].append(r)
    metrics=collections.defaultdict(list)
    scores={'full':'source_feature_mse','full64':'total','visible':'visible','null':'null','basis_sensitive':'basis_sensitive','basis_null':'basis_null','readout_consistency':'readout_consistency','uniform':None,'task_optimal':'source_task_error'}
    components=['total','visible','null','basis_sensitive','basis_null','outside_basis','readout_consistency','target_readout_error','task_cross','task_reconstructed']
    def choose(rr,column,tol):
        v=[float(r[column]) for r in rr];return [i for i in range(3) if v[i]-min(v)<=tol]
    for (seed,h,pair,ep,pool),rr in groups.items():
        assert len(rr)==3;task=[float(r['source_task_error']) for r in rr];best=min(task);ts=choose(rr,'source_task_error',1e-12);full=choose(rr,'source_feature_mse',1e-8);v={}
        for name,col in scores.items():
            idx=list(range(3)) if col is None else choose(rr,col,1e-12 if name in ('task_optimal','readout_consistency') else 1e-8)
            v[name+'_task_error']=sum(task[i] for i in idx)/len(idx);v[name+'_task_regret']=v[name+'_task_error']-best;v[name+'_task_optimal']=sum(i in ts for i in idx)/len(idx)
        for m in components:
            a=[float(r[m]) for r in rr];v[m+'_mean']=sum(a)/3;v[m+'_range']=max(a)-min(a);v[m+'_full_minus_task']=sum(a[i] for i in full)/len(full)-sum(a[i] for i in ts)/len(ts)
        v['full_float64_choice_differs']=float(set(full)!=set(choose(rr,'total',1e-8)))
        v['feature_flat']=float(max(float(r['source_feature_mse']) for r in rr)-min(float(r['source_feature_mse']) for r in rr)<=1e-8);v['task_flat']=float(max(task)-min(task)<=1e-12)
        counts=[0,0,0,0]
        for i in range(3):
            for j in range(i+1,3):
                dt=task[i]-task[j];df=float(rr[i]['source_feature_mse'])-float(rr[j]['source_feature_mse'])
                if abs(dt)<=1e-12 or abs(df)<=1e-8 or (df<0)==(dt<0):continue
                counts[0]+=1
                def diff(m):return float(rr[i][m])-float(rr[j][m])
                dn,dv,dbn,dbs=map(diff,['null','visible','basis_null','basis_sensitive'])
                counts[1]+=int(abs(dn)>1e-8 and abs(dv)>1e-8 and (dn<0)!=(dt<0) and (dv<0)==(dt<0))
                counts[2]+=int(abs(dbn)>1e-8 and abs(dbs)>1e-8 and (dbn<0)!=(dt<0) and (dbs<0)==(dt<0))
                counts[3]+=int(abs(dv)>1e-8 and (dv<0)!=(dt<0))
        v.update(zip(['order_conflict','null_overrides_visible','basis_null_overrides_sensitive','visible_itself_conflicts'],[x/3 for x in counts]))
        metrics[seed,h,pair].append(v)
    paired=readcsv(out/'paired_components.csv');assert len(paired)==576 and all(len(v)==12 for v in metrics.values())
    pairedmax=0.
    for r in paired:
        vv=metrics[int(r['seed']),int(r['horizon']),r['pair']]
        pairedmax=max(pairedmax,max(abs(sum(v[k] for v in vv)/12-float(r[k])) for k in vv[0]))
    assert pairedmax<1e-12
    s=json.loads((out/'summary.json').read_text());ix=np.random.default_rng(0).integers(0,96,(10000,96));weights=np.stack([np.bincount(x,minlength=96) for x in ix])/96
    statmax=0.;n=0
    for cell in s['cells']:
        ids=sorted({key[2] for key in metrics if key[1]==cell['horizon']});assert len(ids)==96
        for m,w in cell['stats'].items():
            a=np.array([sum(sum(v[m] for v in metrics[seed,cell['horizon'],pair])/12 for seed in (17,29,43))/3 for pair in ids]);lo,hi=np.quantile(weights@a,[.025,.975]);delta=max(abs(a.mean()-w['mean']),abs(lo-w['low']),abs(hi-w['high']));assert delta<2e-12;statmax=max(statmax,delta);n+=1
    assert n==128;c.verify()
    result=dict(status='PASS_INDEPENDENT_SVD_RESIDUAL_AND_RANK_AUDIT',vectors=20736,horizon_decisions=6912,paired_rows=576,statistics=n,component_max_difference=component_max,null_readout_leak_max=null_leak,paired_max_difference=pairedmax,interval_max_difference=statmax,no_model_calls=True,GPU=0)
    (out/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main(sys.argv[1])
