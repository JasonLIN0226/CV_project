"""Original-validation cache-only residual decomposition and deletion rankings."""
import collections,csv,json,sys,time
from pathlib import Path
import numpy as np
from cache import Cache,ROOT,sha
from kernel import operators,restore,decompose,group_metrics,COMPONENTS
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def savecsv(p,rr):
    with Path(p).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)

def aggregate(rows):
    groups=collections.defaultdict(list)
    for r in rows:groups[int(r['seed']),int(r['horizon']),r['pair'],r['episode'],r['pool']].append(r)
    assert len(groups)==6912 and all(len(v)==3 for v in groups.values())
    combined=collections.defaultdict(list)
    for (seed,h,pair,ep,pool),rr in groups.items():
        assert len({int(r['deleted']) for r in rr})==3
        combined[seed,h,pair].append(group_metrics(sorted(rr,key=lambda r:int(r['deleted']))))
    paired=[]
    for (seed,h,pair),rr in sorted(combined.items()):
        assert len(rr)==12
        paired.append(dict(seed=seed,horizon=h,pair=pair,**{k:float(np.mean([v[k] for v in rr])) for k in rr[0]}))
    assert len(paired)==576
    return paired

def summarize(rows):
    fields=[k for k in rows[0] if k not in ('seed','horizon','pair')];cells=[]
    ix=np.random.default_rng(0).integers(0,96,(10000,96))
    for h in (3,6):
        rr=[r for r in rows if int(r['horizon'])==h];pairs=sorted({r['pair'] for r in rr});assert len(rr)==288 and len(pairs)==96
        stats={}
        for k in fields:
            a=np.array([np.mean([float(r[k]) for r in rr if r['pair']==pair]) for pair in pairs]);lo,hi=np.quantile(a[ix].mean(1),[.025,.975]);stats[k]=dict(mean=float(a.mean()),low=float(lo),high=float(hi))
        energy=stats['total_mean']['mean'];conflict=stats['order_conflict']['mean']
        cells.append(dict(horizon=h,pairs=96,stats=stats,
                          null_energy_fraction=stats['null_mean']['mean']/energy,
                          outside_basis_energy_fraction=stats['outside_basis_mean']['mean']/energy,
                          conflict_explained_by_null_override=None if conflict==0 else stats['null_overrides_visible']['mean']/conflict,
                          conflict_explained_by_basis_null_override=None if conflict==0 else stats['basis_null_overrides_sensitive']['mean']/conflict))
    return cells

def main(out):
    began=time.monotonic();cache=Cache();d=cache.decoder;op=operators(cache.B,d['weight'],d['std'],d['mean'],d['bias'])
    out=Path(out);out.mkdir(exist_ok=False);rows=[];replay=0.;identity=0.;float64_difference=0.
    for start in range(0,20736,64):
        rr=cache.raw[start:start+64];target,truth=map(np.array,zip(*(cache.target(r) for r in rr)))
        p=restore(cache.pred[start:start+len(rr)],np.array([[float(r['geometry_x']),float(r['geometry_y'])] for r in rr]),np.array([r['allow_update']=='True' for r in rr]),op)
        comp=decompose(p,target,truth,op)
        for j,r in enumerate(rr):
            values={k:float(v[j]) for k,v in comp.items()};source=float(r['feature_mse']);task=float(r['task_error'])
            # Reproduce the saved parent arithmetic; keep float64 decomposition separate.
            exact_feature=values['total'] if r['allow_update']=='True' else float(np.mean((cache.pred[start+j]-target[j])**2))
            replay=max(replay,abs(exact_feature-source),abs(values['task_reconstructed']-task));assert replay<1e-7
            float64_difference=max(float64_difference,abs(values['total']-source))
            identity=max(identity,abs(values['total']-values['visible']-values['null']),abs(values['total']-values['basis_sensitive']-values['basis_null']-values['outside_basis']))
            rows.append(dict(seed=int(r['seed']),split='validation',pair=r['pair'],episode=r['episode'],pool=r['pool'],parent=r['parent'],deleted=int(r['deleted']),kept=r['kept'],query=int(r['query']),horizon=int(r['horizon']),witness_index=int(r['witness_index']),allow_update=r['allow_update'],source_feature_mse=source,source_task_error=task,**values))
        if time.monotonic()-began>1200:raise RuntimeError('Read-only CPU time limit; preserve partial, no automatic retry')
    assert len(rows)==20736 and identity<1e-9
    savecsv(out/'raw_components.csv',rows);paired=aggregate(rows);savecsv(out/'paired_components.csv',paired)
    cache.verify();write(out/'asset_ledger.json',cache.assets)
    write(out/'summary.json',dict(status='COMPLETE_VALIDATION_RESIDUAL_DECOMPOSITION_NOT_ACCEPTANCE',raw_rows=20736,horizon_decisions=6912,paired_rows=576,
        cells=summarize(paired),raw_sha256=sha(out/'raw_components.csv'),paired_sha256=sha(out/'paired_components.csv'),
        original_scalar_replay_max_difference=replay,float64_vs_source_max_difference=float64_difference,energy_identity_max_difference=identity,
        decoder_singular_values=np.linalg.svd(op['A'],compute_uv=False).tolist(),decoder_condition_number=float(np.linalg.cond(op['A'])),
        source_lock_sha256=sha(ROOT/'source_lock.json'),protocol_sha256=sha(ROOT/'protocol.json'),parent_vector_sha256=cache.p['parent_vectors_sha256'],
        elapsed_seconds=time.monotonic()-began,GPU=0,model_calls=0,training=False,fitting=False,test_access=False,original_Phase1_GO=False,fresh9137_opened=False))
if __name__=='__main__':main(sys.argv[1])
