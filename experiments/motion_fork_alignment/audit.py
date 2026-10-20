"""Independent scalar reconstruction, using explicit deletion comparisons."""
import collections,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(out):
    out=Path(out);p=json.loads((ROOT/'protocol.json').read_text());parent=Path(p['parent'])
    assert sha(parent/'raw_deletions.csv')==p['raw_sha256']
    with (parent/'raw_deletions.csv').open() as f:raw=list(csv.DictReader(f))
    groups=collections.defaultdict(list)
    for r in raw:groups[r['split'],int(r['horizon']),int(r['seed']),r['pair'],r['episode'],r['pool']].append(r)
    assert len(groups)==27648 and all(len(x)==3 for x in groups.values())
    totals=collections.defaultdict(list)
    for (split,h,seed,pair,ep,pool),rows in groups.items():
        rows=sorted(rows,key=lambda r:int(r['deleted']))
        other=sorted(groups[split,9-h,seed,pair,ep,pool],key=lambda r:int(r['deleted']))
        f=[float(r['feature_mse']) for r in rows];t=[float(r['task_error']) for r in rows]
        ff=[float(r['feature_mse']) for r in other];best=min(t)
        selected=[i for i in range(3) if f[i]-min(f)<=1e-8]
        transferred=[i for i in range(3) if ff[i]-min(ff)<=1e-8]
        optimal=[i for i in range(3) if t[i]-best<=1e-12]
        ft=sum(t[i] for i in selected)/len(selected);xt=sum(t[i] for i in transferred)/len(transferred)
        conflict=aligned=tied=0
        for i in range(3):
            for j in range(i+1,3):
                if abs(f[i]-f[j])<=1e-8 or abs(t[i]-t[j])<=1e-12:tied+=1
                elif (f[i]<f[j])==(t[i]<t[j]):aligned+=1
                else:conflict+=1
        nonflat=float(max(f)-min(f)>1e-8 and max(t)-min(t)>1e-12)
        rate=sum(i in optimal for i in selected)/len(selected)
        totals[split,h,seed,pair].append(dict(feature_task_regret=ft-best,transfer_task_regret=xt-best,
            uniform_task_regret=sum(t)/3-best,feature_task_error=ft,transfer_task_error=xt,
            task_optimal_error=best,uniform_task_error=sum(t)/3,fraction_task_optimal=rate,
            transfer_fraction_task_optimal=sum(i in optimal for i in transferred)/len(transferred),
            best_set_overlap=float(any(i in optimal for i in selected)),feature_flat=float(max(f)-min(f)<=1e-8),
            task_flat=float(max(t)-min(t)<=1e-12),both_nonflat=nonflat,task_optimal_joint_nonflat=nonflat*rate,
            pair_order_conflict=conflict/3,pair_order_aligned=aligned/3,pair_order_tied=tied/3))
    with (out/'paired_alignment.csv').open() as f:paired=list(csv.DictReader(f))
    maxpair=0.
    for row in paired:
        key=row['split'],int(row['horizon']),int(row['seed']),row['pair'];values=totals[key];assert len(values)==12
        for m in values[0]:maxpair=max(maxpair,abs(sum(v[m] for v in values)/12-float(row[m])))
    assert maxpair<1e-12 and len(paired)==len(totals)==2304
    summary=json.loads((out/'summary.json').read_text());ix=np.random.default_rng(0).integers(0,128,(10000,128))
    weights=np.stack([np.bincount(a,minlength=128) for a in ix])/128
    maxci=0.;n=0
    for cell in summary['cells']:
        keys=sorted({k[3] for k in totals if k[:2]==(cell['split'],cell['horizon'])});assert len(keys)==128
        for metric,wanted in cell['stats'].items():
            a=np.array([sum(sum(v[metric] for v in totals[cell['split'],cell['horizon'],seed,pair])/12 for seed in (17,29,43))/3 for pair in keys])
            low,high=np.percentile(weights@a,[2.5,97.5]);d=max(abs(a.mean()-wanted['mean']),abs(low-wanted['low']),abs(high-wanted['high']))
            assert d<2e-12;maxci=max(maxci,d);n+=1
    assert n==102
    assert sha(parent/'raw_deletions.csv')==p['raw_sha256']
    result=dict(status='PASS_INDEPENDENT_SCALAR_FORK_AUDIT',raw_rows=82944,horizon_decisions=27648,
                paired_rows=2304,statistics=n,max_paired_difference=maxpair,max_interval_difference=maxci,
                primary_source_lock_sha256=sha(ROOT/'source_lock.json'),no_model_calls=True,GPU=0)
    (out/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main(sys.argv[1])
