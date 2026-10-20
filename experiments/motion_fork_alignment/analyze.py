"""Read existing scalar forks; no predictor/encoder imports or calls."""
import collections,csv,hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
METRICS=('feature_task_regret','transfer_task_regret','uniform_task_regret',
         'feature_task_error','transfer_task_error','task_optimal_error','uniform_task_error',
         'fraction_task_optimal','transfer_fraction_task_optimal','best_set_overlap',
         'feature_flat','task_flat','both_nonflat','task_optimal_joint_nonflat','pair_order_conflict','pair_order_aligned','pair_order_tied')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def fork_metrics(losses,other_feature):
    a=np.asarray(losses,float);assert a.shape==(3,2) and np.isfinite(a).all()
    feature,task=a.T
    fs=np.flatnonzero(feature<=feature.min()+1e-8)
    ts=np.flatnonzero(task<=task.min()+1e-12)
    xs=np.flatnonzero(other_feature<=np.min(other_feature)+1e-8)
    labels=[]
    for i,j in ((0,1),(0,2),(1,2)):
        f=feature[i]-feature[j];t=task[i]-task[j]
        labels.append(0 if abs(f)<=1e-8 or abs(t)<=1e-12 else (1 if f*t>0 else -1))
    v=dict(feature_task_regret=float(task[fs].mean()-task.min()),transfer_task_regret=float(task[xs].mean()-task.min()),
           uniform_task_regret=float(task.mean()-task.min()),feature_task_error=float(task[fs].mean()),
           transfer_task_error=float(task[xs].mean()),task_optimal_error=float(task.min()),uniform_task_error=float(task.mean()),
           fraction_task_optimal=float(np.isin(fs,ts).mean()),transfer_fraction_task_optimal=float(np.isin(xs,ts).mean()),
           best_set_overlap=float(bool(set(fs)&set(ts))),feature_flat=float(np.ptp(feature)<=1e-8),
           task_flat=float(np.ptp(task)<=1e-12),both_nonflat=float(np.ptp(feature)>1e-8 and np.ptp(task)>1e-12),
           pair_order_conflict=labels.count(-1)/3,pair_order_aligned=labels.count(1)/3,pair_order_tied=labels.count(0)/3)
    v['task_optimal_joint_nonflat']=v['both_nonflat']*v['fraction_task_optimal']
    return v

def load_raw(path,protocol):
    assert sha(path)==protocol['raw_sha256']
    with Path(path).open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==82944
    groups=collections.defaultdict(dict)
    for r in rows:
        seed=int(r['seed']);h=int(r['horizon']);d=int(r['deleted']);parent=tuple(map(int,r['parent'].split(';')))
        assert r['task']=='motion' and r['pool_group']=='clock_all' and seed in (17,29,43)
        assert r['split'] in protocol['splits'] and h in (3,6) and r['pool'] in [f'clock_{i}' for i in range(6)]
        assert len(parent)==3 and parent==tuple(sorted(set(parent))) and d in parent
        assert set(map(int,r['kept'].split(';')))==set(parent)-{d} and int(r['history_count'])==2
        assert max(parent)<int(r['query'])-3 and int(r['feature_dim'])==5120
        k=(r['split'],seed,r['pair'],r['episode'],r['pool'])
        assert (h,d) not in groups[k];groups[k][h,d]=r
    assert len(groups)==13824
    return groups

def calculate(groups):
    accumulated=collections.defaultdict(list);counts=collections.Counter();replay=collections.defaultdict(list)
    for (split,seed,pair,episode,pool),rr in sorted(groups.items()):
        parent=tuple(map(int,next(iter(rr.values()))['parent'].split(';')))
        assert set(rr)=={(h,d) for h in (3,6) for d in parent}
        assert all(tuple(map(int,r['parent'].split(';')))==parent for r in rr.values())
        a=np.array([[[float(rr[h,d][m]) for m in ('feature_mse','task_error')] for d in parent] for h in (3,6)])
        assert np.isfinite(a).all()
        for i,h in enumerate((3,6)):
            accumulated[split,h,seed,pair].append(fork_metrics(a[i],a[1-i,:,0]))
            counts[split,h,seed]+=1
            same=np.flatnonzero(a[i,:,0]<=a[i,:,0].min()+1e-8);other=np.flatnonzero(a[1-i,:,0]<=a[1-i,:,0].min()+1e-8)
            replay[split,h,seed,pair].append(np.stack([a[i].mean(0),a[i,0],a[i,-1],a[i,same].mean(0),a[i,other].mean(0)]))
    pairs=[];replays=[]
    for (split,h,seed,pair),values in sorted(accumulated.items()):
        assert len(values)==12
        pairs.append(dict(split=split,horizon=h,seed=seed,pair=pair,**{m:float(np.mean([v[m] for v in values])) for m in METRICS}))
        v=np.mean(replay[split,h,seed,pair],0)
        replays.append(dict(split=split,horizon=h,seed=seed,pair=pair,values=v))
    assert len(pairs)==2304
    return pairs,replays

def summarize(rows):
    ix=np.random.default_rng(0).integers(0,128,(10000,128));cells=[]
    for split in ('test_id','test_long_gap','test_distractor_shift'):
        for h in (3,6):
            rr=[r for r in rows if r['split']==split and int(r['horizon'])==h]
            ids=sorted({r['pair'] for r in rr});assert len(ids)==128 and len(rr)==384
            matrix=np.array([[np.mean([float(r[m]) for r in rr if r['pair']==p]) for m in METRICS] for p in ids])
            stats={}
            for j,m in enumerate(METRICS):
                lo,hi=np.quantile(matrix[ix,j].mean(1),[.025,.975]);stats[m]=dict(mean=float(matrix[:,j].mean()),low=float(lo),high=float(hi))
            denom=stats['uniform_task_regret']['mean'];ratio=None if denom<=1e-12 else stats['feature_task_regret']['mean']/denom
            cells.append(dict(split=split,horizon=h,pairs=128,episode_pools_per_seed=1536,
                              stats=stats,uniform_task_headroom_lost_by_feature_choice=ratio,
                              nonflat_conditional_task_optimal=None if stats['both_nonflat']['mean']==0 else stats['task_optimal_joint_nonflat']['mean']/stats['both_nonflat']['mean']))
    return cells

def main(out):
    protocol=read(ROOT/'protocol.json');parent=Path(protocol['parent'])
    assert sha(parent/'summary.json')==protocol['parent_summary_sha256'] and sha(parent/'paired_deletions.csv')==protocol['paired_sha256']
    groups=load_raw(parent/'raw_deletions.csv',protocol);pairs,replay=calculate(groups)
    with (parent/'paired_deletions.csv').open() as f:old={(r['split'],int(r['horizon']),int(r['seed']),r['pair']):r for r in csv.DictReader(f)}
    maximum=0.
    for r in replay:
        target=old[r['split'],r['horizon'],r['seed'],r['pair']]
        for i,p in enumerate(('uniform','fifo','reject','oracle_same','oracle_transfer')):
            for j,m in enumerate(('feature_mse','task_error')):maximum=max(maximum,abs(r['values'][i,j]-float(target[p+'_'+m])))
    assert maximum<1e-12
    out=Path(out);out.mkdir(exist_ok=False)
    with (out/'paired_alignment.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
    write(out/'summary.json',dict(status='COMPLETE_FROZEN_FORK_ALIGNMENT_DIAGNOSIS',raw_rows=82944,episode_pools=13824,
        horizon_decisions=27648,paired_rows=2304,raw_sha256=sha(parent/'raw_deletions.csv'),paired_sha256=sha(out/'paired_alignment.csv'),
        original_compact_replay_max_difference=maximum,cells=summarize(pairs),GPU=0,new_model_calls=0,training=False,
        original_Phase1_GO=False,fresh9137_opened=False,no_acceptance_change=True))
    assert sha(parent/'raw_deletions.csv')==protocol['raw_sha256']
if __name__=='__main__':
    import sys
    main(sys.argv[1])
