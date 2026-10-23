"""Independent raw-CSV audit; no models, training, or result selection."""
import argparse,csv,json,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def interval(base, alt, pair_ids, *, ratio):
    groups=defaultdict(list)
    for a,b,k in zip(base,alt,pair_ids): groups[k].append([a,b])
    x=np.asarray([np.mean(v,axis=0) for v in groups.values()])
    def score(y):
        a,b=y[...,0].mean(axis=-1),y[...,1].mean(axis=-1)
        return (a-b)/np.maximum(a,.02) if ratio else a-b
    draws=np.random.default_rng(0).integers(0,len(x),size=(2000,len(x)))
    v=score(x[draws]);q=np.quantile(v,[.025,.975])
    return {'mean':float(score(x)),'low':float(q[0]),'high':float(q[1]),'pairs':len(x)}
def raw(path,task):
    with Path(path).open(newline='') as f: rows=list(csv.DictReader(f))
    if len(rows)!=34560: raise ValueError('Expected34560 raw records per task')
    index={}
    for r in rows:
        key=tuple(r[x] for x in ('seed','task','split','horizon','condition','pair','episode'))
        if key in index or r['task']!=task: raise ValueError('Duplicate/wrong task')
        for k,v in r.items():
            if k not in ('episode','pair','task','split','condition','sibling') and not np.isfinite(float(v)):
                raise ValueError('Nonfinite raw scalar')
        index[key]=r
    counts=defaultdict(lambda:defaultdict(set))
    for r in rows: counts[(r['seed'],r['split'],r['horizon'],r['condition'])][r['pair']].add(r['episode'])
    conditions={'recent_only','full_history','privileged_small_budget','random_small_budget','wrong_evidence','persistence'}
    expected={(str(s),sp,str(h),c) for s in (17,29,43) for sp in ('validation','test_id','test_long_gap','test_distractor_shift') for h in (3,6) for c in conditions}
    if set(counts)!=expected: raise ValueError('Missing raw slice')
    for (_,split,_,_),g in counts.items():
        if len(g)!=(96 if split=='validation' else 128) or any(len(v)!=2 for v in g.values()): raise ValueError('Incomplete pair/sibling slice')
    return rows,index

def audit(root):
    root=Path(root);outcome=json.loads((root/'outcome.json').read_text());results={}
    if set(outcome['tasks'])!={'gate'}: raise ValueError('Both tasks required')
    for task in ('gate',):
        r=root;path=r/'evaluation/raw_records.csv';rows,cand=raw(path,task)
        old=Path('/scratch/xl4358/phase1_v3_dev')/task/'reference';_,ref=raw(old/'evaluation/raw_records.csv',task)
        if set(cand)!=set(ref): raise ValueError('Reference/candidate axes differ')
        compare=json.loads((r/'development_comparison.json').read_text())
        metrics=json.loads((r/'evaluation/metrics.json').read_text())
        if sha(path)!=compare['candidate_raw_sha256'] or sha(path)!=metrics['raw_sha256']: raise ValueError('Raw SHA mismatch')
        independent=[]
        for check in compare['fixed_reference_checks']:
            name=check['slice'];_,split,h,seed=name.split('/');h=h[1:];seed=seed[4:]
            selected=[(k,v) for k,v in cand.items() if k[2]==split and k[3]==h and k[4]=='privileged_small_budget' and (seed=='all_seeds_mean' or k[0]==seed)]
            grouped=defaultdict(list)
            for k,v in selected:
                recent=list(k);recent[4]='recent_only'
                grouped[(k[5],k[6])].append([float(ref[k]['normalized_mse']),float(v['normalized_mse']),float(ref[tuple(recent)]['normalized_mse'])])
            order=sorted(grouped);values=np.asarray([np.mean(grouped[k],axis=0) for k in order]);pairs=[k[0] for k in order]
            if check['kind']=='critical_absolute_mse_reduction':
                got=interval(values[:,0],values[:,1],pairs,ratio=False);expected=check['absolute_mse_reduction'];passed=got['mean']>0 and got['low']>0
            else:
                use_ratio=values[:,2].mean()>=.02;got=interval(values[:,2],values[:,1],pairs,ratio=use_ratio);expected=check['improvement'];passed=got['mean']>=(.1 if use_ratio else .01) and got['low']>0
            if any(not np.isclose(got[k],expected[k],rtol=0,atol=1e-12) for k in got) or bool(check['passed'])!=bool(passed): raise ValueError('Independent paired statistics mismatch')
            independent.append({'slice':name,'kind':check['kind'],'passed':bool(passed),'independent':got})
        if len(independent)!=32:raise ValueError('Expected32 fixed reference checks')
        results[task]={'raw_records':len(rows),'raw_sha256':sha(path),'all_axes_finite_pairs_pass':True,'paired_checks':independent}
    return {'status':'INDEPENDENT_RAW_AXIS_FINITE_AND_32_FIXED_COMPARISONS_PASS','tasks':results,'phase1_go':False,'confirmation_opened':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');args=parser.parse_args();root=Path(args.root)
    result=audit(root);(root/'independent_scalar_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['status'])
