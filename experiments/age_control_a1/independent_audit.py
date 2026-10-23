"""Independent raw-fork reconstruction; imports neither primary analyzer nor model."""
import csv,hashlib,json,os,socket,subprocess,sys
from pathlib import Path
from collections import defaultdict
import numpy as np

SEEDS=(17,29,43)
SPLITS=('test_id','test_long_gap','test_distractor_shift')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def bounds(a,repeats):
    a=np.asarray(a,np.float64);gen=np.random.default_rng(0)
    boots=a[gen.integers(0,len(a),size=(repeats,len(a)))].mean(axis=1)
    return np.array([a.mean(),np.quantile(boots,.025),np.quantile(boots,.975)])

def audit_rows(rows,compact,summary,pairs=128,repeats=10000):
    assert len(rows)==124416*pairs//128
    raw={};blueprints={};episodes=defaultdict(set);pair_sets=defaultdict(set)
    for r in rows:
        t,sp,pair,ep,pool=r['task'],r['split'],r['pair'],r['episode'],r['pool'];seed=int(r['seed']);h=int(r['horizon'])
        assert t in ('gate','motion') and sp in SPLITS and seed in SEEDS and h in (3,6)
        q=int(r['query']);end=int(r['evidence_end']);ev=tuple(map(int,r['evidence'].split(';')))
        parent=tuple(map(int,r['parent'].split(';')));kept=set(map(int,r['kept'].split(';')));d=int(r['deleted'])
        assert len(set(parent))==3 and parent==tuple(sorted(parent)) and 0<=min(parent)<=max(parent)<q-3
        assert d in parent and kept==set(parent)-{d} and len(kept)==int(r['history_count'])==2 and int(r['feature_dim'])==5120
        if t=='gate':
            assert len(ev)==1
            expected={'cue_oldest':(ev[0],end+1,q-4),'cue_middle':(ev[0]-1,ev[0],end+1),'cue_newest':(0,ev[0]-1,ev[0])}
            assert pool in expected and parent==expected[pool] and r['pool_group']==pool
            assert r['deleted_role']==('cue' if d==ev[0] else 'other')
        else:
            assert len(ev)==2 and r['pool_group']=='clock_all' and pool in {f'clock_{i}' for i in range(6)}
            keybytes=json.dumps((20261017,'clock_only',pair),sort_keys=True,separators=(',',':')).encode()
            random_seed=int(hashlib.sha256(keybytes).hexdigest()[:16],16)%(2**32)
            gen=np.random.default_rng(random_seed);triples=set()
            while len(triples)<6:triples.add(tuple(sorted(map(int,gen.choice(q-3,3,replace=False)))))
            assert parent==sorted(triples)[int(pool.split('_')[1])]
            assert r['deleted_role']==('start' if d==ev[0] else ('end' if d==ev[1] else 'other'))
        meta=(parent,q,end,ev)
        bk=(t,sp,pair,pool)
        assert bk not in blueprints or blueprints[bk]==meta
        blueprints[bk]=meta
        cell=(t,seed,sp,pair,ep,pool,h)
        assert d not in raw.setdefault(cell,{})
        value=np.array([float(r['feature_mse']),float(r['task_error'])]);assert np.isfinite(value).all()
        raw[cell][d]=value;episodes[(t,sp,pair)].add(ep);pair_sets[(t,seed,sp)].add(pair)
    expected_tasks={(t,s,sp) for t in ('gate','motion') for s in SEEDS for sp in SPLITS}
    assert set(pair_sets)==expected_tasks and all(len(v)==pairs for v in pair_sets.values())
    assert all(len(v)==2 for v in episodes.values())
    for t,s,sp in expected_tasks:
        assert pair_sets[(t,s,sp)]==pair_sets[(t,17,sp)]
        pools=('cue_oldest','cue_middle','cue_newest') if t=='gate' else tuple(f'clock_{i}' for i in range(6))
        for pair in pair_sets[(t,s,sp)]:
            for ep in episodes[(t,sp,pair)]:
                for pool in pools:
                    for h in (3,6):assert len(raw[(t,s,sp,pair,ep,pool,h)])==3
    accumulated=defaultdict(list);gate_values=defaultdict(list)
    for (t,s,sp,pair,ep,pool,h),values in raw.items():
        parent=sorted(values);target=np.stack([values[d] for d in parent])
        other=raw[(t,s,sp,pair,ep,pool,9-h)]
        best_now=min(v[0] for v in values.values());best_other=min(v[0] for v in other.values())
        choose_now=[d for d in parent if values[d][0]<=best_now+1e-8]
        choose_other=[d for d in parent if other[d][0]<=best_other+1e-8]
        group=pool if t=='gate' else 'clock_all';k=(t,s,sp,group,pair,h)
        val={'uniform':target.mean(0),'fifo':target[0],'reject':target[-1],
             'oracle_same':np.mean([values[d] for d in choose_now],axis=0),
             'oracle_transfer':np.mean([values[d] for d in choose_other],axis=0)}
        accumulated[k].append(val)
        if t=='gate':
            cue=blueprints[(t,sp,pair,pool)][3][0]
            gate_values[k].append((values[cue],np.mean([values[d] for d in parent if d!=cue],axis=0)))
    computed={}
    for k,rr in accumulated.items():
        assert len(rr)==(2 if k[0]=='gate' else 12)
        computed[k]={p:np.mean([x[p] for x in rr],axis=0) for p in ('uniform','fifo','reject','oracle_same','oracle_transfer')}
        if k[0]=='gate':
            v=np.mean(gate_values[k],axis=0);computed[k].update(critical=v[0],other=v[1])
    compact_map={}
    for r in compact:
        k=(r['task'],int(r['seed']),r['split'],r['pool_group'],r['pair'],int(r['horizon']))
        assert k not in compact_map;compact_map[k]=r
    assert set(compact_map)==set(computed) and len(compact_map)==9216*pairs//128
    for k,vals in computed.items():
        for label,arr in vals.items():
            for j,m in enumerate(('feature_mse','task_error')):assert np.isclose(float(compact_map[k][label+'_'+m]),arr[j],atol=2e-12,rtol=1e-9)
    def vector(spec,label,metric):
        t,s,sp,g,h=spec['task'],spec['seed'],spec['split'],spec['pool_group'],spec['horizon']
        ids=sorted(pair_sets[(t,17,sp)]);j=('feature_mse','task_error').index(metric)
        seeds=SEEDS if s=='mean' else (int(s),)
        return np.array([np.mean([computed[(t,x,sp,g,p,h)][label][j] for x in seeds]) for p in ids])
    count=0;gain_keys=set();critical_keys=set()
    for spec in summary['oracle_baseline_comparisons']:
        identity=tuple(spec[k] for k in ('task','seed','split','pool_group','horizon','choice','baseline'))
        assert identity not in gain_keys;gain_keys.add(identity)
        assert spec['selection_horizon']==(spec['horizon'] if spec['choice']=='oracle_same' else 9-spec['horizon'])
        for m,v in spec['metrics'].items():
            b=vector(spec,spec['baseline'],m);chosen=vector(spec,spec['choice'],m)
            obs=bounds(b-chosen,repeats);want=v['paired_gain']
            assert np.allclose(obs,[want[x] for x in ('mean','low','high')],atol=2e-12,rtol=1e-9)
            assert np.isclose(b.mean(),v['baseline_mean']) and np.isclose(chosen.mean(),v['chosen_mean'])
            relative=None if b.mean()<1e-8 else float((b.mean()-chosen.mean())/b.mean())
            assert (relative is None and v['relative_gain'] is None) or (relative is not None and np.isclose(relative,v['relative_gain']))
            count+=1
        assert spec['positive_both_metrics']==all(v['paired_gain']['low']>0 for v in spec['metrics'].values())
    for spec in summary['gate_critical_contrasts']:
        identity=tuple(spec[k] for k in ('task','seed','split','pool_group','horizon'))
        assert identity not in critical_keys;critical_keys.add(identity)
        flags=[]
        for m,v in spec['metrics'].items():
            c=vector(spec,'critical',m);n=vector(spec,'other',m);obs=bounds(c-n,repeats)
            assert np.allclose(obs,[v['difference'][x] for x in ('mean','low','high')],atol=2e-12,rtol=1e-9)
            assert np.isclose(c.mean(),v['critical_mean']) and np.isclose(n.mean(),v['other_mean'])
            reduction=float((c.mean()-n.mean())/max(c.mean(),1e-8))
            assert np.isclose(reduction,v['relative_reduction'])
            flag=bool(c.mean()>=1e-8 and reduction>=.1 and obs[1]>0)
            assert flag==v['passed'];flags.append(flag);count+=1
        assert spec['passed']==all(flags)
    group_pairs=[('gate',g) for g in ('cue_oldest','cue_middle','cue_newest')]+[('motion','clock_all')]
    expected={(t,s,sp,g,h,o,b) for t,g in group_pairs for s in (*SEEDS,'mean') for sp in SPLITS for h in (3,6) for o in ('oracle_same','oracle_transfer') for b in ('uniform','fifo','reject')}
    assert gain_keys==expected
    assert critical_keys=={(t,s,sp,g,h) for t,g in group_pairs if t=='gate' for s in (*SEEDS,'mean') for sp in SPLITS for h in (3,6)}
    assert len(summary['gate_critical_contrasts'])==72 and len(summary['oracle_baseline_comparisons'])==576 and count==1296
    assert summary['original_phase1_go'] is False and summary['joint_policy_stage_ready'] is False and summary['GPU_seconds']==0
    return dict(status='INDEPENDENT_RAW_SCALAR_AUDIT_PASS',raw_rows=len(rows),paired_rows=len(compact),independent_scalar_statistics=count,
                causal_and_equal_capacity_axes=True,no_model_forward_or_training=True,original_phase1_go=False,joint_policy_stage_ready=False)


def main(folder):
    folder=Path(folder);root=Path(__file__).resolve().parent
    protocol=read(root/'protocol.json');assert protocol['bootstrap']==10000
    job=int(os.environ['SLURM_JOB_ID'])
    state=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',str(job)],text=True)
    info=dict(x.split('=',1) for x in state.split() if '=' in x)
    assert info['JobState']=='RUNNING' and info['Account']=='csci_ga_2271-2026fa' and 'gres/gpu' not in info['AllocTRES']
    assert info['NodeList']==socket.gethostname().split('.')[0]
    summary=read(folder/'summary.json')
    assert summary['raw_rows']==124416 and summary['raw_sha256']==sha(folder/'raw_deletions.csv')
    assert summary['paired_sha256']==sha(folder/'paired_policy_losses.csv')
    with (folder/'raw_deletions.csv').open() as f:rows=list(csv.DictReader(f))
    with (folder/'paired_policy_losses.csv').open() as f:compact=list(csv.DictReader(f))
    result=audit_rows(rows,compact,summary)
    assets=read(folder/'asset_ledger.json');assert sha(folder/'asset_ledger.json')==summary['asset_ledger_sha256']
    assert all(sha(path)==value for path,value in assets.items())
    lock=read(root/'source_lock.json');assert all(sha(root/p)==v for p,v in lock['files'].items())
    for directory in (Path(protocol['predictor_root'])/'control',Path(protocol['old_assay_root'])):
        old_lock=read(directory/'source_lock.json');assert all(sha(directory/p)==v for p,v in old_lock['files'].items())
    result.update(summary_sha256=sha(folder/'summary.json'),raw_sha256=sha(folder/'raw_deletions.csv'),paired_sha256=sha(folder/'paired_policy_losses.csv'),assets_stable=True,
        current_source_lock_sha256=sha(root/'source_lock.json'),native_allocation=state)
    (folder/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main(sys.argv[1])
