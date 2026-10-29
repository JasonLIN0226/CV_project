"""Independent read-only checks on the completed validation artifacts."""
import collections, hashlib, json, math
from pathlib import Path
root=Path('/scratch/xl4358/clean_future_gate_20261022/gate_supervision_v2_20261029/run_20261029')
p=root/'validation_results.json'
data=json.loads(p.read_text())
rows=data['rows']; traces=data['traces']; timing=data['timings']; summary=data['summary']
policies={'fifo','reservoir','temporal_buckets','visual_diversity','prediction_surprise','attention','attention_proxy_student','deletion_cost_student','manual_keyframe','recent_only'}
online=policies-{'manual_keyframe','recent_only'}
assert len(rows)==11520 and len(traces)==576 and len(timing)==576
seen=set(); episodes={}; counts=collections.Counter(); sums=collections.defaultdict(lambda:collections.Counter())
for row in rows:
    key=(row['seed'],row['episode'],row['condition'],row['horizon'])
    assert key not in seen;seen.add(key)
    assert row['seed'] in (17,29,43) and row['condition'] in policies and row['horizon'] in (3,6) and row['split']=='validation'
    assert all(math.isfinite(float(row[k])) for k in ('state_accuracy','normalized_mse','pair_accuracy','event_hit'))
    key2=(row['seed'],row['episode']);episodes[key2]=row['query']
    for dimension in ('all',str(row['gap'])):
        k=(row['condition'],dimension);counts[k]+=1
        for name in ('state_accuracy','normalized_mse','pair_accuracy','event_hit'):
            sums[k][name]+=float(row[name])
assert len(episodes)==576
trace_keys=set(); transitions=0
for tr in traces:
    k=(tr['seed'],tr['episode']);assert k in episodes and k not in trace_keys;trace_keys.add(k)
    assert set(tr['policies'])==online
    q=episodes[k]
    for name,steps in tr['policies'].items():
        memory=[];assert len(steps)==q-3
        for clock,step in enumerate(steps,4):
            arrival=clock-4;selected=step['selected']
            assert step['clock']==clock and step['previous']==memory and step['arrival']==arrival
            assert len(selected)==min(2,len(memory)+1) and len(set(selected))==len(selected) and set(selected)<=set(memory+[arrival])
            memory=selected;transitions+=1
assert trace_keys==set(episodes)
for policy in policies:
    assert counts[(policy,'all')]==1152
    pool=sum(float(row['state_accuracy']) for row in rows if row['condition']==policy)/1152
    assert abs(pool-summary['pooled_route_accuracy'][policy])<1e-10
out={}
for key,n in sorted(counts.items()):
    policy,gap=key;out.setdefault(policy,{})[gap]={'rows':n,**{name:float(value/n) for name,value in sums[key].items()}}
teachers={}
for seed in (17,29,43):
    seal=json.loads((root/f'teacher_{seed}'/'seal.json').read_text());teachers[str(seed)]=seal['timing']
result={'audit_passed':True,'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'rows':len(rows),'episodes_by_seed':192,'seeds':[17,29,43],'horizons':[3,6],'online_policy_episode_traces':len(traces)*len(online),'transitions_verified':transitions,'gap_values':sorted({str(r['gap']) for r in rows}),'policies':out,'teacher_generation_seconds_by_seed':teachers,'test_accessed':False,'note':'Validation descriptive only; original Gate remains failed; future-aware teachers are offline only.'}
output=root/'validation_independent_audit.json'
with output.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
print(json.dumps(result,sort_keys=True),flush=True)
