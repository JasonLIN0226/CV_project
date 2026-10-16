"""Read-only numerical replay of the actual frozen mechanism scalar exports.

Uses no project inference/statistics functions and never fits or trains anything.
"""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
J = json.loads((ROOT/'motion_mechanism.json').read_text())
TRANSFER = json.loads((ROOT/'mechanism_transfer.json').read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name, proof in TRANSFER['files'].items():
    assert proof['matched'] and sha(ROOT/name) == proof['local'] == proof['remote_before'] == proof['remote_after']
for name in ('raw','velocity','oracle'):
    assert sha(ROOT/f'motion_mechanism.{name}.csv') == J[f'{name}_sha256']
assert sha(REPO/'configs/phase1_v4_motion_mechanism_protocol.json') == J['preregistration_sha256']
for name, value in J['diagnostic_file_sha256'].items():
    assert sha(REPO/'src/phase1'/name) == value
with zipfile.ZipFile(REPO.parent/'phase1_jepa_memory_upload.zip') as archive:
    names = archive.namelist()
    for name, value in J['scientific_source_sha256'].items():
        exact = next(n for n in names if n.endswith('/src/phase1/'+name) or n == 'src/phase1/'+name)
        assert hashlib.sha256(archive.read(exact)).hexdigest() == value

def read_csv(name):
    with Path(name).open() as f:
        return list(csv.DictReader(f))

raw = read_csv(ROOT/'motion_mechanism.raw.csv')
oracle = read_csv(ROOT/'motion_mechanism.oracle.csv')
velocity = read_csv(ROOT/'motion_mechanism.velocity.csv')
key_fields = ('seed','task','split','horizon','condition','pair','episode')
metric_fields = J['arms']['reference']['native_reproduction']['compared_metrics']
for rows in (raw,velocity):
    for r in rows:
        r['seed'] = int(r['seed'])
        if 'horizon' in r: r['horizon'] = int(r['horizon'])
for r in raw:
    for m in metric_fields:
        r[m] = float(r[m]); assert np.isfinite(r[m])
for r in oracle:
    r['horizon'] = int(r['horizon'])
    r['oracle_normalized_mse'] = float(r['oracle_normalized_mse'])
    assert np.isfinite(r['oracle_normalized_mse']) and r['target_aware_not_deployable'] == 'True'
assert len(raw) == 69120 and len(oracle) == 1920 and len(velocity) == 5760
raw_keys = [(r['checkpoint_arm'],r['variant'],*(r[k] for k in key_fields)) for r in raw]
assert len(raw_keys) == len(set(raw_keys))
assert all(r['split']=='validation' for r in raw if r['variant']=='pure_p_velocity')
slices = defaultdict(list)
for r in raw:
    slices[(r['checkpoint_arm'],r['variant'],r['seed'],r['split'],r['horizon'],r['condition'])].append(r)
for key, rows in slices.items():
    n = 192 if key[3] == 'validation' else 256
    assert len(rows) == n and len({r['episode'] for r in rows}) == n
    assert set(Counter(r['pair'] for r in rows).values()) == {2}

REPEATS = 2000
max_diff = 0.0
replayed = 0
def compare(a,b):
    global max_diff,replayed
    for k in ('mean','low','high'):
        diff = abs(float(a[k])-float(b[k])); max_diff=max(max_diff,diff)
        assert diff < 2e-12, (k,a,b)
    assert a['pairs']==b['pairs']
    if 'scale' in a: assert a['scale']==b['scale']
    replayed+=1

def interval(values, groups):
    grouped = defaultdict(list)
    for g,v in zip(groups,values): grouped[g].append(v)
    assert set(map(len,grouped.values())) == {2}
    values = np.array([np.mean(v,axis=0) for v in grouped.values()])
    rng=np.random.default_rng(0)
    sampled=values[rng.integers(len(values),size=(REPEATS,len(values)))].mean(axis=1)
    return dict(mean=float(values.mean()),low=float(np.quantile(sampled,.025)),high=float(np.quantile(sampled,.975)),pairs=len(values))

def improvement(a,b,groups,floor,absolute_gain):
    grouped=defaultdict(list)
    for x,y,g in zip(a,b,groups): grouped[g].append([x,y])
    assert set(map(len,grouped.values()))=={2}
    pairs=np.array([np.mean(v,axis=0) for v in grouped.values()])
    absolute=pairs[:,0].mean()<floor
    def effect(x):
        first=x[...,0].mean(axis=-1); second=x[...,1].mean(axis=-1)
        return first-second if absolute else (first-second)/np.maximum(first,floor)
    rng=np.random.default_rng(0)
    boot=effect(pairs[rng.integers(len(pairs),size=(REPEATS,len(pairs)))])
    return dict(mean=float(effect(pairs)),low=float(np.quantile(boot,.025)),high=float(np.quantile(boot,.975)),pairs=len(pairs),scale='absolute_error_reduction' if absolute else 'relative_error_reduction')

native_max=0.0
validation_failures=[]
ceilings=[]
for arm,audit in J['arms'].items():
    c=json.loads((REPO/'runs/phase1_v3_dev/motion'/arm/'resolved_config.json').read_text())
    assert c['evaluation']['bootstrap']==REPEATS and c['training']['seeds']==[17,29,43]
    rows=[r for r in raw if r['checkpoint_arm']==arm]
    native=[r for r in rows if r['variant']=='native']
    old=read_csv(REPO/'runs/phase1_v3_dev/motion'/arm/'evaluation/raw_records.csv')
    old={tuple(int(r[k]) if k in ('seed','horizon') else r[k] for k in key_fields):r for r in old if r['condition']!='persistence'}
    assert len(old)==28800==len(native)
    for r in native:
        prev=old[tuple(r[k] for k in key_fields)]
        for m in metric_fields:
            delta=abs(float(prev[m])-r[m]); native_max=max(native_max,delta); assert delta==0
    assert audit['training_completion']['complete'] and not audit['training_completion']['issues']
    for seed in (17,29,43):
        proof=audit['forward_proofs'][str(seed)]
        for flag in ('clone_exact','weights_unchanged','native_forward_byte_exact','forward_rng_unchanged'): assert proof[flag]
        assert proof['basis_provenance']['checkpoint_buffers_equal_persisted_basis']
        for recent in proof['recent']:
            assert recent['byte_identical'] and len(set(recent['prediction_sha256'].values()))==1
        completion=audit['training_completion']['models'][f'motion/{seed}']
        assert completion['complete'] and completion['completed_epochs']==120 and completion['final_step']==7680
        training=REPO/'runs/phase1_v3_dev/motion'/arm/f'predictor/motion/seed_{seed}/training.json'
        assert sha(training)==completion['training_json_sha256']
    for summary in audit['native_summary']['per_seed']:
        part=[r for r in native if all(r[k]==summary[k] for k in ('seed','split','condition','horizon','task'))]
        for m,saved in summary['metrics'].items(): compare(interval([r[m] for r in part],[r['pair'] for r in part]),saved)
    for check in audit['validation']['checks']:
        seeds=[17,29,43] if check['seed']=='all_seeds_mean' else [check['seed']]
        conditions=['recent_only','full_history','privileged_small_budget','random_small_budget'] if check['kind'].startswith('four') else ['privileged_small_budget']
        horizons=[3,6] if check['kind'].startswith('four') else [check['horizon']]
        part=[r for r in rows if r['split']=='validation' and r['seed'] in seeds and r['condition'] in conditions and r['horizon'] in horizons]
        episodes=sorted({r['episode'] for r in part}); assert len(episodes)==192
        by=defaultdict(list)
        for r in part: by[(r['episode'],r['variant'])].append(r)
        groups=[]; deltas=defaultdict(list)
        for e in episodes:
            x,y=by[(e,'native')],by[(e,'pure_p_velocity')]
            assert len(x)==len(y)==len(seeds)*len(conditions)*len(horizons)
            groups.append(x[0]['pair'])
            for m in ('normalized_mse','state_position_mse'):
                deltas[m].append(float(np.mean([r[m] for r in x])-np.mean([r[m] for r in y])))
        ci=interval(deltas['normalized_mse'],groups); compare(ci,check['absolute_mse_reduction'])
        if 'position_mse_reduction' in check: compare(interval(deltas['state_position_mse'],groups),check['position_mse_reduction'])
        passed=ci['mean']>=0 if check['kind'].startswith('four') else ci['low']>0
        assert passed==check['passed']
        if arm=='reference' and not passed: validation_failures.append({'seed':check['seed'],'horizon':check.get('horizon'),'ci':ci})
    assert audit['validation']['passed']==all(x['passed'] for x in audit['validation']['checks'])
    if arm=='reference':
        lookup={(r['split'],r['horizon'],r['episode']):r for r in oracle}
        assert len(lookup)==len(oracle)
        for check in J['basis_oracle']['checks']:
            seeds=[17,29,43] if check['seed']=='all_seeds_mean' else [check['seed']]
            part=[r for r in native if r['split']==check['split'] and r['horizon']==check['horizon'] and r['seed'] in seeds and r['condition']=='recent_only']
            by=defaultdict(list)
            for r in part: by[r['episode']].append(r)
            episodes=sorted(by); base=[]; floor=[]; groups=[]
            for e in episodes:
                item=lookup[(check['split'],check['horizon'],e)]
                assert len(by[e])==len(seeds) and {r['pair'] for r in by[e]}=={item['pair']}
                base.append(np.mean([r['normalized_mse'] for r in by[e]])); floor.append(item['oracle_normalized_mse']); groups.append(item['pair'])
            assert np.all(np.asarray(floor)<=np.asarray(base)+1e-7+1e-6*np.abs(base))
            ci=improvement(base,floor,groups,c['decision']['baseline_floor'],c['decision']['absolute_latent_gain'])
            compare(ci,check['maximum_possible_improvement'])
            blocks=ci['mean']<check['required_original_threshold']; assert blocks==check['point_ceiling_blocks'];ceilings.append(blocks)
assert not any(ceilings) and not J['basis_oracle']['velocity_training_veto']
assert len(validation_failures)==3 and J['authorize_singlefactor_training'] is False
assert J['weights_unchanged'] and J['caller_rng_restored'] and not J['fitted_any_component']
result={'status':'PASS_READ_ONLY_NUMERICAL_AUDIT','diagnostic_sha256':sha(ROOT/'motion_mechanism.json'),'audit_script_sha256':sha(__file__),
    'raw_rows':len(raw),'unique_complete_slices':len(slices),'velocity_rows':len(velocity),'oracle_rows':len(oracle),
    'native_compared_rows':57600,'native_metric_max_abs_difference':native_max,'intervals_replayed':replayed,'max_interval_difference':max_diff,
    'original_source_zip_and_protocol_hashes_match':True,'validation_primary_failures':validation_failures,
    'basis_point_ceiling_veto':False,'pure_velocity_training_eligible':False,
    'scope':'Independent scalar arithmetic and local compact provenance replay by root; no independent reviewer or new inference/training; no Phase 1 GO.'}
(ROOT/'motion_mechanism_scalar_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
