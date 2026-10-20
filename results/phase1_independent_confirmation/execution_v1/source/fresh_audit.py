"""Independent fresh scalar/statistic/vector audit. No model forward or fitting.

Does not import the primary analyzers, inference correction, or statistics code.
The original pair-first bootstrap arithmetic and acceptance thresholds are
reconstructed here from raw rows and primitive vectors.
"""
import collections
import csv
import json
import hashlib
from pathlib import Path
import numpy as np

SEEDS=(17,29,43)
SPLITS=('validation','test_id','test_long_gap','test_distractor_shift')
CONDITIONS=('recent_only','full_history','privileged_small_budget','random_small_budget','wrong_evidence','persistence')


def read(path):return json.loads(Path(path).read_text())
def csvread(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def bounds(values,groups,repeats=2000):
    grouped={}
    for value,group in zip(values,groups):grouped.setdefault(group,[]).append(float(value))
    a=np.array([np.mean(v) for v in grouped.values()])
    ix=np.random.default_rng(0).integers(0,len(a),(repeats,len(a)))
    lo,hi=np.quantile(a[ix].mean(1),[.025,.975])
    return float(a.mean()),float(lo),float(hi)


def improvement(base,alternative,groups,floor):
    grouped={}
    for x,y,group in zip(base,alternative,groups):grouped.setdefault(group,[]).append((float(x),float(y)))
    a=np.array([np.mean(v,0) for v in grouped.values()]);absolute=a[:,0].mean()<floor
    def effect(v):
        x,y=v[...,0].mean(-1),v[...,1].mean(-1)
        return x-y if absolute else (x-y)/np.maximum(x,floor)
    ix=np.random.default_rng(0).integers(0,len(a),(2000,len(a)))
    lo,hi=np.quantile(effect(a[ix]),[.025,.975])
    return float(effect(a)),float(lo),float(hi),bool(absolute)


def load_raw(path,task):
    rows=csvread(path);index={};axes=collections.defaultdict(lambda:collections.defaultdict(set))
    strings={'episode','pair','task','split','condition','sibling'}
    for row in rows:
        if row['task']!=task:raise ValueError('Raw task mismatch')
        for k,v in row.items():
            if k not in strings and not np.isfinite(float(v)):raise ValueError('Nonfinite raw scalar')
        key=(int(row['seed']),row['split'],row['condition'],int(row['horizon']),row['episode'])
        if key in index:raise ValueError('Duplicate raw row')
        index[key]=row;axes[key[:4]][row['pair']].add(row['episode'])
    expected={(s,sp,co,h) for s in SEEDS for sp in SPLITS for co in CONDITIONS for h in (3,6)}
    if set(axes)!=expected or len(rows)!=62208:raise ValueError('Incomplete full fresh task/role grid')
    for (_,sp,_,_),pairs in axes.items():
        if len(pairs)!=(96 if sp=='validation' else 256) or any(len(e)!=2 for e in pairs.values()):
            raise ValueError('Incomplete pair/sibling coverage')
    return rows,index


def audit_task(c,task):
    root=Path(c['output']);out=root/'evaluation/candidate'/task
    raw,index=load_raw(out/'raw_records.csv',task)
    refraw,ref=load_raw(root/'evaluation/reference'/task/'raw_records.csv',task)
    if set(index)!=set(ref):raise ValueError('Fresh reference/candidate axes differ')
    for key,row in index.items():
        if row['pair']!=ref[key]['pair']:raise ValueError('Reference/candidate pair axis differs')
    metrics=read(out/'metrics.json');result=read(root/'evaluation'/task/'comparison.json')
    probes=read(root/'probes/metrics.json');rule=c['decision']
    from phase1.utils import file_hash
    if (metrics['raw_sha256']!=file_hash(out/'raw_records.csv')
            or metrics['all_models_lock_sha256']!=file_hash(root/'all_models_lock.json')):
        raise ValueError('Raw/model-lock checksum mismatch')
    allchecks=[];structural=True;full_ok=True;maxdiff=0.
    for probe in probes['results']:
        if probe['task']!=task or not probe['split'].startswith('test_'):continue
        if probe['variant']==('gate_cue' if task=='gate' else 'motion_two_frame'):
            allchecks.append(dict(kind='evidence_probe',task=task,split=probe['split'],
                passed=bool(probe['accuracy']['mean']>=rule['probe_accuracy'] and probe['wilson_effective_pairs'][0]>.5)))
        elif probe['variant']=='recent_only':structural&=abs(probe['accuracy']['mean']-.5)<=1e-7
        elif probe['variant']=='state_decoder_on_true_target':
            structural&=(probe['accuracy']['mean']>=rule['readout_accuracy'] if task=='gate'
                else np.sqrt(probe['position_mse']['mean'])<=rule['readout_position_rmse'])
    comparison_axes=set()
    for cmp in metrics['comparisons']:
        sp,h,seed,alt=cmp['split'],int(cmp['horizon']),cmp['seed'],cmp['alternative']
        if not sp.startswith('test_'):continue
        axis=(sp,h,seed,alt)
        if axis in comparison_axes:raise ValueError('Duplicate comparative statistic')
        comparison_axes.add(axis)
        seeds=SEEDS if seed=='all_seeds_mean' else (int(seed),)
        episodes=sorted({r['episode'] for r in raw if r['split']==sp})
        if len(episodes)!=512:raise ValueError('All256 fresh pairs required')
        def values(condition,field):
            return np.mean([[float(index[s,sp,condition,h,e][field]) for e in episodes] for s in seeds],0)
        groups=[index[17,sp,'recent_only',h,e]['pair'] for e in episodes]
        name=f'{task}/{sp}/h{h}/seed{seed}'
        if alt=='wrong_evidence':
            got=bounds(values('privileged_small_budget','pair_margin')-values('wrong_evidence','pair_margin'),groups)
            want=cmp['toward_wrong_future']
            maxdiff=max(maxdiff,*[abs(got[i]-want[k]) for i,k in enumerate(('mean','low','high'))])
            allchecks.append(dict(kind='content_intervention',slice=name,
                passed=got[0]>=rule['wrong_evidence_gain'] and got[1]>0))
        else:
            stats={}
            for metric in ('normalized_mse','latent_mse','state_position_mse','state_accuracy','pair_accuracy'):
                a,b=values('recent_only',metric),values(alt,metric)
                if not metric.endswith('mse'):a,b=1-a,1-b
                floor=(rule['position_baseline_floor'] if metric=='state_position_mse' else
                    1e-8 if metric=='latent_mse' else rule['baseline_floor'])
                got=improvement(a,b,groups,floor);want=cmp['improvement'][metric]
                maxdiff=max(maxdiff,*[abs(got[i]-want[k]) for i,k in enumerate(('mean','low','high'))])
                if got[3]!=(want['scale']=='absolute_error_reduction'):raise ValueError('Effect scale changed')
                stats[metric]=got
            if alt=='privileged_small_budget':
                latent=stats['normalized_mse'];state=stats['state_accuracy' if task=='gate' else 'state_position_mse']
                lt=rule['absolute_latent_gain'] if latent[3] else rule['latent_relative_improvement']
                st=(rule['absolute_state_gain'] if task=='gate' else rule['absolute_position_mse_gain']) if state[3] else rule['state_relative_improvement']
                allchecks.append(dict(kind='privileged_benefit',slice=name,
                    passed=latent[0]>=lt and latent[1]>0 and state[0]>=st and state[1]>0))
            elif alt=='full_history':full_ok&=stats['normalized_mse'][0]>=-rule['full_history_tolerance']
    expected={(sp,h,s,alt) for sp in SPLITS[1:] for h in (3,6) for s in (*SEEDS,'all_seeds_mean')
        for alt in ('privileged_small_budget','full_history','random_small_budget','wrong_evidence')}
    if comparison_axes!=expected or len(allchecks)!=51 or maxdiff>1e-12:
        raise ValueError('Original comparative axis or independent CI replay mismatch')
    def checkkey(x):return x['kind'],x.get('slice'),x.get('split')
    if {checkkey(x):bool(x['passed']) for x in allchecks}!={checkkey(x):bool(x['passed']) for x in result['original_checks']}:
        raise ValueError('Original51 acceptance flag mismatch')
    for stat in metrics['per_seed']:
        if not stat['split'].startswith('test_'):continue
        rr=[r for r in raw if int(r['seed'])==stat['seed'] and r['split']==stat['split']
            and int(r['horizon'])==stat['horizon'] and r['condition']==stat['condition']]
        if len(rr)!=512:raise ValueError('Missing seed/split/condition metric slice')
        fraction=float(np.mean([float(r['target_pair_distance'])<rule['pair_distance_floor'] for r in rr]))
        if abs(fraction-stat['almost_indistinguishable_fraction'])>1e-12:raise ValueError('Target separation fraction differs')
        structural&=fraction<=.1
        if stat['condition']=='recent_only':
            accuracy=float(np.mean([float(r['pair_accuracy']) for r in rr]))
            structural&=abs(accuracy-.5)<=1e-7
    possible=(all(bool(x['passed']) for x in allchecks) and bool(full_ok) and bool(structural)
        and read(root/'dataset/split_audit.json')['passed'])
    if possible!=(result['original_status']=='GO'):raise ValueError('Original complete task decision differs')
    fixed=[]
    for check in result['fixed_reference_checks']:
        _,sp,h,seed=check['slice'].split('/');h=int(h[1:]);seed=seed[4:]
        selected=[(k,r) for k,r in index.items() if k[1]==sp and k[2]=='privileged_small_budget'
            and k[3]==h and (seed=='all_seeds_mean' or k[0]==int(seed))]
        grouped={}
        for k,r in selected:
            rk=(k[0],sp,'recent_only',h,k[4])
            grouped.setdefault((r['pair'],r['episode']),[]).append([
                float(ref[k]['normalized_mse']),float(r['normalized_mse']),float(ref[rk]['normalized_mse'])])
        order=sorted(grouped);v=np.array([np.mean(grouped[k],0) for k in order]);groups=[k[0] for k in order]
        if check['kind']=='critical_absolute_mse_reduction':
            got=bounds(v[:,0]-v[:,1],groups);want=check['absolute_mse_reduction'];flag=got[0]>0 and got[1]>0
        else:
            stat=improvement(v[:,2],v[:,1],groups,rule['baseline_floor']);got=stat[:3];want=check['improvement']
            threshold=rule['absolute_latent_gain'] if stat[3] else rule['latent_relative_improvement']
            flag=got[0]>=threshold and got[1]>0
        error=max(abs(got[i]-want[k]) for i,k in enumerate(('mean','low','high')))
        if error>1e-12 or bool(flag)!=bool(check['passed']):raise ValueError('Independent fixed-reference CI/flag mismatch')
        fixed.append(dict(kind=check['kind'],slice=check['slice'],passed=bool(flag),maximum_difference=error))
    if len(fixed)!=32:raise ValueError('All32 fixed-reference checks required')
    return dict(task=task,status='PASS_INDEPENDENT_RAW51_FIXED32',original_possible_go=bool(possible),
        original_checks=51,fixed_checks=32,max_original_statistic_difference=maxdiff,fixed_checks_replay=fixed,
        raw_per_role=len(raw),fresh_reference=True,no_model_forward_or_fitting=True)


def audit_vectors(c,folder,task=None,role=None):
    """Alternative QR geometry constraint, triangle-wave dynamics and RBF replay."""
    from phase1.inputs import FeatureStore
    from phase1.probes import LinearReadout
    from phase1.data import load_episode
    folder=Path(folder);meta=csvread(folder/'witness_metadata.csv')
    with np.load(folder/'witness_vectors.npz',allow_pickle=False) as f:v={k:f[k] for k in f.files}
    P,C,T,H,U=[v[k] for k in ('baseline','prediction','target','reference_history','gate_transport')]
    if P.shape!=C.shape or C.shape!=T.shape or C.shape!=(len(meta),5120) or not np.isfinite(C).all():
        raise ValueError('Invalid full-vector witness axes')
    store=FeatureStore(c,max_cached=8);root=Path(c['output']);decoders={};constraint={};heads={}
    for t in ('gate','motion'):
        decoder=LinearReadout.load(root/f'probes/{t}_state_decoder.npz');decoders[t]=decoder
        if t=='motion':
            A=decoder.weight[:,1:]/decoder.std[:,None];b=decoder.bias[1:]-decoder.mean@A
            with np.load(root/f'predictor/{t}/basis.npz',allow_pickle=False) as f:Q=np.linalg.qr(f['output_basis'].astype(float))[0]
            constraint[t]=(A,b,np.linalg.pinv(Q.T@A)@Q.T)
    for h in (3,6):
        with np.load(root/f'observation_maps/fitted/head_{h}.npz',allow_pickle=False) as f:heads[h]={k:f[k] for k in f.files}
    trace={}
    if (folder/'input_trace.csv').exists():
        trace={(r['episode'],r['condition'],r['kept']):r for r in csvread(folder/'input_trace.csv')}
    maxvector=maxscalar=maxgeometry=0.
    cache=collections.OrderedDict()
    def observed(eid):
        if eid not in cache:
            cache[eid]=load_episode(c,store.records[eid],keys=('rgb','actions'))
            if len(cache)>8:cache.popitem(last=False)
        cache.move_to_end(eid);return cache[eid]
    for i,m in enumerate(meta):
        t=m.get('task',task);rec=store.records[m['episode']];q=rec['query'];h=int(m['horizon'])
        truth=load_episode(c,rec,True,keys=('states','classes'))
        if not np.array_equal(T[i],store.features(rec)[q+h]) or not np.array_equal(v['position'][i],truth['states'][q+h]):
            raise ValueError('Stored witness target differs from actual scoring truth')
        if t=='gate' and role!='reference' and m['condition']!='persistence':want=P[i]-U[i];tolerance=3e-5
        elif t=='motion' and role!='reference' and m['kept'].count(';')==1:
            A,b,R=constraint[t];tr=trace[rec['id'],m['condition'],m['kept']]
            times=tuple(map(int,m['kept'].split(';')));sources=tr['source_ids'].split(';')
            if not 0<=times[0]<times[1]<q-3:raise ValueError('Noncausal retained timestamps')
            evidence=set(map(int,load_episode(c,rec,True,keys=('evidence',))['evidence']))
            expected=[rec['sibling'] if tt in evidence else rec['id'] for tt in times] if m['condition']=='wrong_evidence' else [rec['id'],rec['id']]
            if sources!=expected:raise ValueError('Actual input source differs from fixed intervention')
            own=observed(rec['id']);size=own['rgb'].shape[1];ys=slice(int(.30*size),int(.73*size));xs=slice(int(.10*size),int(.90*size))
            enabled=all(np.any(observed(e)['rgb'][tt,ys,xs]!=own['rgb'][q,ys,xs]) for e,tt in zip(sources,times))
            if enabled!=(m['enabled']=='True'):raise ValueError('Independent observed-RGB guard mismatch')
            z=np.stack([store.features(store.records[e])[tt] for e,tt in zip(sources,times)])
            xy=decoders[t].predict(z)[:,1:];velocity=(xy[1,0]-xy[0,0])/(times[1]-times[0]);geo=xy[1].copy()
            for action in own['actions'][times[1]:q+h]:
                x=geo[0]+velocity+float(np.dot(action,[.0015,.001]))
                if x<.08 or x>.92:
                    band=int(np.floor((x-.08)/.84));phase=(x-.08)%1.68;x=.08+(phase if phase<=.84 else 1.68-phase)
                    if band%2:velocity=-velocity
                geo[0]=x
            maxgeometry=max(maxgeometry,float(np.max(abs(geo-[float(m['geometry_x']),float(m['geometry_y'])]))))
            desired=geo if enabled else P[i].astype(float)@A+b
            want=H[i].astype(float)+(desired-(H[i].astype(float)@A+b))@R
            if enabled:
                f=heads[h];zz=(geo-f['mean'])/f['scale'];tt=(f['train_X']-f['mean'])/f['scale']
                k=np.exp(-np.sum((tt-zz)**2,1)/float(f['bandwidth']));k=k-k.mean()-f['row_mean']+f['grand_mean']
                innovation=k@f['coef']+f['target_mean']
                delta=.5*(((store.features(rec)[q]-want)@f['N']+innovation)@f['N'].T)
                if abs(delta@A).max()>=1e-10:raise ValueError('Innovation changes decoded position')
                want+=delta
            tolerance=1e-7
        else:want=P[i];tolerance=0.
        difference=float(abs(want-C[i]).max());maxvector=max(maxvector,difference)
        if difference>tolerance:raise ValueError('Independent full prediction vector differs')
        decoded=decoders[t].predict(C[i][None])[0]
        scalar=float(np.mean((C[i]-T[i])**2))
        taskerror=(float((decoded[0]>0)!=(truth['classes'][q+h]>0)) if t=='gate' and m['condition']=='deletion'
            else float(np.mean((decoded[1:]-truth['states'][q+h])**2)))
        err=max(abs(scalar-float(m['feature_mse'])),abs(taskerror-float(m['task_error'])))
        maxscalar=max(maxscalar,err)
        if err>2e-6 or maxgeometry>1e-8:raise ValueError('Primitive scalar or geometry witness mismatch')
    return dict(status='PASS_INDEPENDENT_PRIMITIVE_VECTORS',vectors=len(meta),max_vector_difference=maxvector,
        max_scalar_difference=maxscalar,max_geometry_difference=maxgeometry,no_model_forward_or_fitting=True)


def audit_trace(c,folder):
    from phase1.inputs import FeatureStore
    from phase1.data import load_episode
    folder=Path(folder);path=folder/'input_trace.csv'
    if not path.exists():raise ValueError('Actual retained-input trace absent')
    rows=csvread(path);store=FeatureStore(c,max_cached=8);seen=set();cache=collections.OrderedDict()
    def observed(eid):
        if eid not in cache:
            cache[eid]=load_episode(c,store.records[eid],keys=('rgb','actions'))
            if len(cache)>8:cache.popitem(last=False)
        cache.move_to_end(eid);return cache[eid]
    def sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
    for row in rows:
        rec=store.records[row['episode']];q=rec['query'];times=tuple(map(int,row['kept'].split(';')))
        source=row['source_ids'].split(';');key=(rec['id'],row['condition'],times,tuple(source))
        if key in seen or len(times)!=2 or len(source)!=2 or not 0<=times[0]<times[1]<q-3 or int(row['query'])!=q:
            raise ValueError('Trace repeats or violates actual equal-capacity/causal boundary')
        seen.add(key);ev=set(map(int,load_episode(c,rec,True,keys=('evidence',))['evidence']))
        expected=[rec['sibling'] if t in ev else rec['id'] for t in times] if row['condition']=='wrong_evidence' else [rec['id'],rec['id']]
        if source!=expected:raise ValueError('Input source differs from the fixed content intervention')
        z=np.stack([store.features(store.records[e])[t] for e,t in zip(source,times)])
        own=observed(rec['id']);n=own['rgb'].shape[1];ys=slice(int(.30*n),int(.73*n));xs=slice(int(.10*n),int(.90*n))
        enabled=all(np.any(observed(e)['rgb'][t,ys,xs]!=own['rgb'][q,ys,xs]) for e,t in zip(source,times))
        if (enabled!=(row['enabled']=='True') or sha(z)!=row['latent_input_sha256']
                or sha(store.features(rec)[q])!=row['current_input_sha256']
                or sha(own['actions'][times[1]:q+6])!=row['actions_used_sha256']):
            raise ValueError('Actual latent/current/action/RGB input trace does not replay')
    if not rows:raise ValueError('Empty actual-input trace')
    return dict(status='PASS_ALL_ACTUAL_INPUT_TRACES',traces=len(rows),all_latent_current_actions_hashes_exact=True,
        causal_equal_capacity=True,metadata_ids_not_predictor_inputs=True)
