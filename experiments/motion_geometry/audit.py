"""Independent scalar/vector and geometric replay; no predictor forward or fitting."""
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x): Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def independent_roll(position, velocity, actions, use_actions):
    xy=np.array(position,dtype=np.float64); v=float(velocity)
    for a in actions:
        x=xy[0]+v+(float(np.dot(a,[.0015,.001])) if use_actions else 0.)
        if x<.08 or x>.92:
            width=.84; band=int(np.floor((x-.08)/width)); cycle=(x-.08)%(2*width)
            x=.08+(cycle if cycle<=width else 2*width-cycle)
            if band%2:v=-v
        xy[0]=x
    return xy

def main(out):
    out=Path(out);s=json.loads((out/'summary.json').read_text())
    assert sha(out/'validation_geometry.csv')==s['raw_sha256']
    assert sha(out/'prediction_witnesses.npz')==s['witness_sha256']
    assert sha(out/'asset_ledger.json')==s['asset_ledger_sha256']
    assets=json.loads((out/'asset_ledger.json').read_text());assert all(sha(k)==v for k,v in assets.items())
    with (out/'validation_geometry.csv').open() as f: raw=list(csv.DictReader(f))
    keys=[(int(r['seed']),r['episode'],int(r['horizon'])) for r in raw]
    assert len(keys)==len(set(keys))==1152
    assert {int(r['seed']) for r in raw}=={17,29,43} and {int(r['horizon']) for r in raw}=={3,6}
    for seed in (17,29,43):
        for h in (3,6):
            group=[r for r in raw if int(r['seed'])==seed and int(r['horizon'])==h]
            assert len(group)==192 and len({r['pair'] for r in group})==96
            assert all(sum(x['pair']==r['pair'] for x in group)==2 for r in group)
    AGE=Path('/scratch/xl4358/phase1_age_control_a1');sys.path.insert(0,str(AGE));import run as old
    _,_,configs=old.verify_control(old.OLD/'control');c=configs['motion'];store=old.FeatureStore(c,max_cached=2)
    records={r['id']:r for r in store.manifest['records'] if r['task']=='motion' and r['split']=='validation'}
    decoder=old.LinearReadout.load(Path(c['output'])/'probes/motion_state_decoder.npz')
    with np.load(out/'prediction_witnesses.npz',allow_pickle=False) as f: pred=f['predictions'];target=f['targets']
    assert pred.shape==(1152,3,5120) and target.shape==(1152,5120) and np.isfinite(pred).all() and np.isfinite(target).all()
    xy=decoder.predict(pred.reshape(-1,5120)).reshape(1152,3,3)[:,:,1:]
    maximum=0.
    methods=['kinematic_actions','kinematic_no_actions','hold_last_visible','oracle_position_only','oracle_velocity_only','oracle_both','true_target_readout','recent_only','privileged_small_budget','full_history']
    for i,r in enumerate(raw):
        q=int(r['query']);end=int(r['end']);start=int(r['start']);h=int(r['horizon']);rec=records[r['episode']]
        assert int(rec['query'])==q and start<end<q-3
        a=old.load_episode(c,rec,keys=('actions',))['actions'][end:q+h]
        t=np.array([float(r['target_x']),float(r['target_y'])]);obs=np.array([[float(r['observed_start_x']),float(r['observed_start_y'])],[float(r['observed_end_x']),float(r['observed_end_y'])]])
        states=old.load_episode(c,rec,True,keys=('states',))['states'];assert np.array_equal(t,states[q+h].astype(np.float64))
        v=(obs[1,0]-obs[0,0])/(end-start);tv=(float(r['true_end_x'])-float(r['true_start_x']))/(end-start)
        for k,p in [('kinematic_actions',independent_roll(obs[1],v,a,True)),('kinematic_no_actions',independent_roll(obs[1],v,a,False)),('oracle_position_only',independent_roll(states[end],v,a,True)),('oracle_velocity_only',independent_roll(obs[1],tv,a,True)),('oracle_both',independent_roll(states[end],tv,a,True))]:
            maximum=max(maximum,float(np.max(np.abs(p-[float(r[k+'_x']),float(r[k+'_y'])]))));assert maximum<1e-8
        for j,k in enumerate(['recent_only','privileged_small_budget','full_history']):
            feature=float(np.mean((pred[i,j]-target[i])**2));assert abs(feature-float(r[k+'_feature_mse']))<1e-7
            assert np.max(np.abs(xy[i,j]-[float(r[k+'_x']),float(r[k+'_y'])]))<1e-8
        for k in methods:
            pos=np.array([float(r[k+'_x']),float(r[k+'_y'])]);assert abs(float(np.mean((pos-t)**2))-float(r[k+'_mse']))<1e-10
        assert np.array_equal(target[i],store.features(rec)[q+h])
    cells=[];pairrows=[]
    for h in (3,6):
        subset=[r for r in raw if int(r['horizon'])==h];pairs=sorted({r['pair'] for r in subset})
        bypair={p:[r for r in subset if r['pair']==p] for p in pairs};assert all(len(v)==6 for v in bypair.values())
        errors={m:np.array([np.mean([float(r[m+'_mse']) for r in bypair[p]]) for p in pairs]) for m in methods}
        rng=np.random.default_rng(0);ix=rng.integers(0,96,size=(2000,96))
        for m in methods:
            samples=errors[m][ix].mean(1);lo,hi=np.percentile(samples,[2.5,97.5]);cells.append(dict(horizon=h,method=m,mean=float(errors[m].mean()),low=float(lo),high=float(hi),pairs=96))
        for m in ['hold_last_visible','recent_only','privileged_small_budget','full_history','oracle_position_only','oracle_velocity_only','oracle_both']:
            # Positive = that comparator has more error than the legal geometry diagnostic.
            diff=errors[m]-errors['kinematic_actions'];lo,hi=np.percentile(diff[ix].mean(1),[2.5,97.5])
            cells.append(dict(horizon=h,method=m+'_minus_kinematic_actions',mean=float(diff.mean()),low=float(lo),high=float(hi),pairs=96))
        for j,p in enumerate(pairs):pairrows.append(dict(pair=p,horizon=h,**{m:float(errors[m][j]) for m in methods}))
    with (out/'paired_position_errors.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(pairrows[0]));w.writeheader();w.writerows(pairrows)
    extra={}
    once=[r for r in raw if int(r['seed'])==17 and int(r['horizon'])==3]
    extra['visible_velocity_RMSE']=float(np.sqrt(np.mean([(float(r['estimated_velocity'])-float(r['true_velocity']))**2 for r in once])))
    extra['visible_start_position_MSE']=float(np.mean([np.mean([(float(r['observed_start_'+ax])-float(r['true_start_'+ax]))**2 for ax in ['x','y']]) for r in once]))
    extra['visible_end_position_MSE']=float(np.mean([np.mean([(float(r['observed_end_'+ax])-float(r['true_end_'+ax]))**2 for ax in ['x','y']]) for r in once]))
    write(out/'independent_audit.json',dict(status='PASS',rows=1152,pairs=96,all_raw_axes_finite_and_paired=True,all_vector_losses_recomputed=True,geometric_replay_max_difference=maximum,statistics=cells,visible_readout_diagnostics=extra,paired_sha256=sha(out/'paired_position_errors.csv'),scope='Original validation diagnosis, not a new gate or confirmation',GPU=0,predictor_forward=False,training=False))

if __name__=='__main__':main(sys.argv[1])
