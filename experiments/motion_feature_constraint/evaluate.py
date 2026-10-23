"""Re-score retained validation vectors only; no predictor or encoder forward."""
import csv,datetime,hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path
import numpy as np
from core import maps,constrain,oracle_feature,interval

ROOT=Path(__file__).resolve().parent
PARENT=Path('/scratch/xl4358/phase1_motion_geometry_validation')
OLD=PARENT/'output/job_8336'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def csvread(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def csvwrite(p,rows):
    with Path(p).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main(job):
    begin=time.monotonic();protocol=read(ROOT/'protocol.json')
    allocation=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',str(job)],text=True);native=dict(t.split('=',1) for t in allocation.split() if '=' in t)
    assert str(job)==os.environ['SLURM_JOB_ID'] and native['JobState']=='RUNNING' and native['Partition']=='n2c48m24' and native['Account']=='csci_ga_2271-2026fa'
    assert native['NodeList']==socket.gethostname().split('.')[0] and 'gres/gpu' not in native['AllocTRES'] and os.environ['CUDA_VISIBLE_DEVICES']==''
    assert all(sha(ROOT/k)==v for k,v in read(ROOT/'source_lock.json')['files'].items())
    assert sha(OLD/'summary.json')==protocol['parent_summary_sha256'] and sha(PARENT/'source_lock.json')==protocol['parent_source_lock_sha256']
    parent=read(OLD/'summary.json');assert read(PARENT/'execution_receipt.json')['status']=='COMPLETE_WITH_INDEPENDENT_AUDIT'
    assert read(OLD/'independent_audit.json')['status']=='PASS'
    assert sha(OLD/'validation_geometry.csv')==parent['raw_sha256'] and sha(OLD/'prediction_witnesses.npz')==parent['witness_sha256']
    assets=read(OLD/'asset_ledger.json');assert all(sha(k)==v for k,v in assets.items())
    before=dict(assets)
    out=ROOT/'output'/f'job_{job}';out.mkdir(parents=True,exist_ok=False)
    write(out/'attempt.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),job=job,allocation=allocation,model_forward=False,training=False,fitting=False,GPU=0))
    rows=csvread(OLD/'validation_geometry.csv');assert len(rows)==1152
    with np.load(OLD/'prediction_witnesses.npz',allow_pickle=False) as f:p=f['predictions'][:,1,:].astype(np.float64);recent=f['predictions'][:,0,:].astype(np.float64);target=f['targets'].astype(np.float64)
    assert p.shape==target.shape==(1152,5120)
    ref=Path('/scratch/xl4358/phase1_v3_dev/motion/reference')
    with np.load(ref/'predictor/motion/basis.npz',allow_pickle=False) as f:B=f['output_basis'].copy()
    with np.load(ref/'probes/motion_state_decoder.npz',allow_pickle=False) as f:decoder={k:f[k] for k in f.files}
    B,A,b,right,gram=maps(B,decoder['weight'],decoder['std'],decoder['mean'],decoder['bias'])
    # Only legal decoded-geometry fields enter the normal feature correction.
    g=np.array([[float(r['kinematic_actions_x']),float(r['kinematic_actions_y'])] for r in rows])
    normal,c=constrain(p,g,B,A,b,right)
    position_residual=float(np.max(np.abs(normal@A+b-g)));assert position_residual<1e-8
    lookup={(int(r['seed']),r['pair'],int(r['horizon'])):[] for r in rows}
    for i,r in enumerate(rows):lookup[int(r['seed']),r['pair'],int(r['horizon'])].append(i)
    pairs=sorted({r['pair'] for r in rows});swapped=np.zeros_like(g)
    for seed in (17,29,43):
        for h in (3,6):
            for j,pair in enumerate(pairs):
                own=sorted(lookup[seed,pair,h],key=lambda i:rows[i]['episode']);other=sorted(lookup[seed,pairs[(j+1)%96],h],key=lambda i:rows[i]['episode'])
                assert len(own)==len(other)==2
                swapped[own]=g[other]
    negative,cn=constrain(p,swapped,B,A,b,right)
    # Truth-dependent arms are isolated scoring-only controls, constructed after normal predictions.
    truth=np.array([[float(r['target_x']),float(r['target_y'])] for r in rows])
    oracle_xy,co=constrain(p,truth,B,A,b,right)
    oracle_B,cb=oracle_feature(p,target,B,gram)
    arms={'baseline':p,'geometry_constraint':normal,'shuffled_geometry_control':negative,'oracle_position_constraint':oracle_xy,'oracle_full_basis':oracle_B}
    raw=[];losses={}
    for name,pred in arms.items():
        xy=pred@A+b;feature=np.mean((pred-target)**2,axis=1);pos=np.mean((xy-truth)**2,axis=1);losses[name]=(feature,pos)
        assert np.isfinite(pred).all() and np.isfinite(feature).all() and np.isfinite(pos).all()
        for i,r in enumerate(rows):raw.append(dict(seed=int(r['seed']),pair=r['pair'],episode=r['episode'],horizon=int(r['horizon']),arm=name,feature_mse=float(feature[i]),position_mse=float(pos[i]),delta_feature_mse=float(np.mean((pred[i]-p[i])**2)),prediction_x=float(xy[i,0]),prediction_y=float(xy[i,1])))
    replay=max(float(np.max(np.abs(losses['baseline'][0]-[float(r['privileged_small_budget_feature_mse']) for r in rows]))),float(np.max(np.abs(losses['baseline'][1]-[float(r['privileged_small_budget_mse']) for r in rows]))))
    assert replay<1e-7
    csvwrite(out/'validation_feature_constraint.csv',raw)
    np.savez_compressed(out/'constraint_coefficients.npz',geometry=c,negative=cn,oracle_position=co,oracle_basis=cb)
    recent_feature=np.mean((recent-target)**2,axis=1);recent_pos=np.mean((recent@A+b-truth)**2,axis=1)
    cells=[]
    for seed in (17,29,43,'mean'):
        for h in (3,6):
            ids=[i for i,r in enumerate(rows) if int(r['horizon'])==h and (seed=='mean' or int(r['seed'])==seed)]
            groups=[rows[i]['pair'] for i in ids]
            for arm,(feature,pos) in losses.items():
                cell=dict(seed=seed,horizon=h,arm=arm,feature_mse=float(feature[ids].mean()),position_mse=float(pos[ids].mean()),feature_vs_recent=interval(recent_feature[ids],feature[ids],groups),position_vs_recent=interval(recent_pos[ids],pos[ids],groups,1e-5,1e-6),feature_vs_baseline=interval(losses['baseline'][0][ids],feature[ids],groups))
                cells.append(cell)
    candidate=[x for x in cells if x['arm']=='geometry_constraint']
    qualifies=all(x['feature_vs_recent']['numeric_rule_met'] and x['position_vs_recent']['numeric_rule_met'] and x['feature_vs_baseline']['low']>0 for x in candidate)
    assert all(sha(k)==v for k,v in before.items())
    assert all(sha(ROOT/k)==v for k,v in read(ROOT/'source_lock.json')['files'].items())
    write(out/'asset_ledger.json',before)
    write(out/'summary.json',dict(status='VALIDATION_FEATURE_CONSTRAINT_SUPPORTED_FOR_DELETION_PREPARATION' if qualifies else 'VALIDATION_FEATURE_CONSTRAINT_REJECTED_STOP_BEFORE_DELETION',job=job,allocation=allocation,rows=len(raw),basepairs=96,source_rows=1152,arms=list(arms),cells=cells,feature_numeric_checks=sum(x['feature_vs_recent']['numeric_rule_met'] for x in candidate),position_numeric_checks=sum(x['position_vs_recent']['numeric_rule_met'] for x in candidate),incremental_feature_positive_cells=sum(x['feature_vs_baseline']['low']>0 for x in candidate),numeric_checks_total=8,conditional_deletion_eligible=qualifies,deletion_executed=False,baseline_replay_max_difference=replay,normal_position_constraint_max_difference=position_residual,matrix_dimensions=dict(B=list(B.shape),A=list(A.shape)),raw_sha256=sha(out/'validation_feature_constraint.csv'),coefficients_sha256=sha(out/'constraint_coefficients.npz'),parent_vector_sha256=parent['witness_sha256'],asset_ledger_sha256=sha(out/'asset_ledger.json'),source_lock_sha256=sha(ROOT/'source_lock.json'),protocol_sha256=sha(ROOT/'protocol.json'),model_forward=False,training=False,fitting=False,encoding=False,GPU=0,test_forward=False,fresh9137_opened=False,original_gates_changed=False,original_Phase1_GO=False,elapsed=time.monotonic()-begin))

if __name__=='__main__':
    try:main(int(sys.argv[1]))
    except Exception:
        import traceback
        p=ROOT/'output'/f"job_{os.environ.get('SLURM_JOB_ID','unknown')}";p.mkdir(parents=True,exist_ok=True);write(p/'failure.json',dict(error=traceback.format_exc(),no_automatic_retry=True));raise
