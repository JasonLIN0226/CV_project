"""Re-score cached deletion predictions only; zero model/encoder forward or fitting."""
import csv, datetime, hashlib, json, os, socket, subprocess, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
from guard import allow_update
import legacy_analysis as analysis
analysis.GROUPS={'motion':('clock_all',)}; analysis.SPLITS=('validation',)
ROOT=Path(__file__).resolve().parent
PARENT=Path('/scratch/xl4358/phase1_motion_feature_deletion_validation')
OLD=PARENT/'output/job_8339'
AGE=Path('/scratch/xl4358/phase1_age_control_a1')
sys.path.insert(0,str(AGE)); import run as old
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def loadcsv(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def savecsv(p,rows):
    with Path(p).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main(job):
    start=time.monotonic();protocol=read(ROOT/'protocol.json')
    native=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',str(job)],text=True)
    f=dict(t.split('=',1) for t in native.split() if '=' in t)
    assert str(job)==os.environ['SLURM_JOB_ID'] and f['JobState']=='RUNNING' and f['Account']=='csci_ga_2271-2026fa'
    assert f['Partition']=='n2c48m24' and f['NodeList']==socket.gethostname().split('.')[0] and 'gres/gpu' not in f['AllocTRES'] and os.environ['CUDA_VISIBLE_DEVICES']==''
    assert all(sha(ROOT/k)==v for k,v in read(ROOT/'source_lock.json')['files'].items())
    parent=read(OLD/'summary.json');assert sha(OLD/'summary.json')==protocol['parent_summary_sha256']
    assert sha(PARENT/'source_lock.json')==protocol['parent_source_lock_sha256']
    assert all(sha(PARENT/k)==v for k,v in read(PARENT/'source_lock.json')['files'].items())
    assert read(OLD/'independent_audit.json')['status']=='PASS' and read(PARENT/'execution_receipt.json')['status']=='COMPLETE_WITH_INDEPENDENT_AUDIT'
    assert sha(OLD/'raw_deletion_constraint.csv')==parent['raw_sha256'] and sha(OLD/'baseline_vectors.npz')==parent['vectors_sha256']
    assets=read(OLD/'asset_ledger.json');assert all(sha(k)==v for k,v in assets.items())
    _,_,configs=old.verify_control(old.OLD/'control');c=configs['motion'];store=old.FeatureStore(c,max_cached=1)
    recs={r['id']:r for r in store.manifest['records'] if r['task']=='motion' and r['split']=='validation'};assert len(recs)==192
    source=loadcsv(OLD/'raw_deletion_constraint.csv');assert len(source)==41472
    by=defaultdict(dict)
    for r in source:
        assert r['arm'] not in by[int(r['witness_index'])];by[int(r['witness_index'])][r['arm']]=r
    assert set(by)==set(range(20736))
    # The gate receives only the actual kept RGB and current RGB. No deleted/future inputs.
    decision={};receipts=[]
    for eid,rec in recs.items():
        obs=Path(c['output'])/'dataset'/rec['obs'];assets[str(obs)]=sha(obs)
        q=int(rec['query']);rgb=old.load_episode(c,rec,keys=('rgb',))['rgb']
        sets=sorted({tuple(map(int,x['baseline']['kept'].split(';'))) for x in by.values() if x['baseline']['episode']==eid})
        for kept in sets:
            assert len(kept)==2 and max(kept)<q-3
            enabled=allow_update(rgb[list(kept)],rgb[q]);decision[eid,kept]=enabled
            receipts.append(dict(episode=eid,query=q,kept=';'.join(map(str,kept)),allow_update=enabled,observation_sha256=assets[str(obs)]))
        if time.monotonic()-start>900:raise RuntimeError('CPU bound exceeded; partial only')
    out=ROOT/'output'/f'job_{job}';out.mkdir(parents=True,exist_ok=False)
    write(out/'attempt.json',dict(job=job,allocation=native,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),GPU=0,model_forward=False))
    raw=[];counts=defaultdict(lambda:[0,0]);fallback_max=0.
    for i in range(20736):
        rr=by[i];base=rr['baseline'];geo=rr['geometry_constraint'];kept=tuple(map(int,base['kept'].split(';')))
        enabled=decision[base['episode'],kept];chosen=geo if enabled else base
        # Visibility is copied only for posthoc scoring; never enters the decision.
        group=int(base['retained_visible_count']);counts[group][int(enabled)]+=1
        for arm,s in [('baseline',base),('unconditional',geo),('observed_guard',chosen)]:
            r=dict(s,arm=arm,allow_update=enabled);raw.append(r)
        if not enabled:fallback_max=max(fallback_max,abs(float(chosen['feature_mse'])-float(base['feature_mse'])),abs(float(chosen['task_error'])-float(base['task_error'])))
    assert len(raw)==62208 and fallback_max==0
    savecsv(out/'raw_guard.csv',raw);savecsv(out/'observed_decisions.csv',receipts)
    summaries={};paired=[]
    for arm in ('baseline','unconditional','observed_guard'):
        compact,coverage=analysis.aggregate([r for r in raw if r['arm']==arm],pairs=96)
        summaries[arm]={'statistics':analysis.summarize(compact,repeats=10000,pairs=96),'coverage':coverage}
        paired.extend([dict(arm=arm,**r) for r in compact])
    savecsv(out/'paired_guard.csv',paired)
    changes=[]
    for seed in (17,29,43,'mean'):
        for h in (3,6):
            for metric in ('feature_mse','task_error'):
                arrays={}
                for arm in ('baseline','unconditional','observed_guard'):
                    rr=[r for r in paired if r['arm']==arm and int(r['horizon'])==h and (seed=='mean' or int(r['seed'])==seed)]
                    pp=sorted({r['pair'] for r in rr});arrays[arm]=np.array([np.mean([float(r['uniform_'+metric]) for r in rr if r['pair']==p]) for p in pp]);assert len(pp)==96
                b=arrays['baseline'];ix=np.random.default_rng(0).integers(0,96,(10000,96))
                for arm in ('unconditional','observed_guard'):
                    x=arrays[arm];v=(b[ix].mean(1)-x[ix].mean(1))/b[ix].mean(1);lo,hi=np.percentile(v,[2.5,97.5])
                    changes.append(dict(seed=seed,horizon=h,metric=metric,arm=arm,baseline=float(b.mean()),candidate=float(x.mean()),relative_reduction=float((b.mean()-x.mean())/b.mean()),low=float(lo),high=float(hi),positive_ci=bool(lo>0)))
    assert all(sha(k)==v for k,v in assets.items());write(out/'asset_ledger.json',assets)
    write(out/'summary.json',dict(status='COMPLETE_VALIDATION_OBSERVED_GUARD_DIAGNOSTIC_NOT_ACCEPTANCE',job=job,allocation=native,raw_rows=len(raw),paired_rows=len(paired),basepairs=96,witness_vectors=20736,decision_rows=len(receipts),activation_by_scoring_visible_count=dict(counts),fallback_exact=True,summaries=summaries,changes=changes,raw_sha256=sha(out/'raw_guard.csv'),paired_sha256=sha(out/'paired_guard.csv'),decision_sha256=sha(out/'observed_decisions.csv'),asset_ledger_sha256=sha(out/'asset_ledger.json'),parent_summary_sha256=sha(OLD/'summary.json'),parent_vector_sha256=parent['vectors_sha256'],source_lock_sha256=sha(ROOT/'source_lock.json'),protocol_sha256=sha(ROOT/'protocol.json'),elapsed=time.monotonic()-start,training=False,fitting=False,encoding=False,model_forward=False,GPU=0,policy_training=False,fresh9137_opened=False,test_forward=False,original_gates_changed=False,original_Phase1_GO=False,synthetic_renderer_specific=True,RGB_observations_new_guard_modality=True))

if __name__=='__main__':
    try:main(int(sys.argv[1]))
    except Exception:
        import traceback
        out=ROOT/'output'/f"job_{os.environ.get('SLURM_JOB_ID','unknown')}";out.mkdir(parents=True,exist_ok=True);write(out/'failure.json',dict(error=traceback.format_exc(),no_automatic_retry=True));raise
