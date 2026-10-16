"""Read-only current-model provenance and independent scalar replay on job7226."""
from pathlib import Path
import csv, datetime, hashlib, json, os, socket, subprocess, sys
from collections import defaultdict

ROOT = Path('/scratch/xl4358')
TRAIN = ROOT/'phase1_v5_real_frame_training/job_7209'
CONTROL = ROOT/'phase1_v5_real_diagnostics_7209'
OUT = ROOT/'phase1_v5_real_diagnostic_results/job_7226'

def digest(path):
    path = Path(path)
    assert path.is_file() and not any(p.is_symlink() for p in (path, *path.parents))
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''): value.update(block)
    return value.hexdigest()

def save(name, value):
    path=OUT/name
    assert not path.exists()
    path.write_text(json.dumps(value, indent=2)+'\n')

assert socket.gethostname() == 'b-10-111'
active = subprocess.check_output(['/opt/slurm/bin/squeue','-h','-j','7226','-o','%i|%T|%a|%P|%N'], text=True).strip()
assert active == '7226|RUNNING|csci_ga_2271-2026fa|n2c48m24|b-10-111'
slurm = subprocess.check_output(['/opt/slurm/bin/sacct','-X','-j','7187,7188,7207,7209,7223,7226','-n','-P','--format=JobIDRaw,State,ExitCode,Account,Partition,Start,End,Elapsed,AllocTRES'], text=True)
parent = [line.split('|') for line in slurm.splitlines() if line.startswith('7209|')][0]
assert parent[:5] == ['7209','COMPLETED','0:0','csci_ga_2271-2026fa','c12m85-a100-1'] and 'gres/gpu=1' in parent[-1]
locks = {
 'phase1_jepa_memory_v5_encoding_frame':'cd667fc4f029660c198353570686a8ed8f069a0e2dab1434e912776f5521985e',
 'phase1_v5_a100_cache_audit_7187_frame':'fc02564863f8af84a79eed2bf0861b210252667c2f70411518a32a9eeb219d00',
 'phase1_jepa_memory_v5_real_frame_training':'1e5ba4b9242469df3bc7f6fe6f03eec53a62cdfd9c47f5fdda73845ebfb5504f',
 'phase1_v5_real_diagnostic_handoff_7209':'870f1e43047e733158dce14cfe14544c7a64d7b362dc8ff80847f70c9b37a3c9',
 'phase1_v5_real_diagnostics_7209':'8d00977db2202c22f303badc1d3008125426224e8147b6692238ddf39625419a',
}
source_proofs=[]
for name, expected in locks.items():
    source=ROOT/name; lock=source/'source_lock.json'
    assert digest(lock)==expected
    files=json.loads(lock.read_text())['files']
    for rel, want in files.items():
        assert not Path(rel).is_absolute() and '..' not in Path(rel).parts
        assert digest(source/rel)==want, rel
    source_proofs.append({'source':str(source),'lock_sha256':expected,'files':len(files),'all_current_files_match':True})
protocol=json.loads((CONTROL/'protocol.json').read_text())
sys.path[:0]=[str(CONTROL/'src'),str(CONTROL/'scripts')]
import torch
torch.set_num_threads(1)
from phase1.v5_real_diagnostics import completion_proof
proof=completion_proof(TRAIN,protocol['actual_checkpoint_files'])
assert proof == json.loads((CONTROL/'actual_all6_model_proof.json').read_text())
assert proof == json.loads((OUT/'all_models_completion.json').read_text())
audit_path=ROOT/'phase1_v5_real_encoding_frame/job_7187/full_cache_audit.json'
assert digest(audit_path)=='afbbc24866cb97c7ae1f8d29e6bb440373fee890088a4e1d6ef53c9c9e3b3649'
cache=json.loads(audit_path.read_text())['actual_feature_files']; assert len(cache)==78
cache_proofs=[]
for row in cache:
    file=Path(row['path']); before=digest(file); size=file.stat().st_size; after=digest(file)
    assert before==after==row['sha256']
    cache_proofs.append({'path':str(file),'bytes':size,'before_sha256':before,'after_sha256':after})
fit_sha=digest(TRAIN/'train_only_fit.npz')
assert fit_sha=='d90f99e4a91d290b7d27311de2820775981afe08dc053dc642e8a40636ed1fb3'

# Independent query -> episode -> recording -> participant replay. No project statistics.
import numpy as np
raw=OUT/'health_scalar_records.csv'; before=digest(raw)
rows=list(csv.DictReader(raw.open())); assert len(rows)==113088
metrics=('mse','persistence_mse','basis_oracle_mse'); paired={}; episode=defaultdict(list)
for row in rows:
    seed=int(row['seed']); horizon=int(row['horizon']); policy=row['policy']
    assert row['split']=='validation' and seed in (17,29,43) and horizon in (1,4,16,64) and policy in ('recent','FIFO','Reservoir','full')
    key=(seed,horizon,row['participant'],row['recording'],int(row['episode']),int(row['time']))
    assert (key,policy) not in paired; paired[(key,policy)]=float(row['mse'])
    vals=[float(row[m]) for m in metrics]; assert np.isfinite(vals).all()
    episode[(seed,horizon,policy,row['participant'],row['recording'],int(row['episode']))].append(vals)
assert all(sum(1 for key,pol in paired if key[0]==s and key[1]==h and pol==p)==2356 for s in (17,29,43) for h in (1,4,16,64) for p in ('recent','FIFO','Reservoir','full'))
recording=defaultdict(list)
for (s,h,p,person,record,ep), vals in episode.items(): recording[(s,h,p,person,record)].append(np.mean(vals,axis=0))
people=defaultdict(list)
for (s,h,p,person,record), vals in recording.items(): people[(s,h,p,person)].append(np.mean(vals,axis=0))
values={key:np.mean(vals,axis=0) for key,vals in people.items()}
people_ids=sorted({key[3] for key in values}); assert len(people_ids)==10
means={str(s):{str(h):{p:{m:float(np.mean([values[(s,h,p,person)][i] for person in people_ids])) for i,m in enumerate(metrics)} for p in ('recent','FIFO','Reservoir','full')} for h in (1,4,16,64)} for s in (17,29,43)}
saved=json.loads((OUT/'visual_health.json').read_text()); replay={}; max_diff=0.
for h in (1,4,16,64):
    replay[str(h)]={}
    arrays=[]
    for s in (17,29,43):
        arrays.append(np.array([values[(s,h,'recent',person)][0]-values[(s,h,'full',person)][0] for person in people_ids]))
    for label, vals in zip(('17','29','43','pooled'),[*arrays,np.mean(arrays,axis=0)]):
        draws=np.random.default_rng(0).integers(0,10,size=(10000,10))
        lower,upper=np.quantile(vals[draws].mean(axis=1),[.025,.975]); point=float(vals.mean())
        actual=saved['full_vs_recent_paired_checks'][str(h)][label]
        delta=max(abs(point-actual['point'][0]),abs(float(lower)-actual['ci95_lower'][0]),abs(float(upper)-actual['ci95_upper'][0]));max_diff=max(max_diff,delta)
        assert delta<1e-12 and bool(lower>0)==actual['supported'][0]
        replay[str(h)][label]={'point':point,'ci95':[float(lower),float(upper)],'supported':bool(lower>0)}
after=digest(raw);assert after==before
health={'status':'INDEPENDENT_ORIGINAL_SCALAR_HIERARCHICAL_REPLAY_MATCHES','rows':len(rows),'raw_bytes':raw.stat().st_size,'raw_sha256':before,'raw_after_sha256':after,'participant_ids':people_ids,'participant_equal_weight_mse':means,'full_vs_recent_checks':replay,'max_difference_from_executed_summary':max_diff,'visual_history_status':saved['visual_history_status'],'new_model_inference_fit_or_training':False,'independent_delegated_review':False,'large_raw_file_stays_hpc':True}
save('root_health_scalar_replay_20261013T2330.json',health)
snapshot={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actual_slurm':slurm,'current_native_route_guard':active,'all_frozen_sources':source_proofs,'all6_actual_current_model_proof':proof,'recomputed_equals_executed_model_proofs':True,'all78_current_cache_files':cache_proofs,'fit_npz_sha256':fit_sha,'health_scalar_replay_sha256':digest(OUT/'root_health_scalar_replay_20261013T2330.json'),'current_diagnostic_report':json.loads((OUT/'diagnostic_report.json').read_text()),'stderr_bytes':(CONTROL/'logs/diagnostic-7226.err').stat().st_size,'no_new_model_inference_fit_or_training':True}
save('root_monitor_all6_health_20261013T2330.json',snapshot)
print(json.dumps({'status':'NATIVE_CURRENT_ALL6_ALL78_ALL_FROZEN_SOURCE_AND_HEALTH_SCALAR_REPLAY_VERIFIED','health_sha256':digest(OUT/'root_health_scalar_replay_20261013T2330.json'),'monitor_sha256':digest(OUT/'root_monitor_all6_health_20261013T2330.json'),'max_scalar_difference':max_diff,'progress':snapshot['current_diagnostic_report'].get('progress')}))
