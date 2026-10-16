"""Explicit compact metadata transfer, guarded live course route and three hashes."""
from pathlib import Path
import argparse, csv, datetime, hashlib, json, shlex, subprocess, sys

ROOT=Path('/scratch/xl4358')
CONFIG='/private/tmp/phase1-hpc-control/config'
parser=argparse.ArgumentParser()
parser.add_argument('--alias',default='phase1-science7268')
parser.add_argument('--live-job',type=int,default=7268)
parser.add_argument('--live-host',default='b-10-102')
parser.add_argument('--final',action='store_true')
args=parser.parse_args()
ALIAS=args.alias
DEST=Path(__file__).parent/'native7251_7268'
DEST.mkdir(exist_ok=True)

def command(args):
    result=subprocess.run(args,capture_output=True,text=True,timeout=40)
    if result.returncode:raise RuntimeError((result.returncode,result.stderr[:2000]))
    return result.stdout

def remote(code):
    return command(['ssh','-F',CONFIG,'-o','BatchMode=yes',ALIAS,
        '/scratch/xl4358/phase1_env/bin/python -c '+shlex.quote(code)])

def metadata(path):
    code='''from pathlib import Path
import hashlib,json
p=Path(PATH)
assert p.is_file() and not any(v.is_symlink() for v in (p,*p.parents))
assert p.suffix in ('.json','.csv','.out','.err') and p.stat().st_size<=1000000
b=p.read_bytes();print(json.dumps({'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}))
'''.replace('PATH',repr(str(path)))
    return json.loads(remote(code))

partition='n2c48m24' if args.live_job==7268 else 'interactive'
if args.live_job!=7268:assert args.final and ALIAS!= 'phase1-science7268'
guard=remote("import subprocess,socket;assert socket.gethostname()=="+repr(args.live_host)+";v=subprocess.check_output(['/opt/slurm/bin/squeue','-h','-j',"+repr(str(args.live_job))+",'-o','%i|%T|%a|%P|%N'],text=True).strip();assert v=="+repr(f'{args.live_job}|RUNNING|csci_ga_2271-2026fa|{partition}|{args.live_host}')+";print(v)")
TRAIN=ROOT/'phase1_v6_real_wd020/job_7251'
CONTROL=ROOT/'phase1_v6_real_diagnostics_7251'
OUT=ROOT/'phase1_v6_real_diagnostic_results/job_7268'
items={
 'training_report.json':TRAIN/'training_report.json',
 'fit_proof.json':TRAIN/'fit_proof.json',
 'source_lock_diagnostic.json':CONTROL/'source_lock.json',
 'protocol.json':CONTROL/'protocol.json',
 'submission.json':CONTROL/'submission.json',
 'all_models_completion_bound.json':CONTROL/'actual_all6_model_proof.json',
 'all_models_completion.json':OUT/'all_models_completion.json',
 'visual_health.json':OUT/'visual_health.json',
 'fixed_old_reference_comparison.json':OUT/'fixed_old_reference_comparison.json',
 'root_health_scalar_replay_7268_20261014T0144.json':OUT/'root_health_scalar_replay_7268_20261014T0144.json',
 'root_fixed_old_reference_replay_7268_20261014T0144.json':OUT/'root_fixed_old_reference_replay_7268_20261014T0144.json',
 'root_monitor_all6_health_7268_20261014T0144.json':OUT/'root_monitor_all6_health_7268_20261014T0144.json',
}
for seed in (17,29,43):
    for name in ('training.csv','checkpoint_proof.json'):
        items[f'seed_{seed}_{name}']=TRAIN/f'seed_{seed}'/name
if args.final:
    final_state=remote("import subprocess;v=subprocess.check_output(['/opt/slurm/bin/sacct','-X','-j','7268','-n','-P','--format=JobIDRaw,State,ExitCode,Account,Partition'],text=True).strip();assert len(v.splitlines())==1;assert v.split('|')[:5]==['7268','COMPLETED','0:0','csci_ga_2271-2026fa','n2c48m24'];print(v)")
    for name in ('diagnostic_report.json','deletion_summary.json','replay_participant_summary.json','root_final_audit_7268.json'):
        items[name]=OUT/name
    items['train-7251.out']=ROOT/'phase1_jepa_memory_v6_real_wd020/logs/train-7251.out'
    items['train-7251.err']=ROOT/'phase1_jepa_memory_v6_real_wd020/logs/train-7251.err'
    items['handoff-7253.out']=ROOT/'phase1_v6_diagnostic_handoff_7251/logs/handoff-7253.out'
    items['handoff-7253.err']=ROOT/'phase1_v6_diagnostic_handoff_7251/logs/handoff-7253.err'
    items['diagnostic-7268.out']=CONTROL/'logs/diagnostic-7268.out'
    items['diagnostic-7268.err']=CONTROL/'logs/diagnostic-7268.err'

proof=[]
for name,path in items.items():
    before=metadata(path);local=DEST/name
    if local.exists():
        value={'bytes':local.stat().st_size,'sha256':hashlib.sha256(local.read_bytes()).hexdigest()}
        if value!=before:raise RuntimeError('Existing local artifact differs; do not overwrite '+name)
    else:
        # Current7271's default SFTP transfer stalled at0bytes; native legacy
        # SCP transport was verified with exact helper hashes on the same route.
        command(['scp','-O','-F',CONFIG,'-o','BatchMode=yes',ALIAS+':'+str(path),str(local)])
    payload=local.read_bytes();value={'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}
    after=metadata(path)
    assert before==value==after,name
    if local.suffix=='.json':json.loads(payload)
    if local.suffix=='.csv':
        rows=list(csv.DictReader(payload.decode().splitlines()));assert len(rows)==30
    proof.append({'remote':str(path),'local':str(local.resolve()),'before':before,'local_proof':value,'after':after,'matched':True})
record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'guard':guard,'files':proof,'explicit_compact_only':True,'weights_features_videos_labels_arrays_downloaded':False}
(DEST/('transfer_integrity_final.json' if args.final else 'transfer_integrity_initial.json')).write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'matched_files':len(proof),'output':str(DEST)}))
