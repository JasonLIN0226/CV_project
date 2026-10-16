"""Native transfer of explicitly allowed completed scalar artifacts, never payloads."""
import argparse,datetime,hashlib,json,os,shlex,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--alias',required=True);p.add_argument('--verified-control-job',type=int,required=True);a=p.parse_args()
assert a.verified_control_job==7050
root=Path(__file__).resolve().parent/'evaluation_results'
config='/private/tmp/phase1-hpc-control/config'
ssh=['ssh','-F',config,'-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1',a.alias]
scp=['scp','-F',config,'-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1']
base='/scratch/xl4358/phase1_v5_motion_evaluation'
files={f'{base}/{rel}':rel for rel in ('development_comparison.json','all_models_completion.json','report.md','evaluation/raw_records.csv','evaluation/metrics.json','evaluation/calibration_summary.json','evaluation/latency_memory.json','root_current_weights_recheck_20261013T0740.json')}
for control,name in (('phase1_jepa_memory_v5_motion','train-7000'),('phase1_jepa_memory_v5_motion_evaluation','eval-7001')):
 for ext in ('out','err'):files[f'/scratch/xl4358/{control}/logs/{name}.{ext}']=f'logs/{name}.{ext}'
for seed in (17,29,43):files[f'/scratch/xl4358/phase1_v5_motion_development/predictor/motion/seed_{seed}/training.json']=f'training_curves/seed_{seed}/training.json'
# This exact allowlist only contains JSON, Markdown, scalar CSV and logs.
remote="""import datetime,hashlib,json,socket,subprocess
from pathlib import Path
files=FILES
q=subprocess.check_output(['/opt/slurm/bin/squeue','-h','-j','7050','-o','%i|%a|%T|%N'],text=True)
assert '7050|csci_ga_2271-2026fa|RUNNING|'+socket.gethostname() in q
result={}
for name in files:
 p=Path(name)
 assert p.is_file() and all(not x.is_symlink() for x in (p,*p.parents))
 assert p.stat().st_size<=20000000
 payload=p.read_bytes();payload.decode('utf-8')
 result[name]={'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}
print(json.dumps({'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':socket.gethostname(),'squeue':q,'files':result}))
""".replace('FILES',repr(list(files)))
def manifest():
 cmd='/scratch/xl4358/phase1_env/bin/python -c '+shlex.quote(remote)
 return json.loads(subprocess.check_output(ssh+[cmd],text=True,timeout=90))
before=manifest();root.mkdir(parents=True,exist_ok=True)
(root/'transfer_before.json').write_text(json.dumps(before,indent=2)+'\n')
for src,rel in files.items():
 dest=root/rel;dest.parent.mkdir(parents=True,exist_ok=True)
 partial=dest.with_name(dest.name+'.native_transfer_partial')
 assert not partial.exists()
 if dest.exists():
  assert hashlib.sha256(dest.read_bytes()).hexdigest()==before['files'][src]['sha256'];continue
 subprocess.run(scp+[a.alias+':'+src,str(partial)],check=True,timeout=90)
 assert partial.stat().st_size==before['files'][src]['bytes'] and hashlib.sha256(partial.read_bytes()).hexdigest()==before['files'][src]['sha256']
 os.replace(partial,dest)
after=manifest();assert before['files']==after['files']
(root/'transfer_after.json').write_text(json.dumps(after,indent=2)+'\n')
for src,rel in files.items():
 dest=root/rel
 assert dest.stat().st_size==after['files'][src]['bytes'] and hashlib.sha256(dest.read_bytes()).hexdigest()==after['files'][src]['sha256']
proof={'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'control_connection_job':7050,'before_local_after_equal':True,'files':{rel:after['files'][src] for src,rel in files.items()},'large_payloads_or_symlinks_transferred':False,'new_science':False}
(root/'transfer_integrity.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps({'files':len(files),'all_before_local_after_sha_equal':True}))
