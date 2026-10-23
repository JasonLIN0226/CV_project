"""Read compact execution progress through a currently RUNNING pinned route.

No scientific metrics, model weights, or test records are returned. Stage-boundary
artifacts still require fetch_confirmation_compact.py's three-SHA transfer.
"""
import argparse,json,shlex,subprocess
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--alias',required=True);p.add_argument('--route-job',type=int,required=True)
p.add_argument('--remote-root',required=True);p.add_argument('--output',type=Path)
a=p.parse_args()
code='''import pathlib,json,subprocess,datetime
R=pathlib.Path(REMOTE)
D=pathlib.Path(json.loads((R/'config.json').read_text())['output']) if (R/'config.json').exists() else R/'fresh'
raw=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',str(JOB)],text=True,timeout=15)
f=dict(s.split('=',1) for s in raw.split() if '=' in s)
assert f['JobState']=='RUNNING' and f['Account']=='csci_ga_2271-2026fa'
x={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'route_job':JOB,'native_route_state':f['JobState'],
 'squeue':subprocess.check_output(['/opt/slurm/bin/squeue','-h','-u','xl4358','-o','%i|%j|%T|%M|%N'],text=True,timeout=15),
 'receipts':{},'completed':{},'failures':{},'guardian':{},'progress':{}}
for category,folder in [('receipts','receipts'),('completed','stage_completed'),('failures','stage_failures')]:
 for q in sorted((R/folder).glob('*.json')):
  if 'attempt' in q.name:continue
  assert q.stat().st_size<100000
  x[category][q.name]=json.loads(q.read_text())
for n in ['watchdog_submission.json','watchdog_cleanup_submission.json','guardian_native_preflight.json','watchdog_terminal.json','guardian_peer_terminal.json']:
 q=R/n
 if q.exists():x['guardian'][n]=json.loads(q.read_text())
q=R/'watchdog_status.json'
if q.exists():
 d=json.loads(q.read_text());x['guardian']['heartbeat']={k:d[k] for k in ['job','updated_epoch','charged_GPU_seconds','reason'] if k in d}
for n in ['sequence_budget.json','last_budget_forecast.json','budget_stop.json']:
 q=D/n
 if q.exists():
  d=json.loads(q.read_text());x['progress'][n]={k:d[k] for k in ['GPU_job_ids','epochs','encoded_frames','actual_GPU_seconds','remaining_cap_seconds','fits','projected_cumulative_seconds'] if k in d}
# Return concise progress; boundary proofs are fetched separately with three SHA checks.
for category in ['receipts','completed']:
 x[category]={n:{k:d[k] for k in ['job','parent','execution_completed'] if k in d} for n,d in x[category].items()}
for n,d in list(x['guardian'].items()):
 if n=='heartbeat':continue
 x['guardian'][n]={k:d[k] for k in ['job','status','guardian','cleanup','reason','real_paired_heartbeat_cycles'] if k in d}
print(json.dumps(x))
'''.replace('REMOTE',repr(a.remote_root)).replace('JOB',repr(a.route_job))
r=subprocess.run(['ssh','-F','/private/tmp/phase1-hpc-control/config','-o','BatchMode=yes',a.alias,shlex.join(['/scratch/xl4358/phase1_env/bin/python','-c',code])],capture_output=True,text=True,check=True,timeout=45)
x=json.loads(r.stdout)
if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(x,indent=2)+'\n')
print(json.dumps(x))
