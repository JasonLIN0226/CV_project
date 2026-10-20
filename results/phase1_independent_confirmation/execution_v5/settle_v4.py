"""Read-only native metadata settlement on an authenticated course CPU node."""
import os,sys,json,pathlib,subprocess,hashlib,socket
R=pathlib.Path('/scratch/xl4358/phase1_independent_confirmation_execution_v4')
sys.path.insert(0,str(R))
from budget_guard import native_allocations
from watchdog import native_job
me=native_job(int(os.environ.get('SLURM_JOB_ID',sys.argv[1])))
assert me['JobState']=='RUNNING' and me['Account']=='csci_ga_2271-2026fa' and 'gres/gpu=' not in me.get('AllocTRES','')
assert me['NodeList']==socket.gethostname().split('.')[0]
rows=native_allocations([8792])
assert len(rows)==1 and rows[0]['state']=='CANCELLED' and rows[0]['exitcode']=='0:0' and rows[0]['tres'].get('gres/gpu')=='1'
assert rows[0]['attempt_count']==1
cpus=native_allocations([8770,8783,8784,8785])
assert all(x['state'] not in ['RUNNING','PENDING','CONFIGURING','COMPLETING'] and 'gres/gpu' not in x['tres'] for x in cpus)
lock=json.loads((R/'source_lock.json').read_text())
assert all(hashlib.sha256((R/n).read_bytes()).hexdigest()==h for n,h in lock['files'].items())
terminal=json.loads((R/'watchdog_terminal.json').read_text());peer=json.loads((R/'guardian_peer_status.json').read_text())
loop=terminal['GPU_jobs']['8792']['observed_UTC']
from datetime import datetime
race_age=datetime.fromisoformat(loop).timestamp()-peer['updated_epoch']
assert -.1 < race_age < 0
out=dict(status='NATIVE_V4_EXECUTION_STOP_SETTLED',GPU_rows=rows,CPU_terminals=cpus,prior_GPU_seconds=39651,new_GPU_seconds=rows[0]['elapsed'],cumulative_GPU_seconds=39651+rows[0]['elapsed'],cap_GPU_seconds=54051,remaining_GPU_seconds=54051-39651-rows[0]['elapsed'],old_computed_heartbeat_age=race_age,all_v4_source_files_match=True,fresh_data_exists=(R/'fresh/data').exists(),actual_CPU=me)
assert not (R/'fresh/datasets').exists()
p=R/'v4_stop_native_settlement.json';p.open('x').write(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
