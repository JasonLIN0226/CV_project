"""Read-only native metadata settlement on an authenticated course CPU node."""
import os,sys,json,pathlib,subprocess,hashlib,socket
R=pathlib.Path('/scratch/xl4358/phase1_independent_confirmation_execution_v5')
sys.path.insert(0,str(R))
from budget_guard import native_allocations
from watchdog import native_job
me=native_job(int(os.environ.get('SLURM_JOB_ID',sys.argv[1])))
assert me['JobState']=='RUNNING' and me['Account']=='csci_ga_2271-2026fa' and 'gres/gpu=' not in me.get('AllocTRES','')
assert me['NodeList']==socket.gethostname().split('.')[0]
rows=native_allocations([8799])
assert len(rows)==1 and rows[0]['state']=='CANCELLED' and rows[0]['exitcode']=='0:0' and rows[0]['tres'].get('gres/gpu')=='1'
assert rows[0]['attempt_count']==1
cpus=native_allocations([8794,8795,8796])
assert all(x['state'] not in ['RUNNING','PENDING','CONFIGURING','COMPLETING'] and 'gres/gpu' not in x['tres'] for x in cpus)
lock=json.loads((R/'source_lock.json').read_text())
assert all(hashlib.sha256((R/n).read_bytes()).hexdigest()==h for n,h in lock['files'].items())
terminal=json.loads((R/'watchdog_terminal.json').read_text());peer=json.loads((R/'guardian_peer_status.json').read_text())
loop=terminal['GPU_jobs']['8799']['observed_UTC']
from datetime import datetime
race_age=datetime.fromisoformat(loop).timestamp()-peer['updated_epoch']
assert terminal['reason']=='Cloud startup exceeded the frozen allocation allowance'
out=dict(status='NATIVE_V5_STARTUP_STOP_SETTLED',GPU_rows=rows,CPU_terminals=cpus,prior_GPU_seconds=39712,new_GPU_seconds=rows[0]['elapsed'],cumulative_GPU_seconds=39712+rows[0]['elapsed'],cap_GPU_seconds=54051,remaining_GPU_seconds=54051-39712-rows[0]['elapsed'],old_computed_heartbeat_age=race_age,all_v5_source_files_match=True,fresh_data_exists=(R/'fresh/data').exists(),actual_CPU=me)
assert not (R/'fresh/datasets').exists()
out['GPU_node_metadata']=subprocess.run(['/opt/slurm/bin/scontrol','show','node','b-19-44','-o'],capture_output=True,text=True,timeout=15).stdout
assert not (R/'stage_attempts/preflight_gpu.json').exists()
out['GPU_batch_never_entered']=True
proof=json.loads((R/'preflight_cpu.json').read_text());assert proof['all_exact'] and proof['samples']==9728 and proof['validation_batches']==24
out['CPU_equivalence_8794_actual_complete']=next(x for x in cpus if x['job']==8794)['state']=='COMPLETED'
assert out['CPU_equivalence_8794_actual_complete']
p=R/'v5_stop_native_settlement.json';p.open('x').write(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
