import json,csv,datetime,subprocess,io
from pathlib import Path
root=Path('/scratch/xl4358/phase1_independent_confirmation_execution_v2')
args=['/opt/slurm/bin/sacct','-D','-X','-n','-P','-u','xl4358','-A','csci_ga_2271-2026fa','-S','2026-10-19','-E','2026-10-20','-j','8533,8536','--format=JobID,State,ExitCode,ElapsedRaw,Start,End,Submit,Eligible,Account,AllocTRES']
raw=subprocess.run(args,check=True,text=True,capture_output=True).stdout
rows=[];cpu=[]
for line in raw.splitlines():
 f=line.split('|')
 if len(f)==11 and f[-1]=='':f.pop()
 assert len(f)==10
 job,state,exitcode,elapsed,start,end,submit,eligible,account,tres=f
 assert account=='csci_ga_2271-2026fa' and start.startswith('2026-10-19')
 x=dict(job=int(job),state=state,exitcode=exitcode,elapsed_seconds=int(elapsed),start_UTC=start,end_UTC=end,submit_UTC=submit,eligible_UTC=eligible,account=account,AllocTRES=tres)
 if job=='8536':
  assert 'gres/gpu=1' in tres.split(',')
  assert int(elapsed)==int((datetime.datetime.fromisoformat(end)-datetime.datetime.fromisoformat(start)).total_seconds())
  rows.append(x)
 else:cpu.append(x)
rows.sort(key=lambda r:r['start_UTC']);assert len(rows)==57
assert len({(r['start_UTC'],r['end_UTC']) for r in rows})==len(rows)
assert all(rows[i]['end_UTC']<=rows[i+1]['start_UTC'] for i in range(len(rows)-1))
assert all(r['state']=='REQUEUED' for r in rows[:-1]) and rows[-1]['state'].startswith('CANCELLED')
assert len(cpu)==1 and cpu[0]['state']=='COMPLETED' and cpu[0]['exitcode']=='0:0'
assert not (root/'permit.json').exists() and not (root/'preflight_gpu.json').exists()
assert [p.name for p in (root/'stage_attempts').glob('*.json')]==['preflight_cpu.json']
prior=13131;charge=sum(r['elapsed_seconds'] for r in rows)
active=subprocess.run(['/opt/slurm/bin/squeue','-h','-u','xl4358','-o','%i|%T|%j|%b'],check=True,text=True,capture_output=True).stdout
summary=dict(status='FINAL_EXECUTION_BUDGET_EXCEEDED_NO_FRESH_SCIENTIFIC_RESULT',GPU_cap_seconds=28800,prior_GPU_seconds=prior,startup_job=8536,startup_allocation_attempts=len(rows),automatic_requeues=56,startup_GPU_allocated_seconds=charge,total_GPU_allocated_seconds=prior+charge,excess_seconds=prior+charge-28800,final8536=rows[-1],CPU8533=cpu[0],no_fresh9137_data=True,fresh_models=0,required_fresh_models=12,fresh_maps=0,required_fresh_maps=2,phase1_go=False,new_GPU_submission_forbidden=True,no_GPU_batch_entrypoint_evidence=True,native_active_jobs=active,filtered_native_accounting=raw,rows=rows)
(root/'final_budget_boundary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (root/'startup_allocation_attempts.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps({k:v for k,v in summary.items() if k not in ('rows','filtered_native_accounting')},indent=2))
