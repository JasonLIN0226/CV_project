from pathlib import Path
import datetime, json, os, socket, subprocess, time
HERE = Path(__file__).resolve().parent
D = json.loads((HERE / 'declaration.json').read_text())
assert D['job_id'] == 7423 and socket.gethostname() == D['hostname']
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(n,d):
 p=HERE/n; q=p.with_suffix('.tmp'); q.write_text(json.dumps(d,indent=2)+'\n'); q.replace(p)
def state():
 p=subprocess.run(['/opt/slurm/bin/sacct','-X','-j','7423','-n','-P','-o','JobID,State,Account,Start'],capture_output=True,text=True,timeout=20)
 assert p.returncode == 0, p.stderr
 row=p.stdout.strip().strip('|').split('|')
 assert row[0] == '7423' and row[2] == D['account'] and row[3] == '2026-10-14T21:58:12', row
 return row
initial=state()
assert initial[1] == 'RUNNING', initial
deadline=datetime.datetime.fromisoformat(D['hard_deadline_utc']).timestamp()
save('armed.json', {'status':'ARMED_NOT_YET_TRIGGERED','utc':utc(),'pid':os.getpid(),'job_id':7423,'initial_state':initial,'deadline_utc':D['hard_deadline_utc'],'cancel_request_lead_seconds':D['cancel_request_lead_seconds'],'no_model_or_GPU_work':True})
time.sleep(max(0, deadline-time.time()-35))
try:
 current=state()
 if current[1] not in ['RUNNING','CONFIGURING','COMPLETING','PENDING']:
  save('completion.json',{'status':'ALREADY_TERMINAL_NO_CANCELLATION','utc':utc(),'state':current})
 else:
  time.sleep(max(0,deadline-time.time()-D['cancel_request_lead_seconds']))
  fd=os.open(HERE/'cancellation_attempt.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
  with os.fdopen(fd,'w') as f:json.dump({'utc':utc(),'job_id':7423,'last_verified_state':current,'reason':'original aggregate GPU wall cap'},f)
  p=subprocess.run(['/opt/slurm/bin/scancel','7423'],capture_output=True,text=True,timeout=15)
  save('completion.json',{'status':'CANCELLATION_REQUESTED_AT_ORIGINAL_BUDGET','utc':utc(),'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'actual_final_Slurm_state_requires_later_verification':True})
except Exception as e:
 save('failure.json',{'utc':utc(),'error':str(e),'type':type(e).__name__})
 raise
