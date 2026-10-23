"""Read-only terminal evidence and short transfer window; no scientific forward."""
import datetime,hashlib,json,os,socket,subprocess,time
from pathlib import Path
root=Path('/scratch/xl4358/phase1_age_control_a1');out=root/'output/job_8275'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
job=int(os.environ['SLURM_JOB_ID'])
print('SLURM_JOB_ID='+str(job),flush=True);print(socket.gethostname(),flush=True)
print(subprocess.check_output(['hostname','-I'],text=True).strip(),flush=True)
print(Path('/etc/ssh/ssh_host_ed25519_key.pub').read_text().strip(),flush=True)
state=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',str(job)],text=True)
assert 'Account=csci_ga_2271-2026fa ' in state and 'JobState=RUNNING ' in state
alloc=next(t.split('=',1)[1] for t in state.split() if t.startswith('AllocTRES='));assert 'gres/gpu' not in alloc
native=subprocess.check_output(['/opt/slurm/bin/sacct','-n','-P','-j','8272,8273,8275','--format=JobID,State,ExitCode,Start,End,ElapsedRaw,Account,AllocTRES'],text=True)
lines=[s.split('|') for s in native.splitlines()]
assert any(v[:3]==['8275','COMPLETED','0:0'] for v in lines),native
summary=json.loads((out/'summary.json').read_text());audit=json.loads((out/'independent_audit.json').read_text())
assert audit['status']=='INDEPENDENT_RAW_SCALAR_AUDIT_PASS',audit
assert audit['summary_sha256']==sha(out/'summary.json')
assert summary['raw_rows']==124416 and summary['paired_rows']==9216
assert sha(out/'raw_deletions.csv')==summary['raw_sha256']==audit['raw_sha256']
assert sha(out/'paired_policy_losses.csv')==summary['paired_sha256']==audit['paired_sha256']
lock=json.loads((root/'source_lock.json').read_text());assert all(sha(root/k)==v for k,v in lock['files'].items())
proof=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='FINAL_NATIVE_COMPLETION_VERIFIED',science_job=8275,evidence_job=job,native_jobs=native,evidence_allocation=state,raw_rows=124416,paired_rows=9216,summary_sha256=sha(out/'summary.json'),independent_audit_sha256=sha(out/'independent_audit.json'),source_lock_sha256=sha(root/'source_lock.json'),source_files=len(lock['files']),GPU_added=0,no_model_forward=True)
(out/'final_native_evidence.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True)
for _ in range(150):
 if (out/'transfer_finished.json').exists():break
 time.sleep(2)
