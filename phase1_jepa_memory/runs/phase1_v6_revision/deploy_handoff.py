from pathlib import Path
import os,socket,json,zipfile,hashlib,subprocess,datetime
control=Path('/scratch/xl4358/phase1_v6_diagnostic_handoff_7251')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
actual=subprocess.check_output(['/opt/slurm/bin/sacct','-X','-j','7243','-n','-P','--format=JobIDRaw,State,Account,Partition,NodeList'],text=True).strip().split('|')
assert actual[:5]==['7243','RUNNING','csci_ga_2271-2026fa','interactive',socket.gethostname()]
assert not control.exists();control.mkdir(mode=0o700);(control/'logs').mkdir()
with zipfile.ZipFile('/scratch/xl4358/v6_posttrain_handoff_bundle.zip') as z:
 for n in z.namelist():assert not Path(n).is_absolute() and '..' not in Path(n).parts
 z.extractall(control)
proof=json.loads(Path('/scratch/xl4358/v6_posttrain_preparation.json').read_text())
assert sha(control/'source_lock.json')==proof['source_lock_sha256']
for name,want in json.loads((control/'source_lock.json').read_text())['files'].items():assert sha(control/name)==want
assert sha(control/'submit_handoff.sh')==proof['batch_sha256']
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(control/'software/src')+':'+str(control/'software/scripts'))
t=subprocess.run(['/scratch/xl4358/phase1_env/bin/python','-m','pytest','-q',str(control/'software/tests')],env=env,capture_output=True,text=True)
(control/'software_tests.txt').write_text(t.stdout+t.stderr);assert t.returncode==0,t.stdout+t.stderr
parent=subprocess.check_output(['/opt/slurm/bin/sacct','-X','-j','7251','-n','-P','--format=JobIDRaw,State,Account,Partition'],text=True).strip().split('|')
assert parent[0]=='7251' and parent[1] in ('RUNNING','PENDING','CONFIGURING','COMPLETED') and parent[2:4]==['csci_ga_2271-2026fa','c12m85-a100-1']
assert not (control/'handoff_submission_attempt.json').exists()
attempt={'status':'POSTTRAIN_HANDOFF_SUBMIT_ATTEMPT','parent':7251,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source':str(control),'source_lock_sha256':proof['source_lock_sha256'],'batch_sha256':proof['batch_sha256'],'hpc_tests':t.stdout,'single_future_diagnostic_submission_only_after_actual_all6_proof':True}
(control/'handoff_submission_attempt.json').write_text(json.dumps(attempt,indent=2)+'\n')
job=subprocess.check_output(['/opt/slurm/bin/sbatch','--parsable','--no-requeue','--dependency=afterok:7251',str(control/'submit_handoff.sh')],text=True).strip().split(';')[0]
assert job.isdigit(),'unknown submission, no retry'
attempt.update(status='ACTUAL_POSTTRAIN_HANDOFF_SUBMITTED',handoff_job_id=int(job),dependency='afterok:7251')
(control/'handoff_submission.json').write_text(json.dumps(attempt,indent=2)+'\n');print(json.dumps(attempt))
