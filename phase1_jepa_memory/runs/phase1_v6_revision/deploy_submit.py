from pathlib import Path
import os,json,hashlib,zipfile,socket,subprocess,datetime
CONTROL=Path('/scratch/xl4358/phase1_jepa_memory_v6_real_wd020')
BUNDLE=Path('/scratch/xl4358/phase1_v6_revision_gpu_bundle.zip')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
actual=subprocess.check_output(['/opt/slurm/bin/sacct','-X','-j','7243','-n','-P','--format=JobIDRaw,State,Account,Partition,NodeList'],text=True).strip().split('|')
assert actual[:5]==['7243','RUNNING','csci_ga_2271-2026fa','interactive',socket.gethostname()]
assert not CONTROL.exists()
CONTROL.mkdir(mode=0o700);(CONTROL/'logs').mkdir()
with zipfile.ZipFile(BUNDLE) as z:
 for name in z.namelist():
  p=Path(name);assert not p.is_absolute() and '..' not in p.parts
  assert not (z.getinfo(name).external_attr>>16)&0o170000==0o120000
 z.extractall(CONTROL)
lock=CONTROL/'source_lock.json'
expected=json.loads(Path('/scratch/xl4358/phase1_v6_revision_gpu_preparation.json').read_text())
assert sha(lock)==expected['source_lock_sha256']
for name,want in json.loads(lock.read_text())['files'].items():
 p=CONTROL/name;assert not p.is_symlink() and p.is_file() and sha(p)==want
assert sha(CONTROL/'submit_training.sh')==expected['batch_sha256']
for control,want in [(Path('/scratch/xl4358/phase1_jepa_memory_v5_real_frame_training'),'1e5ba4b9242469df3bc7f6fe6f03eec53a62cdfd9c47f5fdda73845ebfb5504f'),(Path('/scratch/xl4358/phase1_jepa_memory_v5_encoding_frame'),'cd667fc4f029660c198353570686a8ed8f069a0e2dab1434e912776f5521985e')]:
 assert sha(control/'source_lock.json')==want
 for name,w in json.loads((control/'source_lock.json').read_text())['files'].items():assert sha(control/name)==w
parent=Path('/scratch/xl4358/phase1_v5_real_frame_training/job_7209')
for name,w in [('train_only_fit.npz','d90f99e4a91d290b7d27311de2820775981afe08dc053dc642e8a40636ed1fb3'),('fit_proof.json','18c44903742143b482aef5a7c035aa2540fb3f60ba8963c50971be8abc5fd7aa')]:assert sha(parent/name)==w
record=subprocess.check_output(['/opt/slurm/bin/sacct','-X','-j','7187,7188,7209,7226','-n','-P','--format=JobIDRaw,State,ExitCode,Account'],text=True)
assert all(line.split('|')[1:4]==['COMPLETED','0:0','csci_ga_2271-2026fa'] for line in record.splitlines())
env=dict(os.environ,PYTHONPATH=str(CONTROL/'src')+':'+str(CONTROL/'scripts'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
result=subprocess.run(['/scratch/xl4358/phase1_env/bin/python','-m','pytest','-q',str(CONTROL/'tests')],env=env,text=True,capture_output=True)
(CONTROL/'software_tests.txt').write_text(result.stdout+result.stderr);assert result.returncode==0,result.stdout+result.stderr
queue=subprocess.check_output(['/opt/slurm/bin/squeue','-h','-u','xl4358','-o','%i|%T|%a|%j'],text=True)
assert all(line.split('|')[0]=='7243' or 'csci_ga_2271-2026fa' not in line for line in queue.splitlines()),queue
attempt={'status':'GPU_SUBMISSION_ATTEMPT_STARTED','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'course':'csci_ga_2271-2026fa','source':str(CONTROL),'source_lock_sha256':sha(lock),'protocol_sha256':sha(CONTROL/'protocol.json'),'batch_sha256':sha(CONTROL/'submit_training.sh'),'prior_all_jobs_native_completed0_0':True,'frozen_sources_and_original_fit_native_sha_verified':True,'local_tests':34,'hpc_tests_stdout':result.stdout,'single_change':'weight_decay0.05->0.20','cpu_fallback':False,'labels_joined':False}
(CONTROL/'submission_attempt.json').write_text(json.dumps(attempt,indent=2)+'\n')
job=subprocess.check_output(['/opt/slurm/bin/sbatch','--parsable','--no-requeue',str(CONTROL/'submit_training.sh')],text=True).strip().split(';')[0]
assert job.isdigit(),'Unknown submit outcome, inspect attempt, never retry'
attempt.update(status='ACTUAL_SINGLE_FACTOR_GPU_JOB_SUBMITTED',job_id=int(job),output='/scratch/xl4358/phase1_v6_real_wd020/job_'+job)
(CONTROL/'submission.json').write_text(json.dumps(attempt,indent=2)+'\n');print(json.dumps(attempt))
