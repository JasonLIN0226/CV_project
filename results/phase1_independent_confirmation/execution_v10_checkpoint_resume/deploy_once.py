import pathlib,json,hashlib,zipfile,socket,subprocess,sys,os
root=pathlib.Path('/scratch/xl4358/phase1_independent_confirmation_execution_v10_checkpoint_resume')
archive=pathlib.Path('/scratch/xl4358/phase1_independent_confirmation_execution_v10_checkpoint_resume_deployment.zip')
expected={'status': 'FROZEN_CHECKPOINT_RECOVERY_NOT_SUBMITTED', 'source_files': 115, 'source_lock_sha256': '4133e4335f87f0b719376f45c71ac9adb52e05b3e2c45550bcb22863472727d9', 'protocol_sha256': '53607a5a865908bc56de553ebeaf53cf7259022e053e3a14b4290ca8c6313d28', 'archive_sha256': '1db6aeb1ff2bdc026964131f595d7211a995dac5c21a08eb96f81a9a9fed6cdd', 'HPC_root': '/scratch/xl4358/phase1_independent_confirmation_execution_v10_checkpoint_resume', 'settled_GPU_seconds': 42185, 'remaining_GPU_seconds': 27114, 'GPU_cap_seconds': 69299, 'reused_encoded_frames': 62418, 'retained_epochs': 80, 'retained_steps': 5120, 'no_completed_stage_replay': True, 'local_focused_tests': 40, 'full_suite_unchanged_science_tests_passed': True, 'fresh_test_unopened': True}
control=int(sys.argv[1])
raw=subprocess.run(['/opt/slurm/bin/scontrol','show','job','-o',str(control)],capture_output=True,text=True,check=True,timeout=15).stdout
f=dict(x.split('=',1) for x in raw.split() if '=' in x)
assert f['JobState']=='RUNNING' and f['Account']=='csci_ga_2271-2026fa' and f['NodeList']==socket.gethostname().split('.')[0]
assert 'gres/gpu=' not in f.get('AllocTRES','')
assert hashlib.sha256(archive.read_bytes()).hexdigest()==expected['archive_sha256']
assert not root.exists(),'Existing or unknown deployment must be inspected, never retried'
with zipfile.ZipFile(archive) as z:
 assert all(not pathlib.PurePosixPath(n).is_absolute() and '..' not in pathlib.PurePosixPath(n).parts for n in z.namelist())
 root.mkdir();z.extractall(root)
assert hashlib.sha256((root/'source_lock.json').read_bytes()).hexdigest()==expected['source_lock_sha256']
lock=json.loads((root/'source_lock.json').read_text())
for n,h in lock['files'].items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
sys.path[:0]=[str(root),str(root/'original_src')]
from orchestrator import source_check,submit
source_check()
(root/'deployment_proof.json').write_text(json.dumps({**expected,'status':'NATIVE_DEPLOYMENT_SOURCE_VERIFIED','control_job':control,'native_control':raw},indent=2)+'\n')
submit('resume_cpu')
