import pathlib,json,hashlib,zipfile,socket,subprocess,sys,os
root=pathlib.Path('/scratch/xl4358/phase1_independent_confirmation_execution_v4')
archive=pathlib.Path('/scratch/xl4358/phase1_independent_confirmation_execution_v4_deployment.zip')
expected={'status': 'FROZEN_PAIRED_CPU_GUARDIANS_NOT_SUBMITTED', 'source_files': 89, 'source_lock_sha256': '5375a36b459771d7d97a64ca15eecfbaa86b87414e576391eb9651fe62fbc764', 'protocol_sha256': '7b0bc6aa587c0fff8a6fe81931693e3b8eadefc7a9892acda681ad0a743fc228', 'archive_sha256': '596709a18f44c3be6bb65a38172abb3ba5e08bbd38be8f6e090c2406f8c936a5', 'HPC_root': '/scratch/xl4358/phase1_independent_confirmation_execution_v4', 'prior_GPU_seconds': 39651, 'additional_authorized_GPU_seconds': 14400, 'cumulative_cap_seconds': 54051, 'full_software_tests': 25, 'latest_guard_tests': 17, 'native_cancel_probe_passed': True, 'scheduler_forces_Requeue1': True, 'GPU_submitted': False, 'fresh9137_accessed': False}
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
submit('preflight_cpu')
