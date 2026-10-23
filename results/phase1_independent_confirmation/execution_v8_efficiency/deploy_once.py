import pathlib,json,hashlib,zipfile,socket,subprocess,sys,os
root=pathlib.Path('/scratch/xl4358/phase1_independent_confirmation_execution_v8_efficiency')
archive=pathlib.Path('/scratch/xl4358/phase1_independent_confirmation_execution_v8_efficiency_deployment.zip')
expected={'status': 'FROZEN_V8_EFFICIENCY_NOT_SUBMITTED', 'source_files': 106, 'source_lock_sha256': '5150b1e4645a4dcebcb9d1c463b2842e73b7eeb1b670491278cb3bce0b0e8ec2', 'protocol_sha256': '4cf98e8466b2645e2ddcd28f8b806cfcae59b4816c2d7ee68be66fc9dcb2b622', 'archive_sha256': 'af25a46f5a25728da799c836b44bdaf6716e486521042e23638ac9251b8baa87', 'HPC_root': '/scratch/xl4358/phase1_independent_confirmation_execution_v8_efficiency', 'prior_GPU_seconds': 40499, 'remaining_authorized_GPU_seconds': 28800, 'cumulative_cap_seconds': 69299, 'local_software_tests': 42, 'reused_frames': 812, 'new_models_trained': 0, 'fresh_test_unopened': True, 'GPU_hard_limits_sum': 23400}
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
