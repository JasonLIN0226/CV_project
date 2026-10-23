import pathlib,json,hashlib,zipfile,socket,subprocess,sys,os
root=pathlib.Path('/scratch/xl4358/phase1_independent_confirmation_execution_v5')
archive=pathlib.Path('/scratch/xl4358/phase1_independent_confirmation_execution_v5_deployment.zip')
expected={'status': 'FROZEN_V5_READY_NOT_SUBMITTED', 'source_files': 92, 'source_lock_sha256': 'a5fd31bbcb5105796e22b31155a13bc5bb5301161e42a0b72d5c5d109e509bca', 'protocol_sha256': '04a7ee8828f71e7e628f4b878cff4341015dd79a9862d9bc47babfff01f42e5b', 'archive_sha256': '220824a10cc096a3ba656ac1476921b5c55b3db31b1fe9cb00795b3f82a6cc74', 'HPC_root': '/scratch/xl4358/phase1_independent_confirmation_execution_v5', 'prior_GPU_seconds': 39712, 'remaining_authorized_GPU_seconds': 14339, 'cumulative_cap_seconds': 54051, 'full_software_tests': 30, 'fresh9137_accessed': False}
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
