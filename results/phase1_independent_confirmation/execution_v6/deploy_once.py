import pathlib,json,hashlib,zipfile,socket,subprocess,sys,os
root=pathlib.Path('/scratch/xl4358/phase1_independent_confirmation_execution_v6')
archive=pathlib.Path('/scratch/xl4358/phase1_independent_confirmation_execution_v6_deployment.zip')
expected={'status': 'FROZEN_V6_READY_NOT_SUBMITTED', 'source_files': 95, 'source_lock_sha256': 'b44524025b86bc96f9bce931008972874f918cbcd6b5cb8aba4e70b2c1375585', 'protocol_sha256': '4a81ab3347d58e5c08f9601af96f6ce6eb016eb26dd786fba975708df6d2889d', 'archive_sha256': '0c25c367b4da9481eb8b55bce6a1b54b657f67a69bfa319f7607aa8f7a55c62b', 'HPC_root': '/scratch/xl4358/phase1_independent_confirmation_execution_v6', 'prior_GPU_seconds': 39897, 'remaining_authorized_GPU_seconds': 14154, 'cumulative_cap_seconds': 54051, 'latest_execution_fixtures': 22, 'fresh9137_accessed': False}
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
