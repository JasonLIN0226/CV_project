import pathlib,json,hashlib,zipfile,socket,subprocess,sys,os
root=pathlib.Path('/scratch/xl4358/phase1_independent_confirmation_execution_v3')
archive=pathlib.Path('/scratch/xl4358/phase1_independent_confirmation_execution_v3_deployment.zip')
expected={'status': 'FROZEN_SOFTWARE_PREFLIGHT_PASS_NOT_SUBMITTED', 'source_files': 87, 'source_lock_sha256': '113816b81c79a68754de9af844a09c9deb482f3851179c65f337c5a401285a49', 'protocol_sha256': 'a5ef4261cc4e30839b3e4423fd97fe6df003c53f37e38991fe780d61e7562f6d', 'archive_sha256': '08868c57d52ed1b377bb1c90a19da427e6c739bda2645e3b7a2e9082e0bf0a89', 'HPC_root': '/scratch/xl4358/phase1_independent_confirmation_execution_v3', 'prior_GPU_seconds': 39651, 'additional_authorized_GPU_seconds': 14400, 'cumulative_cap_seconds': 54051, 'software_tests_passed': 25, 'fresh9137_accessed': False, 'GPU_submitted': False, 'latest_guard_boundary_fixtures': 13}
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
