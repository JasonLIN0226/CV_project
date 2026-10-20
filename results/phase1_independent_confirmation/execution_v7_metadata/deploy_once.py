import pathlib,json,hashlib,zipfile,socket,subprocess,sys,os
root=pathlib.Path('/scratch/xl4358/phase1_independent_confirmation_execution_v7_metadata')
archive=pathlib.Path('/scratch/xl4358/phase1_independent_confirmation_execution_v7_metadata_deployment.zip')
expected={'status': 'FROZEN_V7_METADATA_READY_NOT_SUBMITTED', 'source_files': 100, 'source_lock_sha256': 'a88e1d531184a1c30859267e2ef543439d7d65461636f66ee335231c92ba5005', 'protocol_sha256': '8f005f6d8fb956fb2b1ad19bf830156988af7def124de2cfeeff219b67298017', 'archive_sha256': '51c7f33a09ae564058bfee2bb5305acb521500801eef508f7cf1a3cc9129bef2', 'HPC_root': '/scratch/xl4358/phase1_independent_confirmation_execution_v7_metadata', 'prior_GPU_seconds': 40283, 'remaining_authorized_GPU_seconds': 13768, 'cumulative_cap_seconds': 54051, 'latest_execution_fixtures': 35, 'fresh9137_train_validation_already_generated': True, 'fresh_test_unopened': True, 'models_trained': 0}
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
