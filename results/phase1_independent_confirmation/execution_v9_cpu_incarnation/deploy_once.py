import pathlib,json,hashlib,zipfile,socket,subprocess,sys,os
root=pathlib.Path('/scratch/xl4358/phase1_independent_confirmation_execution_v9_cpu_incarnation')
archive=pathlib.Path('/scratch/xl4358/phase1_independent_confirmation_execution_v9_cpu_incarnation_deployment.zip')
expected={'status': 'FROZEN_CPU_STARTUP_REPAIR_NOT_SUBMITTED', 'source_files': 109, 'source_lock_sha256': '76fd0dd1070fc614bbbf246eee7bc4c3ff7d308388dd67e26b663ee8a9a58a30', 'protocol_sha256': '6cab3122c8bb9e98aebb6d59739f49d5de34d9bb654efcb507f5f6b2938582a2', 'archive_sha256': '114b83a6a3ae05bc559905630780db87557f725e07fbb20a95f0ae7933ad53b9', 'HPC_root': '/scratch/xl4358/phase1_independent_confirmation_execution_v9_cpu_incarnation', 'prior_GPU_seconds': 40499, 'remaining_authorized_GPU_seconds': 28800, 'cumulative_cap_seconds': 69299, 'local_software_tests': 52, 'reused_frames': 812, 'new_models_trained': 0, 'fresh_test_unopened': True, 'GPU_hard_limits_sum': 23400}
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
