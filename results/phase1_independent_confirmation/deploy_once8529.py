import pathlib,json,hashlib,zipfile,os,socket,subprocess,sys
root=pathlib.Path('/scratch/xl4358/phase1_independent_confirmation')
archive=pathlib.Path('/scratch/xl4358/phase1_independent_confirmation_deployment.zip')
expected={'status': 'SOURCE_FROZEN_READY_FOR_NATIVE_DEV_PREFLIGHT_NOT_FRESH_GENERATION', 'source_files': 77, 'source_lock_sha256': '0f0fd1811b288a6aad8584a2389d69269151b1e69759921bcb4ac7cf84e62f16', 'protocol_sha256': '11ba209cacd30b9a1d70a0770f20397dfb4161e9bf37d92229071485d022b26e', 'archive_sha256': '2ec85fe564b786af0eb2e322903231bd8588268df6428a2e8d2596435b74410e', 'archive_bytes': 276440, 'HPC_root': '/scratch/xl4358/phase1_independent_confirmation', 'software_tests': 12, 'prior_GPU_seconds': 13131, 'remaining_GPU_seconds': 15669, 'full_estimated_GPU_including_native_equivalence': 12965.453815506098, 'fresh9137_generated': False, 'submitted': False, 'phase1_go': False}
assert hashlib.sha256(archive.read_bytes()).hexdigest()==expected['archive_sha256']
assert not root.exists()
raw=subprocess.run(['/opt/slurm/bin/scontrol','show','job','-o','8529'],capture_output=True,text=True,check=True).stdout
assert 'JobState=RUNNING ' in raw and 'Account=csci_ga_2271-2026fa ' in raw and 'NodeList='+socket.gethostname().split('.')[0]+' ' in raw
with zipfile.ZipFile(archive) as z:
 assert all(not pathlib.PurePosixPath(n).is_absolute() and '..' not in pathlib.PurePosixPath(n).parts for n in z.namelist())
 root.mkdir();z.extractall(root)
assert hashlib.sha256((root/'source_lock.json').read_bytes()).hexdigest()==expected['source_lock_sha256']
lock=json.loads((root/'source_lock.json').read_text())
for n,sha in lock['files'].items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==sha,n
assert hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest()==expected['protocol_sha256']
proof=dict(status='NATIVE_DEPLOYMENT_SOURCE_VERIFIED_NOT_SCIENTIFIC_PASS',archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),source_lock_sha256=expected['source_lock_sha256'],source_files=len(lock['files']),control_job=8529,native_control=raw,fresh9137_accessed=False,new_GPU_seconds=0)
(root/'deployment_proof.json').write_text(json.dumps(proof,indent=2)+'\n')
sys.path[:0]=[str(root),str(root/'original_src')]
from orchestrator import submit
submit('preflight_cpu')
