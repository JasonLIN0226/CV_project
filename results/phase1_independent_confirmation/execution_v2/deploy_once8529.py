import pathlib,json,hashlib,zipfile,os,socket,subprocess,sys
root=pathlib.Path('/scratch/xl4358/phase1_independent_confirmation_execution_v2')
archive=pathlib.Path('/scratch/xl4358/phase1_independent_confirmation_execution_v2_deployment.zip')
expected={'status': 'EXPLICIT_SLURM_ENVIRONMENT_AMENDMENT_FROZEN', 'source_files': 77, 'source_lock_sha256': 'a41915dfb4fdb32a629e92445af5243c71b84e620433f0bc6256db2da33a4e8a', 'protocol_sha256': '324e3139a0867ae339a5ee5544d36637e6c6778bc8467fe02008d75a4e405736', 'archive_sha256': '608f356c9d6d1dbe2418f3c71435edc179ebbb40d5a96b3b4701dc0115fb4d2c', 'HPC_root': '/scratch/xl4358/phase1_independent_confirmation_execution_v2', 'changed_files': ['config.json', 'orchestrator.py', 'protocol.json', 'stage.sh'], 'new_GPU_seconds': 0, 'fresh9137_generated': False, 'submitted': False, 'software_scientific_fixtures_unchanged': True}
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
from budget_guard import native_allocations
old=native_allocations([8532])[0]
assert old['state']=='CANCELLED' and old['exitcode']=='0:0' and old['elapsed']==0
proof['previous_execution_native_terminal']=old
(root/'deployment_proof.json').write_text(json.dumps(proof,indent=2)+'\n')
submit('preflight_cpu')
