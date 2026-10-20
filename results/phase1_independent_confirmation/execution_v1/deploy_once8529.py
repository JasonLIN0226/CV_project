import pathlib,json,hashlib,zipfile,os,socket,subprocess,sys
root=pathlib.Path('/scratch/xl4358/phase1_independent_confirmation_execution_v1')
archive=pathlib.Path('/scratch/xl4358/phase1_independent_confirmation_execution_v1_deployment.zip')
expected={'status': 'EXECUTION_PATH_ONLY_AMENDMENT_FROZEN', 'UTC': '2026-10-19T04:49:45.944015+00:00', 'source_files': 77, 'source_lock_sha256': '3c735c89a31219ad17be3a3c9d30d208e4a85a5a07e41f69c3c17c0054e7ed7e', 'protocol_sha256': 'c36dee124601896c861f0f72f12d7c60a8d730a2e5de176222c11fdf2b08d98f', 'archive_sha256': '4d08abbf6f7994b86731c1780b9da85cb074f28f7613c475597c13965d5eb9e8', 'HPC_root': '/scratch/xl4358/phase1_independent_confirmation_execution_v1', 'changed_files': ['config.json', 'protocol.json', 'stage.sh'], 'scientific_code_all_bytes_unchanged': True, 'software_tests_inherited': 12, 'shell_syntax_PASS': True, 'source_namespace_fixture_PASS': True, 'submitted': False, 'fresh9137_generated': False, 'new_GPU_seconds': 0}
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
old=native_allocations([8530])[0]
assert old['state']=='FAILED' and old['exitcode']=='2:0' and old['elapsed']==4
proof['previous_execution_native_terminal']=old
(root/'deployment_proof.json').write_text(json.dumps(proof,indent=2)+'\n')
submit('preflight_cpu')
