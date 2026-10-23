from pathlib import Path
import json,hashlib,zipfile,datetime,ast
O=Path(__file__).resolve().parent;S=O/'source';V=O.parent/'execution_v6/source';D=O/'native_v6_evidence'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (O/'ready.json').exists(),'Already frozen; inspect, never rebuild active source'
assert read(D/'transfer_integrity.json')['all_match']
e=read(D/'metadata_failure_native_evidence.json');assert e['JSON_wire_metadata_equal'] and not e['raw_metadata_equal']
assert e['fresh_test_unopened'] and e['encoded_frames']==e['new_models']==e['new_maps']==0
spent=e['cumulative_GPU_seconds'];remain=54051-spent;assert remain>=13200+120
p=S/'budget_guard.py';t=p.read_text();assert 'PRIOR = 39897' in t;p.write_text(t.replace('PRIOR = 39897','PRIOR = '+str(spent)))
p=S/'restart_authorization.json';a=read(p);a.update(prior_actual_GPU_seconds=spent,additional_GPU_seconds=remain,already_spent_under_this_authorization_seconds=spent-a['original_prior_actual_GPU_seconds'],settlement='metadata_failure_native_evidence.json');p.write_text(json.dumps(a,indent=2)+'\n')
(S/'metadata_failure_native_evidence.json').write_bytes((D/'metadata_failure_native_evidence.json').read_bytes())
p=S/'protocol.json';protocol=read(p);protocol['prior_settled_seconds']=spent
proof_names=['metadata_failure_native_evidence.json','preflight_cpu.json','preflight_gpu.json','stage_completed/preflight_cpu.json','stage_completed/preflight_gpu.json','stage_completed/init_cpu.json','watchdog_terminal.json','config.json','permit.json','fresh/dataset/manifest.json']
protocol['metadata_resume']=dict(origin_root='/scratch/xl4358/phase1_independent_confirmation_execution_v6',origin_source_lock_sha256=sha(V/'source_lock.json'),settled_origin_GPU_charges=e['charged_including_configuration'],origin_proofs={n:sha(D/n) for n in proof_names},amendment='Exact JSON-roundtrip metadata comparison only. Actual loaded weights, full encoder metadata, dtype, revision and processor remain checked before first frame. Existing train/validation dataset copied byte-exact; no regeneration. Native GPU equivalence8840 explicitly inherited.',fresh_test_unopened=True,new_models=0,full_scope_unchanged=True,remaining_three_GPU_hard_limits=13200,remaining_authorized_GPU_seconds=remain)
protocol['resource_guard']['remaining_GPU_stage_limits_seconds']={'encode_train':2700,'train_gpu':6600,'encode_test':3900}
p.write_text(json.dumps(protocol,indent=2)+'\n')
# All original scientific modules remain byte-identical, apart from metadata comparison.
allowed={'config.json','stage.sh','watchdog.sh','cleanup.sh','orchestrator.py','budget_guard.py','protocol.json','restart_authorization.json','guarded_encode.py','test_submission.py'}
prior=read(V/'source_lock.json')['files'];changed=[n for n,h in prior.items() if sha(S/n)!=h];assert set(changed)<=allowed,changed
old=(V/'guarded_encode.py').read_text();new=(S/'guarded_encode.py').read_text();assert new.count('matches')==2
for p in S.rglob('*.py'):ast.parse(p.read_text())
files={str(p.relative_to(S)):sha(p) for p in sorted(S.rglob('*')) if p.is_file() and p.name!='source_lock.json' and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts}
lock=dict(status='FROZEN_METADATA_WIRE_FORMAT_ONLY_RESUME',files=files,origin_source_lock_sha256=sha(V/'source_lock.json'),changed_original_files=changed,science_unchanged=True,native_metadata_diagnosis_sha256=sha(D/'metadata_failure_native_evidence.json'))
(S/'source_lock.json').write_text(json.dumps(lock,indent=2)+'\n')
with zipfile.ZipFile(O/'deployment.zip','w',zipfile.ZIP_DEFLATED) as z:
 for n in [*files,'source_lock.json']:z.write(S/n,n)
ready=dict(status='FROZEN_V7_METADATA_READY_NOT_SUBMITTED',source_files=len(files),source_lock_sha256=sha(S/'source_lock.json'),protocol_sha256=sha(S/'protocol.json'),archive_sha256=sha(O/'deployment.zip'),HPC_root='/scratch/xl4358/phase1_independent_confirmation_execution_v7_metadata',prior_GPU_seconds=spent,remaining_authorized_GPU_seconds=remain,cumulative_cap_seconds=54051,latest_execution_fixtures=35,fresh9137_train_validation_already_generated=True,fresh_test_unopened=True,models_trained=0)
(O/'ready.json').write_text(json.dumps(ready,indent=2)+'\n')
t=(O.parent/'execution_v6/deploy_once.py').read_text();a=t.index('expected=');b=t.index('\ncontrol=',a);t=t[:a]+'expected='+repr(ready)+t[b:];t=t.replace('phase1_independent_confirmation_execution_v6','phase1_independent_confirmation_execution_v7_metadata').replace("submit('preflight_cpu')","submit('resume_cpu')");(O/'deploy_once.py').write_text(t)
print(json.dumps(ready))
