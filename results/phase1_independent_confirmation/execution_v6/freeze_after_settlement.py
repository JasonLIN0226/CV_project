from pathlib import Path
import json,hashlib,zipfile
B=Path(__file__).resolve().parents[1];O=B/'execution_v6';S=O/'source';V=B/'execution_v5/source'
D=O/'native_settlement';settled=json.loads((D/'v5_stop_native_settlement.json').read_text());assert json.loads((D/'transfer_integrity.json').read_text())['all_match']
assert settled['GPU_batch_never_entered'] and settled['CPU_equivalence_8794_actual_complete']
spent=settled['cumulative_GPU_seconds'];remain=54051-spent
assert remain>=13800+120,'Full declared hard-allocation sequence no longer fits; do not submit'
(S/'v5_stop_native_settlement.json').write_text(json.dumps(settled,indent=2)+'\n')
p=S/'budget_guard.py';p.write_text(p.read_text().replace('PRIOR = 39712','PRIOR = '+str(spent)))
p=S/'orchestrator.py';t=p.read_text().replace('8536:26520,8792:61}','8536:26520,8792:61,8799:'+str(settled['new_GPU_seconds'])+'}');assert '8799:' in t;p.write_text(t)
p=S/'restart_authorization.json';x=json.loads(p.read_text());x.update(prior_actual_GPU_seconds=spent,additional_GPU_seconds=remain,original_additional_approved_GPU_seconds=14400,original_prior_actual_GPU_seconds=39651,already_spent_under_this_authorization_seconds=spent-39651,settlement='v5_stop_native_settlement.json');p.write_text(json.dumps(x,indent=2)+'\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=S/'protocol.json';x=json.loads(p.read_text());x['prior_settled_seconds']=spent;x['execution_amendment']['v6_startup_recovery']={
 'V5_source_preserved':True,'V5_source_lock_sha256':sha(V/'source_lock.json'),
 'native_GPU8799_charged_seconds':settled['new_GPU_seconds'],'GPU_batch_never_started':True,
 'startup_limit_seconds':300,'preflight_GPU_allocation_limit_seconds':600,'fixed_startup_reserve_added_seconds':300,'forecast_GPU_preflight_seconds':600,
 'maximum_sum_of_four_GPU_allocation_limits_seconds':13800,'remaining_after_native_settlement_seconds':remain,
 'scientific_scope_unchanged':True,'no_fresh_data_accessed_before_fix':True,
 'CPU_proof_origin':dict(job=8794,root='/scratch/xl4358/phase1_independent_confirmation_execution_v5',source_lock_sha256=sha(V/'source_lock.json'),proof_sha256=sha(D/'preflight_cpu.json')),
 'if_startup_300s_fails_again':'Stop at execution boundary; no further automatic restart or increased allowance'}
p.write_text(json.dumps(x,indent=2)+'\n')
allowed={'config.json','stage.sh','watchdog.sh','cleanup.sh','startup_client.py','guardian_preflight.py','orchestrator.py','budget_guard.py','protocol.json','restart_authorization.json'}
prior=json.loads((V/'source_lock.json').read_text())['files'];changed=[n for n,h in prior.items() if sha(S/n)!=h];assert set(changed)<=allowed,changed
files={str(p.relative_to(S)):sha(p) for p in sorted(S.rglob('*')) if p.is_file() and p.name!='source_lock.json' and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts}
(S/'source_lock.json').write_text(json.dumps(dict(status='FROZEN_BOUNDED_STARTUP_RECOVERY',files=files,scientific_files_byte_identical_to_v5=True,execution_files_changed=changed),indent=2)+'\n')
archive=O/'deployment.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for n in [*files,'source_lock.json']:z.write(S/n,n)
ready=dict(status='FROZEN_V6_READY_NOT_SUBMITTED',source_files=len(files),source_lock_sha256=sha(S/'source_lock.json'),protocol_sha256=sha(S/'protocol.json'),archive_sha256=sha(archive),HPC_root='/scratch/xl4358/phase1_independent_confirmation_execution_v6',prior_GPU_seconds=spent,remaining_authorized_GPU_seconds=remain,cumulative_cap_seconds=54051,latest_execution_fixtures=22,fresh9137_accessed=False)
(O/'ready.json').write_text(json.dumps(ready,indent=2)+'\n')
t=(B/'execution_v5/deploy_once.py').read_text();a=t.index('expected=');b=t.index('\ncontrol=',a);t=t[:a]+'expected='+repr(ready)+t[b:];t=t.replace('phase1_independent_confirmation_execution_v5','phase1_independent_confirmation_execution_v6');(O/'deploy_once.py').write_text(t)
print(json.dumps(ready))
