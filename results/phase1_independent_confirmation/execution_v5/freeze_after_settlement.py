from pathlib import Path
import json,hashlib,zipfile,shutil
B=Path(__file__).resolve().parents[1];O=B/'execution_v5';S=O/'source';V=B/'execution_v4/source'
settled=json.loads((O/'native_settlement/v4_stop_native_settlement.json').read_text())
integrity=json.loads((O/'native_settlement/transfer_integrity.json').read_text());assert integrity['all_match']
assert settled['new_GPU_seconds']==61 and settled['cumulative_GPU_seconds']==39712
(S/'v4_stop_native_settlement.json').write_text(json.dumps(settled,indent=2)+'\n')
p=S/'budget_guard.py';p.write_text(p.read_text().replace('PRIOR = 39651','PRIOR = 39712'))
p=S/'orchestrator.py';t=p.read_text().replace('8518:27,8536:26520}','8518:27,8536:26520,8792:61}');assert '8792:61' in t;p.write_text(t)
p=S/'restart_authorization.json';x=json.loads(p.read_text());x.update(prior_actual_GPU_seconds=39712,additional_GPU_seconds=14339,original_additional_approved_GPU_seconds=14400,original_prior_actual_GPU_seconds=39651,already_spent_under_this_authorization_seconds=61,settlement='v4_stop_native_settlement.json');p.write_text(json.dumps(x,indent=2)+'\n')
p=S/'protocol.json';x=json.loads(p.read_text());x['prior_settled_seconds']=39712;x['execution_amendment']['v5_clock_race_fix']={
 'V4_source_preserved':True,'V4_source_lock_sha256':hashlib.sha256((V/'source_lock.json').read_bytes()).hexdigest(),
 'native_GPU8792_charged_seconds':61,'heartbeat_sample_clock_after_read':True,'maximum_age_seconds':25,'bounded_node_clock_skew_seconds':1,
 'actual_concurrent_heartbeat_cycles_before_GPU':8,'CPU_equivalence_rerun':True,'scientific_scope_unchanged':True,'no_fresh_data_accessed_before_fix':True}
p.write_text(json.dumps(x,indent=2)+'\n')
allowed={'config.json','stage.sh','watchdog.sh','cleanup.sh','watchdog.py','guardian_peer.py','startup_client.py','guardian_preflight.py','budget_guard.py','orchestrator.py','protocol.json','restart_authorization.json'}
prior=json.loads((V/'source_lock.json').read_text())['files']
changed=[n for n,h in prior.items() if hashlib.sha256((S/n).read_bytes()).hexdigest()!=h]
assert set(changed)<=allowed,changed
assert all((S/n).read_bytes()==(V/n).read_bytes() for n in prior if n not in allowed)
files={str(p.relative_to(S)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(S.rglob('*')) if p.is_file() and p.name!='source_lock.json' and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts}
lock=dict(status='FROZEN_EXECUTION_CLOCK_RACE_FIX',files=files,scientific_files_byte_identical_to_v4=True,execution_files_changed=changed)
(S/'source_lock.json').write_text(json.dumps(lock,indent=2)+'\n')
archive=O/'deployment.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for n in [*files,'source_lock.json']:z.write(S/n,n)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
ready=dict(status='FROZEN_V5_READY_NOT_SUBMITTED',source_files=len(files),source_lock_sha256=sha(S/'source_lock.json'),protocol_sha256=sha(S/'protocol.json'),archive_sha256=sha(archive),HPC_root='/scratch/xl4358/phase1_independent_confirmation_execution_v5',prior_GPU_seconds=39712,remaining_authorized_GPU_seconds=14339,cumulative_cap_seconds=54051,full_software_tests=30,fresh9137_accessed=False)
(O/'ready.json').write_text(json.dumps(ready,indent=2)+'\n')
t=(B/'execution_v4/deploy_once.py').read_text();a=t.index('expected=');b=t.index('\ncontrol=',a);t=t[:a]+'expected='+repr(ready)+t[b:];t=t.replace('phase1_independent_confirmation_execution_v4','phase1_independent_confirmation_execution_v5')
(O/'deploy_once.py').write_text(t)
print(json.dumps(ready))
