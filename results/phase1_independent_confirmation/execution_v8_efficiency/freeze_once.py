from pathlib import Path
import json,hashlib,zipfile,ast,copy
B=Path(__file__).resolve().parent;S=B/'source';V=B.parent/'execution_v7_metadata';E=V/'native_budget_stop';read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (B/'ready.json').exists(),'Frozen output already exists; no rebuild of active source'
assert read(E/'transfer_integrity.json')['all_match'] and read(V/'local_budget_stop_audit.json')['passed']
proof=read(E/'native_budget_stop_evidence.json');assert proof['cumulative_GPU_seconds']==40499 and proof['models_trained']==0 and proof['fresh_test_unopened'] and proof['committed_encoded_frames']==812
(S/'v7_budget_stop_evidence.json').write_bytes((E/'native_budget_stop_evidence.json').read_bytes())
proofs={n:sha(E/n) for n in ['native_budget_stop_evidence.json','fresh/sequence_budget.json','fresh/budget_stop.json','fresh/features/index.json','fresh/encoder_loaded_train.json','watchdog_terminal.json','guardian_peer_terminal.json','stage_failures/encode_train.json']}
proofs.update({n:sha(V/'source'/n) for n in ['config.json','protocol.json']})
proofs.update({n:sha(V/'native_resume_complete'/n) for n in ['permit.json','stage_completed/resume_cpu.json']})
for n in ['preflight_cpu.json','preflight_gpu.json']:proofs[n]=sha(V/'native_v6_evidence'/n)
p=S/'protocol.json';x=read(p);x['metadata_resume']=dict(origin_root='/scratch/xl4358/phase1_independent_confirmation_execution_v7_metadata',origin_source_lock_sha256=sha(V/'source/source_lock.json'),origin_proofs=proofs,committed_frames=812,prior_GPU_seconds=40499,remaining_GPU_seconds=28800,cumulative_GPU_cap_seconds=69299,explicit_new_remaining8h_authorization='remaining8h_authorization.json',old_V7_preserved=True,all_model_seed_sample_epoch_loss_threshold_math_unchanged=True)
x['predeployment_V8_efficiency_review']=dict(local_software_tests=42,native_encoder_equivalence_and_timing_before_cache_use_required=True,GPU_training_step_equivalence_proof_origin=8840,FP32batch2unchanged=True,complete_GPUhardlimits23400fit28800=True,checkpoint_and_boundary_IO_included_in_rolling_training_rate=True,final_evidence_manifest_legacy_release_file_recreated_from_nativeCPU=True)
p.write_text(json.dumps(x,indent=2)+'\n')
allowed={'config.json','stage.sh','watchdog.sh','cleanup.sh','orchestrator.py','budget_guard.py','protocol.json','restart_authorization.json','authorization.json','guarded_encode.py','fresh_training.py','terminal_evidence.py'}
prior=read(V/'source/source_lock.json')['files'];changed=[n for n,h in prior.items() if sha(S/n)!=h];assert set(changed)<=allowed,changed
# Audit the fresh training numerical path: only two wall-clock accounting statements change.
def parsed(path):return ast.parse(path.read_text())
a=parsed(V/'source/fresh_training.py');b=parsed(S/'fresh_training.py')
class RestoreTiming(ast.NodeTransformer):
 def visit_Assign(self,node):
  if any(isinstance(t,ast.Name) and t.id=='epoch_wall_start' for t in node.targets):return None
  return self.generic_visit(node)
 def visit_Call(self,node):
  self.generic_visit(node)
  if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='sequence_guard' and node.func.attr=='record_epoch':node.args[-1]=ast.Name(id='elapsed',ctx=ast.Load())
  return node
assert ast.dump(a,include_attributes=False)==ast.dump(RestoreTiming().visit(b),include_attributes=False)
for q in S.rglob('*.py'):ast.parse(q.read_text())
files={str(q.relative_to(S)):sha(q) for q in sorted(S.rglob('*')) if q.is_file() and q.name!='source_lock.json' and '__pycache__' not in q.parts and '.pytest_cache' not in q.parts}
lock=dict(status='FROZEN_EXACT_ENCODING_CACHE_AND_NEW_REMAINING8H',files=files,origin_source_lock_sha256=sha(V/'source/source_lock.json'),changed_original_files=changed,training_AST_same_except_complete_wall_clock_measurement=True,all_scientific_original_modules_byte_identical=True)
(S/'source_lock.json').write_text(json.dumps(lock,indent=2)+'\n')
with zipfile.ZipFile(B/'deployment.zip','w',zipfile.ZIP_DEFLATED) as z:
 for name in [*files,'source_lock.json']:z.write(S/name,name)
ready=dict(status='FROZEN_V8_EFFICIENCY_NOT_SUBMITTED',source_files=len(files),source_lock_sha256=sha(S/'source_lock.json'),protocol_sha256=sha(S/'protocol.json'),archive_sha256=sha(B/'deployment.zip'),HPC_root='/scratch/xl4358/phase1_independent_confirmation_execution_v8_efficiency',prior_GPU_seconds=40499,remaining_authorized_GPU_seconds=28800,cumulative_cap_seconds=69299,local_software_tests=42,reused_frames=812,new_models_trained=0,fresh_test_unopened=True,GPU_hard_limits_sum=23400)
(B/'ready.json').write_text(json.dumps(ready,indent=2)+'\n')
t=(V/'deploy_once.py').read_text();i=t.index('expected=');j=t.index('\ncontrol=',i);t=t[:i]+'expected='+repr(ready)+t[j:];t=t.replace('phase1_independent_confirmation_execution_v7_metadata','phase1_independent_confirmation_execution_v8_efficiency');(B/'deploy_once.py').write_text(t)
print(json.dumps(ready))
