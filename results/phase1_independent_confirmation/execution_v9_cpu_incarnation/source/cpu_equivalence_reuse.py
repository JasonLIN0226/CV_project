"""Reuse the actual completed V5 CPU proof with explicit origin, never relabel it.

Only execution/budget wiring changes. Scientific source bytes and the original
encoder objects are checked again on the new course CPU allocation.
"""
from pathlib import Path
import hashlib,json,shutil

EXECUTION_CHANGES={'config.json','stage.sh','watchdog.sh','cleanup.sh','startup_client.py','guardian_preflight.py',
 'orchestrator.py','budget_guard.py','protocol.json','restart_authorization.json'}

def check_scientific_identity(old,new):
    old=Path(old);new=Path(new)
    lock=json.loads((old/'source_lock.json').read_text())
    for name,sha in lock['files'].items():
        if hashlib.sha256((old/name).read_bytes()).hexdigest()!=sha:
            raise ValueError('Original completed CPU source changed: '+name)
        if name not in EXECUTION_CHANGES and hashlib.sha256((new/name).read_bytes()).hexdigest()!=sha:
            raise ValueError('Scientific bytes differ from native CPU proof: '+name)
    return True


def inherit(root,protocol):
    from phase1.utils import read_json,write_json,file_hash
    from budget_guard import native_allocations
    root=Path(root);origin=protocol['execution_amendment']['v6_startup_recovery']['CPU_proof_origin']
    old=Path(origin['root'])
    if file_hash(old/'source_lock.json')!=origin['source_lock_sha256']:
        raise ValueError('Original V5 native source lock changed')
    check_scientific_identity(old,root)
    native=native_allocations([origin['job']])[0]
    if native['state']!='COMPLETED' or native['exitcode']!='0:0' or 'gres/gpu' in native['tres']:
        raise ValueError('Cannot inherit an incomplete or non-CPU proof')
    p=old/'preflight_cpu.json'
    if file_hash(p)!=origin['proof_sha256']:raise ValueError('Native CPU proof changed')
    proof=read_json(p);completed=read_json(old/'stage_completed/preflight_cpu.json')
    assert completed['job']==origin['job'] and completed['source_lock_sha256']==origin['source_lock_sha256']
    assert proof['all_exact'] and proof['samples']==9728 and proof['validation_batches']==24 and not proof['fresh9137_accessed']
    from dev_equivalence import encoder_cache_proof
    if encoder_cache_proof()!=proof['encoder_cache']:raise ValueError('Approved encoder cache changed')
    # These are existing6301 metadata, normalization and basis views. Their configs
    # deliberately retain the original audited V5 output paths and provenance.
    shutil.copytree(old/'dev_equivalence',root/'dev_equivalence')
    proof['execution_proof_reuse']={**origin,'actual_original_native_CPU':native,
      'new_scientific_source_byte_identity_verified':True,'encoder_content_rehashed':True,
      'not_a_new_execution_of_9728_sample_fixture':True,'new_source_lock_sha256':file_hash(root/'source_lock.json')}
    write_json(root/'preflight_cpu.json',proof)
