"""Execution-only continuation of frozen stages after all model training."""
import os, sys, socket, json, pathlib, time, subprocess
R=pathlib.Path('/scratch/xl4358/phase1_independent_confirmation_execution_v10_checkpoint_resume')
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HOME='/scratch/xl4358/phase1_hf_cache',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TZ='UTC')
sys.path[:0]=[str(R),str(R/'original_src'),str(R/'execution_repair')]
import batch_runtime
w=batch_runtime.install()
import additional_charge, startup_client, budget_guard
from watchdog import native_job, ACCOUNT
import locks

def self_live(*args,**kwargs):
    job=int(os.environ['SLURM_JOB_ID']); n=native_job(job)
    if n['JobState']!='RUNNING' or n['Account']!=ACCOUNT or n['NodeList']!=socket.gethostname().split('.')[0]:
        raise ValueError('Current actual course allocation is not valid')
startup_client.require_live=self_live
budget_guard.require_live=self_live
w.startup_require_live=self_live
if __name__=='__main__' and len(sys.argv)>1 and sys.argv[1]=='lock_cpu':
    original_create=locks.create_all_lock
    def reuse_verified_lock(c,permit):
        path=pathlib.Path(c['output'])/'all_models_lock.json'
        if not path.exists():
            return original_create(c,permit)
        lock=json.loads(path.read_text())
        locks.verify_all_lock(c,permit,lock)
        root=pathlib.Path(c['output'])
        if (root/'confirmation_test_generation_attempt.json').exists():
            raise ValueError('Existing test-generation attempt cannot be reopened')
        return lock
    locks.create_all_lock=reuse_verified_lock
STAGES=['train_gpu','lock_cpu','encode_test','evaluate_cpu','audit_cpu','evidence_cpu']
w.CHAIN=STAGES

def already_submitted(stage,parent=None):
    p=R/'receipts'/f'{stage}_submission.json'; r=json.loads(p.read_text())
    assert r['stage']==stage and r['parent']==parent
    assert r['source_lock_sha256']==w.file_hash(R/'source_lock.json')
    assert r['final_continuation'] is True
    return r
w.submit=already_submitted
if __name__=='__main__':
    stage=sys.argv[1]; assert stage in STAGES[1:]
    marker=R/'execution_repair/final_continuation/submitted.json'
    for _ in range(60):
        if marker.exists(): break
        time.sleep(1)
    else: raise RuntimeError('Full stage submission marker missing')
    manifest=json.loads(marker.read_text())
    assert next(x['job'] for x in manifest['receipts'] if x['stage']==stage)==int(os.environ['SLURM_JOB_ID'])
    w.main(stage)
