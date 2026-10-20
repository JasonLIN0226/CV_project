"""Execution-only overlay: immutable science, settled charge floors, prequeued chain."""
import hashlib,json,os,pathlib,sys,time
R=pathlib.Path(__file__).resolve().parent.parent
sys.path[:0]=[str(R),str(R/'original_src'),str(R/'execution_repair')]
STAGES=['train_gpu','lock_cpu','encode_test','evaluate_cpu','audit_cpu','evidence_cpu']
FLOORS={8951:1265,9030:421,9086:2126,9115:21,9118:133,9120:649}

def verify_overlay():
    m=json.loads((R/'execution_repair/batch_manifest.json').read_text())
    for name,h in m['files'].items():
        assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h,name
    import orchestrator as w
    w.source_check()
    return m

def install():
    verify_overlay()
    import orchestrator as w,startup_client
    original=startup_client.guarded_allocation_rows
    def guarded(root,rows):
        return [dict(row,guarded_elapsed=max(row['guarded_elapsed'],FLOORS.get(row['job'],0)))
                for row in original(root,rows)]
    startup_client.guarded_allocation_rows=guarded
    w.CHAIN=STAGES.copy()
    return w

def existing_receipt(stage,parent=None):
    receipt=json.loads((R/'receipts'/f'{stage}_submission.json').read_text())
    assert receipt['stage']==stage and receipt['parent']==parent
    assert receipt['source_lock_sha256']==hashlib.sha256((R/'source_lock.json').read_bytes()).hexdigest()
    assert receipt['prequeued_batch_chain'] is True
    from watchdog import native_job
    n=native_job(receipt['job'])
    assert n['Account']=='csci_ga_2271-2026fa'
    assert n['JobState'] in {'PENDING','CONFIGURING','RUNNING','COMPLETING','COMPLETED'}
    return receipt

def run_stage(stage):
    assert stage in STAGES
    deadline=time.monotonic()+120
    while not (R/'execution_repair/batch_chain_submitted.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Full durable chain marker absent; no model work')
        time.sleep(1)
    full=json.loads((R/'execution_repair/batch_chain_submitted.json').read_text())
    assert [r['stage'] for r in full['receipts']]==STAGES
    assert next(r['job'] for r in full['receipts'] if r['stage']==stage)==int(os.environ['SLURM_JOB_ID'])
    w=install(); w.submit=existing_receipt
    if stage=='train_gpu':
        import fresh_training,fresh_training_resume
        fresh_training.run_one_model=fresh_training_resume.run_one_model
    w.main(stage)

if __name__=='__main__':run_stage(sys.argv[1])
