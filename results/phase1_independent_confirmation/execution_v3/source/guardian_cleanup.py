import sys,json,hashlib
from pathlib import Path
from watchdog import owned_receipts,discover_submitting_jobs,cancel,write
root=Path(sys.argv[1]);request=json.loads((root/'watchdog_request.json').read_text())
sha=hashlib.sha256((root/'source_lock.json').read_bytes()).hexdigest()
assert sha==request['source_lock_sha256']
known=owned_receipts(root,sha,request['stage_allocation_limits'])
known.update(discover_submitting_jobs(request))
for job in known:cancel(job)
write(root/'guardian_cleanup.json',dict(status='REGISTERED_GPU_CLEANUP_REQUESTED',jobs=known,
    native_terminal_verification_still_required=True,no_GPU=True))
