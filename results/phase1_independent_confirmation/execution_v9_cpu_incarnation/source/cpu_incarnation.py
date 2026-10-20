"""Pin the first CPU batch incarnation; never resume a started guardian.

Cloud provisioning may requeue a CPU allocation before its Python process has
ever run. That does not restart a guardian. An exclusive marker distinguishes
this case from a subsequent process/allocation restart, which is refused.
GPU restart rules are unchanged.
"""
import json
from pathlib import Path

ACCOUNT = 'csci_ga_2271-2026fa'


def _cpu(native):
    if (native.get('JobState') != 'RUNNING' or native.get('Account') != ACCOUNT
            or 'gres/gpu=' in native.get('AllocTRES', '')
            or not str(native.get('Restarts', '')).isdigit()):
        raise ValueError('An actual RUNNING course CPU allocation is required')


def claim(root, role, native, source_sha):
    root = Path(root)
    _cpu(native)
    # Both guardians must first start before any GPU submission attempt.
    if any((root/'receipts'/(stage+'_submission_attempt.json')).exists()
           for stage in ('encode_train', 'train_gpu', 'encode_test')):
        raise ValueError('Cannot initialize or replace a guardian after GPU submission')
    proof = dict(job=int(native['JobId']), restarts=int(native['Restarts']),
                 node=native['NodeList'], source_lock_sha256=source_sha,
                 role=role, CPU_only=True, first_batch_start=True)
    with (root/('cpu_incarnation_'+role+'.json')).open('x') as f:
        json.dump(proof, f, indent=2)
    return proof


def require(root, role, job, native, source_sha):
    _cpu(native)
    proof = json.loads((Path(root)/('cpu_incarnation_'+role+'.json')).read_text())
    if (proof['job'] != job or int(native['JobId']) != job
            or int(native['Restarts']) != proof['restarts']
            or native['NodeList'] != proof['node']
            or proof['role'] != role or proof['source_lock_sha256'] != source_sha):
        raise ValueError('CPU guardian incarnation changed after its first batch start')
    return proof
