"""Read-only new-source input equivalence and one guarded native submission chain."""
import json
import os
from pathlib import Path
import subprocess
import re
import hashlib
import datetime
import torch

from phase1.proposal_trial import verify_control
from phase1.exact_runtime_cache import CachedFeatureStore
from phase1.model import fit_basis
from phase1.training import paired_rows
from phase1.resource_guard import reconcile_prior
from phase1.utils import write_json, file_hash
from phase1.common_anchor import recent_only_teacher_batch
from phase1.common_anchor_legacy import recent_only_teacher_batch as original_recent_only_teacher_batch

ROOT = Path('/scratch/xl4358/phase1_predictor_repair_v3/execution')
CONTROL = ROOT / 'control'


def call(args):
    return subprocess.run(args, text=True, capture_output=True, check=True).stdout


def submit_once(name, command):
    marker, receipt = ROOT / (name + '_submission_attempt.json'), ROOT / (name + '_submission.json')
    if marker.exists() or receipt.exists():
        raise ValueError('Existing submission marker/outcome; no implicit retry: ' + name)
    attempt = {'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'command': command, 'source_lock_sha256': file_hash(CONTROL / 'source_lock.json')}
    with marker.open('x') as handle:
        json.dump(attempt, handle, indent=2); handle.write('\n')
    result = subprocess.run(command, text=True, capture_output=True)
    record = {**attempt, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    if result.returncode == 0 and re.fullmatch(r'[0-9]+(?:;[A-Za-z0-9_-]+)?\n?', result.stdout):
        record.update(job_id=int(result.stdout.strip().split(';')[0]), status='SUBMITTED_ACTUAL_RECEIPT')
    else:
        record['status'] = 'SUBMISSION_OUTCOME_REQUIRES_INSPECTION_NO_RETRY'
    write_json(receipt, record)
    if 'job_id' not in record:
        raise RuntimeError('Unresolved actual submission outcome; stop, do not retry')
    print(json.dumps(record), flush=True)
    return record['job_id']


def main():
    if os.environ.get('SLURM_JOB_ACCOUNT') != 'csci_ga_2271-2026fa' or not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Actual course CPU allocation required')
    proof_path = ROOT / 'native_preflight.json'
    if proof_path.exists():
        raise ValueError('Refuse overwrite / implicit CPU preflight resubmission')
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    protocol, lock, configs = verify_control(CONTROL)
    if file_hash(CONTROL / 'repair_decision.json') != protocol['repair_decision_sha256']:
        raise ValueError('Fixed post-diagnosis repair choice changed')
    independent = ROOT / 'independent_audit'
    if file_hash(independent / 'source_lock.json') != protocol['independent_source_lock_sha256']:
        raise ValueError('Independent audit source lock changed')
    for name, expected in json.loads((independent / 'source_lock.json').read_text())['files'].items():
        if file_hash(independent / name) != expected:
            raise ValueError('Independent raw scalar audit source changed')
    parents = call(['/opt/slurm/bin/sacct', '-j', '7485,7486,7487,7518,7535,7536,7537,7574,7575,7576', '-X', '--noheader',
                    '--parsable2', '--format=JobID,State,ExitCode'])
    if set(parents.splitlines()) != {str(j) + '|COMPLETED|0:0' for j in (7485, 7486, 7487, 7518, 7535, 7536, 7537, 7574, 7575, 7576)}:
        raise ValueError('Actual completed prior science, evaluation, audit and read-only diagnosis required')
    ledger = call(['/opt/slurm/bin/sacct', '-j', '7477,7480,7485,7524,7535,7574', '-X', '--noheader',
                   '--parsable2', '--format=JobID,State,ElapsedRaw,Account,AllocTRES'])
    rows = []
    for line in ledger.splitlines():
        job, state, elapsed, account, allocated = line.split('|')[:5]
        tres = dict(v.split('=', 1) for v in allocated.split(','))
        rows.append({'job_id': int(job), 'state': state, 'elapsed_raw': int(elapsed),
                     'account': account, 'allocated_gpus': int(tres.get('gres/gpu', 0))})
    reconciled = reconcile_prior(rows)
    active_gpu = call(['/opt/slurm/bin/squeue', '-u', 'xl4358', '-h', '-o', '%i %b'])
    if any('gpu' in line.lower() for line in active_gpu.splitlines()):
        raise ValueError('Existing GPU allocation/receipt must be resolved before a new trial')
    tasks = {}
    for task, config in configs.items():
        store = CachedFeatureStore(config, max_cached=1216)
        basis = fit_basis(config, store, task); store.bases = {task: basis}
        legal = [row for row in store.manifest['records'] if row['task'] == task and row['split'] in {'train', 'validation'}]
        for row in legal:
            store.features(row); store.projected_features(row)
        result = store.verify_all_inputs()
        validation = paired_rows([row for row in legal if row['split'] == 'validation'])
        store.prepare_validation(validation, 'cpu')
        result['exact_validation_batches'] = store.verify_validation_inputs(validation)
        transformed = 0
        for entries in store.validation_batches.values():
            for batch, _, _, versions in entries:
                original, current = original_recent_only_teacher_batch(batch), recent_only_teacher_batch(batch)
                if (set(original) != set(current) or any(not torch.equal(original[k], current[k]) for k in original)
                        or any(batch[k]._version != version for k, version in versions.items())):
                    raise ValueError('Exact vectorized causal input boundary differs / mutated inputs')
                transformed += 1
        result['exact_original_recent_boundary_batches'] = transformed
        if result['exact_samples_targets_info_rng'] != 4864 or result['exact_validation_batches'] != 12:
            raise ValueError('Incomplete original legal input/target/info/RNG/validation proof')
        tasks[task] = result
        del store
    # Check immutable executable files again after all CPU data reads.
    verify_control(CONTROL)
    proof = {'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'job': os.environ['SLURM_JOB_ID'], 'source_lock_sha256': file_hash(CONTROL / 'source_lock.json'),
             'all_exact': True, 'tasks': tasks, 'prior_native_parents': parents, 'native_GPU_ledger': ledger,
             'prior_GPU_used_seconds': reconciled['used_seconds'],
             'remaining_shared_GPU_seconds': reconciled['remaining_seconds'],
             'no_CPU_model_training': True, 'no_fresh9137_access': True, 'phase1_go': False}
    write_json(proof_path, proof)
    parent = os.environ['SLURM_JOB_ID']
    training = submit_once('training', ['/opt/slurm/bin/sbatch', '--parsable', '--dependency=afterok:' + parent,
                                      str(CONTROL / 'repair_train.sbatch')])
    evaluation = submit_once('analysis', ['/opt/slurm/bin/sbatch', '--parsable', '--dependency=afterany:' + str(training),
                                        '--export=ALL,TRAINING_JOB=' + str(training), str(CONTROL / 'repair_evaluate.sbatch')])
    submit_once('independent', ['/opt/slurm/bin/sbatch', '--parsable', '--dependency=afterany:' + str(evaluation),
                                str(ROOT / 'independent_audit/audit.sbatch')])


if __name__ == '__main__':
    main()
