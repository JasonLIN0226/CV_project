"""Independent final fresh decision; failures remain failures, with no retuning."""
from pathlib import Path
import csv
from phase1.utils import read_json,write_json,file_hash
from staged_data import verify_permit,verify_training_subset
from fresh_audit import audit_task,audit_vectors,audit_trace
from deletion_independent import audit_rows
from budget_guard import SequenceGuard


def audit(c,permit):
    verify_permit(c,permit,models_locked=True)
    root=Path(c['output'])
    verify_training_subset(c,read_json(root/'training_subset_lock.json'))
    if not read_json(root/'probes/frozen_readout_final_verification.json')['passed']:
        raise ValueError('Train-only readout/fit-input lock failed')
    tasks={};vectors={};traces={}
    for task in ('gate','motion'):
        tasks[task]=audit_task(c,task)
        for role in ('reference','candidate'):
            folder=root/'evaluation'/role/task
            vectors[role+'/'+task]=audit_vectors(c,folder,task,role)
        if task=='motion':traces[task]=audit_trace(c,root/'evaluation/candidate'/task)
    folder=root/'deletion';summary=read_json(folder/'summary.json')
    if (file_hash(folder/'raw_deletions.csv')!=summary['raw_sha256']
            or file_hash(folder/'paired_policy_losses.csv')!=summary['paired_sha256']):
        raise ValueError('Fresh raw/paired deletion checksum changed')
    with (folder/'raw_deletions.csv').open() as f:rows=list(csv.DictReader(f))
    with (folder/'paired_policy_losses.csv').open() as f:paired=list(csv.DictReader(f))
    deletion=audit_rows(rows,paired,summary,pairs=256,repeats=10000)
    vectors['deletion']=audit_vectors(c,folder)
    traces['deletion']=audit_trace(c,folder)
    primary=read_json(root/'primary_outcome.json')
    flags={task:(tasks[task]['original_possible_go'] and
        all(v['passed'] for v in tasks[task]['fixed_checks_replay']) and
        summary['acceptance'][task]['passed']==72) for task in ('gate','motion')}
    for task in flags:
        if (tasks[task]['original_possible_go'] and all(v['passed'] for v in tasks[task]['fixed_checks_replay']))!=(primary['tasks'][task]['status']=='SUPPORTED'):
            raise ValueError('Independent prediction decision differs from primary')
    verify_permit(c,permit,models_locked=True)
    budget=SequenceGuard(root).check()
    result=dict(status='PASS_INDEPENDENT_FRESH_PROTOCOL_AUDIT',tasks=tasks,vectors=vectors,
        actual_input_traces=traces,deletion=deletion,
        accepted_original_thresholds_unchanged=True,all12_fresh_models_locked_before_any_test=True,
        all_train_only_transforms_and_two_maps_locked=True,
        no_model_forward_or_fitting_in_final_audit=True,
        GPU_settled_seconds=budget['actual_GPU_seconds'],GPU_cap_seconds=28800,
        source_permit_sha256=file_hash(permit),all_models_lock_sha256=file_hash(root/'all_models_lock.json'))
    write_json(root/'independent_final_audit.json',result)
    go=all(flags.values())
    write_json(root/'final_outcome.json',dict(status='PHASE1_GO' if go else 'PHASE1_ORIGINAL_STANDARD_NOT_MET',
        phase1_go=go,fresh9137=True,independent_audit='PASS',tasks_supported=flags,
        prediction_counts={task:dict(original=primary['tasks'][task]['original_passed'],original_total=51,
            fixed=primary['tasks'][task]['fixed_passed'],fixed_total=32) for task in flags},
        deletion_counts=summary['acceptance'],all_failed_slices_retained=True,
        no_retuning_after_confirmation=True,retention_policy_trained=False,
        scope='Controlled synthetic I-JEPA plus fixed supervised readout and observed-RGB guard; not V-JEPA or real-video validation',
        learned_policy_advantage_or_online_utility_estimation_proven=False,
        GPU_settled_seconds=budget['actual_GPU_seconds'],GPU_cap_seconds=28800,
        independent_audit_sha256=file_hash(root/'independent_final_audit.json')))
