"""Single-receipt CPU/GPU stage chain; no scientific work occurs at import.

All GPU allocations share one actual wall ledger. CPU rendering, normalization,
basis/readout fitting and final evaluation are outside any GPU reservation.
"""
import datetime
import json
import math
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'original_src'))
from phase1.utils import read_json,write_json,file_hash,digest,seed_all
from budget_guard import SequenceGuard,CAP,PRIOR,native_allocations,BudgetStop
from startup_client import ensure as startup_ensure, require_live as startup_require_live

CHAIN=['preflight_cpu','preflight_gpu','init_cpu','encode_train','fit_cpu','train_gpu',
       'lock_cpu','encode_test','evaluate_cpu','audit_cpu','evidence_cpu']
GPU_LIMITS={'preflight_gpu':600,'encode_train':2700,'train_gpu':6600,'encode_test':3900}


def source_check():
    lock=read_json(ROOT/'source_lock.json')
    for name,sha in lock['files'].items():
        if file_hash(ROOT/name)!=sha:raise ValueError('Frozen complete source changed: '+name)
    protocol=read_json(ROOT/'protocol.json')
    if protocol['GPU_cap_seconds']!=CAP or protocol['prior_settled_seconds']!=PRIOR:
        raise ValueError('Explicit new cumulative authorization changed')
    authorization=read_json(ROOT/'restart_authorization.json')
    if (authorization['cumulative_cap_seconds']!=CAP or authorization['prior_actual_GPU_seconds']!=PRIOR
            or authorization['additional_GPU_seconds']!=CAP-PRIOR
            or authorization['status']!='HUMAN_APPROVED_ADDITIONAL_GPU_BUDGET'):
        raise ValueError('Explicit additional budget authorization required')
    for name,sha in protocol['development_evidence'].items():
        if file_hash(name)!=sha:raise ValueError('Accepted native development evidence changed: '+name)
    return protocol,lock


def native(stage):
    job=int(os.environ['SLURM_JOB_ID'])
    raw=subprocess.run(['/opt/slurm/bin/scontrol','show','job','-o',str(job)],
        check=True,capture_output=True,text=True).stdout
    f=dict(v.split('=',1) for v in raw.split() if '=' in v)
    gpu=stage in GPU_LIMITS
    if (f['JobState']!='RUNNING' or f['Account']!='csci_ga_2271-2026fa'
            or f['NodeList']!=socket.gethostname().split('.')[0]
            or f['Partition']!=('c12m85-a100-1' if gpu else 'n2c48m24')
            or ('gres/gpu=1' in f['AllocTRES'])!=gpu):
        raise ValueError('Current actual course allocation/stage mismatch')
    if gpu:
        if f.get('Requeue') not in ('0','1') or f.get('Restarts')!='0':
            raise ValueError('Known native requeue flag and zero restarts required with paired external guardians')
        startup_require_live(ROOT,file_hash(ROOT/'source_lock.json'))
    return job,raw,f


def terminal(job):
    row=native_allocations([job])[0]
    if row['state']!='COMPLETED' or row['exitcode']!='0:0':
        raise ValueError('Parent stage lacks actual COMPLETED0:0 evidence')
    return row


def submit(stage,parent=None):
    source_check()
    receipts=ROOT/'receipts';receipts.mkdir(exist_ok=True)
    marker=receipts/(stage+'_submission_attempt.json')
    # Unknown/prior outcomes are never submitted again.
    with marker.open('x') as f:json.dump(dict(stage=stage,parent=parent,unknown_do_not_retry=True),f)
    gpu=stage in GPU_LIMITS
    if gpu:
        startup_require_live(ROOT,file_hash(ROOT/'source_lock.json'))
        report=SequenceGuard(ROOT/'fresh').check()
        limit=min(GPU_LIMITS[stage],int(report['remaining_cap_seconds'])-120)
        if limit<GPU_LIMITS[stage]:raise BudgetStop('Full declared stage hard limit no longer fits')
    else:limit=10800 if stage=='evaluate_cpu' else 3600
    if stage=='evidence_cpu':limit=900
    args=['/opt/slurm/bin/sbatch','--parsable','--account=csci_ga_2271-2026fa',
        '--partition='+('c12m85-a100-1' if gpu else 'n2c48m24'),
        '--cpus-per-task='+('1' if stage in ('audit_cpu','evidence_cpu') else '4'),
        '--mem='+('60G' if gpu else '1500M' if stage=='evidence_cpu' else '16G'),
        '--time='+str(math.ceil(limit/60)), '--no-requeue',
        '--job-name=p1-'+stage+'-'+file_hash(ROOT/'source_lock.json')[:8],
        '--output='+str(ROOT/(stage+'-%j.out')),'--error='+str(ROOT/(stage+'-%j.err'))]
    if gpu:args.append('--gres=gpu:1')
    if parent is not None:args.append('--dependency=afterok:'+str(parent))
    args.extend(['--export=NIL',str(ROOT/'stage.sh'),stage])
    result=subprocess.run(args,check=True,capture_output=True,text=True,timeout=30).stdout.strip()
    job=int(result.split(';')[0])
    receipt=dict(stage=stage,job=job,parent=parent,stdout=result,
        source_lock_sha256=file_hash(ROOT/'source_lock.json'),actual_submission_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write_json(receipts/(stage+'_submission.json'),receipt)
    if gpu:
        guard=SequenceGuard(ROOT/'fresh');guard.state.setdefault('GPU_job_ids',[]).append(job);guard.save()
    print(json.dumps(receipt),flush=True)
    return receipt


def reconcile_prior():
    expected={7477:100,7480:126,7485:6021,7524:164,7535:3386,7574:2807,
              7582:119,8496:42,8511:312,8512:27,8518:27,8536:26520,8792:61,8799:185}
    rows=native_allocations(list(expected))
    for row in rows:
        if (row['elapsed']!=expected[row['job']] or row['state'] not in ('COMPLETED','FAILED','CANCELLED','TIMEOUT')
                or row['tres'].get('gres/gpu')!='1' or row['account']!='csci_ga_2271-2026fa'):
            raise ValueError('Previous actual GPU ledger changed/incomplete')
    if sum(expected.values())!=PRIOR:raise ValueError('Prior settlement mismatch')
    data=ROOT/'fresh';data.mkdir(exist_ok=False)
    write_json(data/'sequence_budget.json',dict(prior_settled_seconds=PRIOR,cap_seconds=CAP,
        native_prior_verified=True,native_prior=rows,total_frames=198656,GPU_job_ids=[],epochs={},encoded_frames=0))
    # Release verification for the old evidence receiver is read-only.
    previous=native_allocations([8523])[0]
    if previous['state'] in ('RUNNING','CONFIGURING','COMPLETING'):
        raise ValueError('Prior evidence receiver is still active; no duplicate control')
    write_json(ROOT/'resource_release8523.json',previous)


def config():return read_json(ROOT/'config.json')


def make_permit(c):
    protocol,lock=source_check()
    proofs=[ROOT/'preflight_cpu.json',ROOT/'preflight_gpu.json']
    if any(not read_json(p).get('all_exact',False) for p in proofs):
        raise ValueError('Full native development equivalence required before fresh generation')
    evidence={str(p):file_hash(p) for p in proofs}
    for name,sha in protocol['development_evidence'].items():
        if file_hash(name)!=sha:raise ValueError('Validated development evidence changed')
        evidence[name]=sha
    p=ROOT/'permit.json'
    with p.open('x') as f:
        json.dump(dict(status='FROZEN_CONFIRMATION_AUTHORIZED',development_tasks={'gate':'SUPPORTED','motion':'SUPPORTED'},
            independent_development_audit='PASS',GPU_cap_seconds=CAP,canonical_config_digest=digest(c),
            all_models_lock=str(Path(c['output'])/'all_models_lock.json'),evidence_files=evidence,
            source_files={str(ROOT/k):v for k,v in lock['files'].items()},
            source_lock_sha256=file_hash(ROOT/'source_lock.json'),
            method='Gate transportOFF;Motion same-history visual+guardedposition+fixed.5training-onlyinnovation',
            fresh_seed=9137,no_retuning_after_confirmation_failure=True),f,indent=2)
    return p


def execute(stage):
    protocol,source=source_check();job,allocation,fields=native(stage)
    print('JOB',job,'HOST',socket.gethostname(),'IP',socket.gethostbyname(socket.gethostname()),flush=True)
    print(Path('/etc/ssh/ssh_host_ed25519_key.pub').read_text(),flush=True)
    attempt=ROOT/'stage_attempts'/(stage+'.json');attempt.parent.mkdir(exist_ok=True)
    with attempt.open('x') as f:json.dump(dict(stage=stage,job=job,native=allocation,source_lock_sha256=file_hash(ROOT/'source_lock.json')),f)
    previous=CHAIN[CHAIN.index(stage)-1] if stage!='preflight_cpu' else None
    if previous:
        receipt=read_json(ROOT/'receipts'/(previous+'_submission.json'))
        parent=terminal(receipt['job']);write_json(ROOT/(previous+'_terminal.json'),parent)
    c=config();data=Path(c['output']);permit=ROOT/'permit.json'
    if stage=='preflight_cpu':
        reconcile_prior()
        startup_ensure(ROOT,protocol,GPU_LIMITS)
        from guardian_preflight import verify
        verify(ROOT)
        from cpu_equivalence_reuse import inherit
        inherit(ROOT,protocol)
    elif stage=='preflight_gpu':
        from dev_equivalence import gpu_equivalence
        gpu_equivalence(ROOT,protocol,SequenceGuard(data))
    elif stage=='init_cpu':
        SequenceGuard(data).check();make_permit(c)
        from staged_data import generate_train_validation
        generate_train_validation(c,permit)
    elif stage in ('encode_train','encode_test'):
        from phase1.repair_trial import actual_allocation_start
        from guarded_encode import encode
        allocation_start,_=actual_allocation_start()
        encode(c,permit,SequenceGuard(data),allocation_start,training_only=stage=='encode_train')
    elif stage=='fit_cpu':
        from staged_data import require_stage,audit_stage_features,stage_load_index,fit_stage_normalization,stage_subset_lock
        from staged_probes import fit_training_readouts
        audit_stage_features(c,require_stage(c),stage_load_index(c,complete=False))
        fit_stage_normalization(c);fit_training_readouts(c,permit)
        write_json(data/'training_subset_lock.json',stage_subset_lock(c))
        from train_maps import prepare
        prepare(c,permit)
    elif stage=='train_gpu':
        from fresh_training import make_task_store,run_one_model
        from phase1.repair_trial import actual_allocation_start
        from train_maps import fit_cuda
        guard=SequenceGuard(data);start,_=actual_allocation_start();guard.check(start)
        stop={'requested':False}
        for signum in (signal.SIGTERM,signal.SIGINT):signal.signal(signum,lambda *_:stop.update(requested=True))
        from phase1.resource_guard import elapsed_from_allocation
        deadline=time.perf_counter()+GPU_LIMITS['train_gpu']-elapsed_from_allocation(start)-120
        contexts={task:make_task_store(c,task) for task in ('gate','motion')}
        for role in ('reference','candidate'):
            for task in ('gate','motion'):
                store,basis,train,val=contexts[task]
                for seed in (17,29,43):
                    reference=data/f'models/reference/{task}/seed_{seed}/best.pt'
                    result=run_one_model(c,task,seed,role,data/f'models/{role}/{task}/seed_{seed}',
                        store,basis,train,val,permit,guard,stop,deadline,
                        reference_path=reference if role=='candidate' else None)
                    if result is None:raise BudgetStop('Actual incomplete training boundary; no test access or retry')
        fit_cuda(c,permit,guard,start)
        guard.state['maps_locked']=True;guard.save();guard.check(start)
    elif stage=='lock_cpu':
        write_json(data/'training_terminal.json',read_json(ROOT/'train_gpu_terminal.json'))
        from locks import create_all_lock
        from staged_data import append_locked_tests
        create_all_lock(c,permit)
        SequenceGuard(data).check()
        append_locked_tests(c,read_json(data/'training_subset_lock.json'),permit)
    elif stage=='evaluate_cpu':
        from phase1 import data as original_data,cache
        from staged_data import verify_training_subset
        from staged_probes import evaluate_frozen_readouts
        from fresh_evaluation import evaluate_role,compare_task
        original_data.require_audit(c)
        ix=cache.load_index(c,complete=False);cache.audit_features(c,original_data.manifest(c),ix)
        subset=read_json(data/'training_subset_lock.json');verify_training_subset(c,subset)
        guard=SequenceGuard(data);guard.state['full_encoding_audited']=True;guard.save();guard.check()
        evaluate_frozen_readouts(c,subset,permit)
        results={}
        for task in ('gate','motion'):
            reference,_=evaluate_role(c,task,'reference',permit)
            candidate,metrics=evaluate_role(c,task,'candidate',permit)
            results[task]=compare_task(c,task,reference,candidate,metrics)
        from fresh_deletion import evaluate
        deletion=evaluate(c,permit)
        write_json(data/'primary_outcome.json',dict(tasks=results,deletion_acceptance=deletion['acceptance'],
            phase1_go=False,independent_final_audit_pending=True,fresh9137=True))
    elif stage=='audit_cpu':
        from final_audit import audit
        audit(c,permit)
    elif stage=='evidence_cpu':
        from terminal_evidence import collect
        collect(ROOT,c)
    else:raise ValueError('Unknown declared stage')
    source_check()
    write_json(ROOT/'stage_completed'/(stage+'.json'),dict(stage=stage,job=job,execution_completed=True,
        scientific_pass_not_implied=True,source_lock_sha256=file_hash(ROOT/'source_lock.json')))
    if stage!='evidence_cpu':submit(CHAIN[CHAIN.index(stage)+1],job)
    else:write_json(ROOT/'watchdog_done.json',dict(all_GPU_ended=True))


def main(stage):
    import torch
    torch.set_num_threads(1 if stage in ('audit_cpu','evidence_cpu') else 4)
    torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    try:execute(stage)
    except Exception:
        p=ROOT/'stage_failures'/(stage+'.json');p.parent.mkdir(exist_ok=True)
        write_json(p,dict(stage=stage,job=os.environ.get('SLURM_JOB_ID'),error=traceback.format_exc(),
            phase1_go=False,no_automatic_retry=True,fresh_retuning_forbidden=True))
        raise


if __name__=='__main__':main(sys.argv[1])
