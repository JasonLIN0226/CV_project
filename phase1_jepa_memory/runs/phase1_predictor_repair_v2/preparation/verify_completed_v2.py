"""Read-only final verification in a real course CPU allocation; no forward."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path('/scratch/xl4358/phase1_predictor_repair_v2/execution')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def main():
    job = sys.argv[1]
    current = subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',job],text=True)
    fields = dict(v.split('=',1) for v in current.split() if '=' in v)
    assert fields['JobState']=='RUNNING' and fields['Account']=='csci_ga_2271-2026fa'
    assert fields['NodeList']==socket.gethostname().split('.')[0] and 'gres/gpu' not in fields['AllocTRES']
    native = subprocess.check_output(['/opt/slurm/bin/sacct','-X','-n','-P','-j',
        '7477,7480,7485,7524,7535,7569,7574,7575,7576','--format=JobID,State,ExitCode,Start,End,ElapsedRaw,Account,AllocTRES'],
        text=True,env={**os.environ,'TZ':'UTC'})
    jobs={int(v[0]):dict(zip(('job','state','exit','start','end','elapsed','account','tres'),v))
          for line in native.splitlines() if (v:=line.split('|')) and v[0].isdigit()}
    assert all(jobs[j]['state']=='COMPLETED' and jobs[j]['exit']=='0:0' for j in (7569,7574,7575,7576))
    assert all(v['account']=='csci_ga_2271-2026fa' for v in jobs.values())
    ledger={str(j):int(jobs[j]['elapsed']) for j in (7477,7480,7485,7524,7535,7574)}
    assert all('gres/gpu=1' in jobs[j]['tres'] for j in (7477,7480,7485,7524,7535,7574))
    assert sum(ledger[str(j)] for j in (7477,7480,7485,7524,7535))==9797 and sum(ledger.values())<=28800
    locks={}
    for folder,expected in [('control','0118e9b61f13d6ae0b271f3faa77bf276d6e7af1f6c97992ae0974902e6ba2d0'),
          ('independent_audit','1e784f6dec75dd6ba820829d8e447e854a2426c706e7020ff0ce9c0685c4e3cc')]:
        path=ROOT/folder
        assert sha(path/'source_lock.json')==expected
        lock=read(path/'source_lock.json')
        actual={k:sha(path/k) for k in lock['files']}
        assert actual==lock['files']
        locks[folder]={'lock_sha256':expected,'current_files':actual}
    assert sha(ROOT/'control/protocol.json')=='4fc74cc1ed12eb7c1f39d62aaf992567f58f94fc8e3c0d308a8844f8e1bd5f57'
    train=ROOT/'training/job_7574'; evaluation=ROOT/'evaluation/job_7575'
    a=read(train/'all_models_completion.json'); b=read(evaluation/'all_models_completion.json')
    assert a['complete'] and b['complete'] and a['models']==b['models'] and len(a['models'])==6
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0,str(ROOT/'control/src'))
    from phase1.proposal_trial import verify_completed_model, verify_control
    protocol, _, configs = verify_control(ROOT/'control')
    proofs={}
    for key,proof in a['models'].items():
        task,seed=key.split('/')
        folder=train/'predictor'/task/f'seed_{seed}'
        reference=Path('/scratch/xl4358/phase1_v3_dev')/task/'reference/predictor'/task/f'seed_{seed}/best.pt'
        assert sha(reference)==proof['signature']['reference_best_sha256']
        ref=torch.load(reference,map_location='cpu',weights_only=False)['model']
        got=verify_completed_model(folder,proof['signature'],ref)
        assert got==proof
        assert sha(Path(configs[task]['output'])/'features/normalization.npz')==proof['signature']['normalization_sha256']
        proofs[key]=got
    sys.path.insert(0,str(ROOT/'independent_audit'))
    from verify_scalar_results import audit
    saved=read(evaluation/'independent_scalar_audit.json'); recomputed=audit(evaluation)
    assert all(saved[k]==v for k,v in recomputed.items())
    outcome=read(evaluation/'outcome.json')
    assert not outcome['phase1_go'] and not outcome['confirmation_opened']
    compact={}
    patterns=['training_submission.json','analysis_submission.json','independent_submission.json',
        'training/job_7574/*.json','training/job_7574/predictor/*/seed_*/training.json',
        'training/job_7574/predictor/*/seed_*/training.csv','evaluation/job_7575/*.json',
        'evaluation/job_7575/*/development_comparison.json','evaluation/job_7575/*/evaluation/metrics.json',
        'evaluation/job_7575/*/evaluation/calibration_summary.json','logs/train-7574.*',
        'logs/evaluate-7575.*','logs/independent-7576.*']
    for pattern in patterns:
        for path in ROOT.glob(pattern):
            if path.is_file() and path.suffix in ('.json','.csv','.out','.err') and path.stat().st_size<1500000:
                compact[str(path.relative_to(ROOT))]={'sha256':sha(path),'bytes':path.stat().st_size}
    result={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'native_current_CPU_allocation':current,
        'sacct':native,'jobs':jobs,'GPU_ledger':ledger,'GPU_used_seconds':sum(ledger.values()),
        'GPU_remaining_seconds':28800-sum(ledger.values()),'GPU_cap_seconds':28800,
        'locks':locks,'six_current_checkpoint_proofs':proofs,'independent_raw_recalculation':recomputed,
        'development_status':outcome['status'],'tasks':{t:{'status':v['status'],
            'original_passed':sum(bool(c['passed']) for c in v['original_checks']),
            'original_total':len(v['original_checks']),'fixed_passed':sum(bool(c['passed']) for c in v['fixed_reference_checks']),
            'fixed_total':len(v['fixed_reference_checks']),'reasons':v['reasons']} for t,v in outcome['tasks'].items()},
        'compact_ledger':compact,'no_forward_or_training':True,'phase1_go':False,
        'confirmation_opened':False,'script_sha256':sha(__file__)}
    dest=ROOT/'final_native_verification.json'
    with dest.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'file':str(dest),'sha256':sha(dest),'GPU_used_seconds':sum(ledger.values()),
        'development_status':outcome['status'],'tasks':result['tasks']}),flush=True)

if __name__=='__main__':
    main()
