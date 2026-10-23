"""CPU-only terminal/checkpoint evidence; no forward, fit, test, or submission."""
import datetime, hashlib, json, math, os, pathlib, socket, subprocess, sys
R = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(R), str(R/'original_src')]
from budget_guard import native_allocations
from phase1.proposal_trial import finite_tree

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    import torch
    torch.set_num_threads(1)
    job = int(os.environ['SLURM_JOB_ID'])
    raw = subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',str(job)],text=True)
    f = dict(x.split('=',1) for x in raw.split() if '=' in x)
    assert f['JobState']=='RUNNING' and f['Account']=='csci_ga_2271-2026fa'
    assert f['NodeList']==socket.gethostname().split('.')[0] and 'gres/gpu=' not in f['AllocTRES']
    assert sha(R/'source_lock.json')=='4133e4335f87f0b719376f45c71ac9adb52e05b3e2c45550bcb22863472727d9'
    lock=json.loads((R/'source_lock.json').read_text())
    assert all(sha(R/n)==h for n,h in lock['files'].items())
    rows=native_allocations([9105,9119,9120])
    assert all(x['state'] in {'CANCELLED','COMPLETED','FAILED','TIMEOUT'} for x in rows)
    guard=json.loads((R/'watchdog_status.json').read_text())
    row=next(x for x in rows if x['job']==9120)
    entry=guard['GPU_jobs']['9120']
    end=datetime.datetime.fromisoformat(row['end']).replace(tzinfo=datetime.timezone.utc).timestamp()
    charge=max(row['elapsed'],entry['charged_seconds'],math.ceil(end-entry['first_allocated_start']))
    D=pathlib.Path(json.loads((R/'config.json').read_text())['output'])
    budget=json.loads((D/'sequence_budget.json').read_text())
    assert budget['GPU_job_ids']==[8951,9030,9086,9115,9118,9120]
    models=[]
    for p in sorted((D/'models').glob('*/*/seed_*/latest.pt')):
        d=torch.load(p,map_location='cpu',weights_only=False)
        t=json.loads((p.parent/'training.json').read_text()); b=torch.load(p.parent/'best.pt',map_location='cpu',weights_only=False)
        logs=t['logs']; e=d['epoch']; s=d['signature']
        assert logs==d['logs'] and e==len(logs) and d['step']==64*e
        assert [r['epoch'] for r in logs]==list(range(1,e+1))
        assert [r['step'] for r in logs]==list(range(64,e*64+1,64))
        assert s==b['signature']==t['signature'] and finite_tree(d) and finite_tree(b)
        assert d['scheduler']['last_epoch']==e
        assert set(d['rng'])=={'python','numpy','torch','cuda'} and d['rng']['cuda']
        assert all(float(v['step'])==64*e for v in d['optimizer']['state'].values())
        selected=min(logs,key=lambda x:x['validation_mse'])
        assert b['epoch']==selected['epoch'] and b['best']==selected['validation_mse']
        assert s['normalization_sha256']==sha(D/'features/normalization.npz')
        assert s['basis_sha256']==sha(D/'predictor'/s['task']/'basis.npz')
        assert s['permit_sha256']==sha(R/'permit.json')
        key=f"{s['role']}/{s['task']}/{s['seed']}"
        assert budget['epochs'][key]==e
        models.append(dict(key=key,epoch=e,step=d['step'],latest_sha256=sha(p),best_sha256=sha(p.parent/'best.pt'),
                           completion_proof=(p.parent/'completion_proof.json').exists(),finite=True,earliest_best=True,optimizer_scheduler_RNG=True))
    assert sum(x['epoch']==120 and x['completion_proof'] for x in models)==6
    assert next(x for x in models if x['key']=='candidate/gate/17')['epoch']==113
    assert {x['split'] for x in json.loads((D/'dataset/manifest.json').read_text())['records']}=={'train','validation'}
    assert not (D/'all_models_lock.json').exists() and not (D/'confirmation_test_generation_attempt.json').exists()
    out=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),audit_job=job,source_SHA=sha(R/'source_lock.json'),
             source_files=len(lock['files']),source_exact=True,terminals=rows,models=models,encoded_frames=budget['encoded_frames'],
             tests_unopened=True,no_model_forward=True,prior_settled_seconds=44465,charge9120=charge,
             cumulative_GPU_seconds=44465+charge,remaining_GPU_seconds=69299-44465-charge,
             chain_cause='9105 CANCELLED by0; primary lost RUNNING; cleanup cancelled9120 under user account; motive not exposed')
    (R/'execution_repair/audit_9120.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out))

if __name__=='__main__': main()
