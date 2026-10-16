"""Read-only final allocation and real partial-checkpoint evidence. No forward."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path('/scratch/xl4358/phase1_predictor_repair/repair_execution')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    current = sys.argv[1]
    out = subprocess.run(['/opt/slurm/bin/scontrol', 'show', 'job', '-o', current],
                         check=True, capture_output=True, text=True).stdout
    fields = dict(x.split('=', 1) for x in out.split() if '=' in x)
    assert fields['JobState'] == 'RUNNING' and fields['Account'] == 'csci_ga_2271-2026fa'
    assert fields['NodeList'] == socket.gethostname().split('.')[0]
    assert 'gres/gpu' not in fields['AllocTRES']
    native = subprocess.run(['/opt/slurm/bin/sacct', '-j', '7477,7480,7485,7517,7518,7520,7523,7524,7525,7526',
                             '-X', '--noheader', '--parsable2',
                             '--format=JobID,State,ExitCode,Start,End,ElapsedRaw,Account,AllocTRES'],
                            check=True, capture_output=True, text=True,
                            env={**os.environ, 'TZ': 'UTC'}).stdout
    jobs = {int(v[0]): dict(zip(('job','state','exit','start','end','elapsed','account','tres'), v))
            for line in native.splitlines() if (v := line.split('|')) and v[0].isdigit()}
    assert all(jobs[j]['account'] == 'csci_ga_2271-2026fa' for j in jobs)
    assert all(jobs[j]['state'] not in ('RUNNING','PENDING','CONFIGURING','COMPLETING')
               for j in (7477,7480,7485,7520,7523,7524,7525,7526))
    assert jobs[7524]['state'] == 'FAILED' and jobs[7524]['exit'] == '3:0'
    assert all('gres/gpu=1' in jobs[j]['tres'] for j in (7477,7480,7485,7524))
    ledger = {str(j): int(jobs[j]['elapsed']) for j in (7477,7480,7485,7524)}
    assert sum(ledger[str(j)] for j in (7477,7480,7485)) == 6247
    locks = {}
    for name, folder, expected in (
        ('science55', ROOT/'control', '9ab65cd2e422085672d753a9b8bb1ff462861271cfbb42899ce54c116e838ab4'),
        ('independent3', ROOT/'independent_audit', '379304f565c5c7fe896c8017113f7747c75c8ce9a746a6bece4eac8a42628da9')):
        assert sha(folder/'source_lock.json') == expected
        lock = json.loads((folder/'source_lock.json').read_text())
        actual = {rel: sha(folder/rel) for rel in lock['files']}
        assert actual == lock['files']
        locks[name] = {'lock_sha256':expected,'current_files':actual}
    import torch
    torch.set_num_threads(1)
    def finite(value):
        if isinstance(value,torch.Tensor):
            return not value.is_floating_point() or bool(torch.isfinite(value).all())
        if isinstance(value,dict): return all(finite(x) for x in value.values())
        if isinstance(value,(list,tuple)): return all(finite(x) for x in value)
        return True
    partials = {}
    for task in ('gate','motion'):
        config = json.loads((Path('/scratch/xl4358/phase1_v3_dev')/task/'reference/resolved_config.json').read_text())
        ref_path = Path(config['output'])/'predictor'/task/'seed_17/best.pt'
        ref = torch.load(ref_path,map_location='cpu',weights_only=False)['model']
        for kind in ('best','latest'):
            path=ROOT/'training/job_7524/predictor'/task/'seed_17'/f'{kind}.pt'
            before=sha(path)
            ck=torch.load(path,map_location='cpu',weights_only=False)
            assert (ck['epoch'],ck['step'],len(ck['logs']),ck['scheduler']['last_epoch']) == (1,64,1,1)
            assert ck['logs'][0]['mix_counts'] == {k:1024 for k in ('recent_only','full_history','privileged_small_budget','random_small_budget')}
            assert finite(ck['model']) and finite(ck['optimizer'])
            assert ck['signature']['source_lock_sha256'] == locks['science55']['lock_sha256']
            assert ck['signature']['reference_best_sha256'] == sha(ref_path)
            assert all(torch.equal(ck['model']['reference.'+k],v) for k,v in ref.items())
            assert all(int(x['step']) == 64 for x in ck['optimizer']['state'].values())
            assert {'python','numpy','torch','cuda'}.issubset(ck['rng'])
            assert sha(path)==before
            partials[f'{task}/{kind}']={'sha256':before,'signature':ck['signature'],
                'epoch':1,'step':64,'scheduler_last_epoch':1,'logs_rows':1,
                'model_and_optimizer_finite':True,'optimizer_steps64':True,
                'exact_current_reference':True,'RNG_keys':sorted(ck['rng']),
                'best_score':ck['best'],'checkpoint_stable':True}
            del ck
    outcome=json.loads((ROOT/'training/job_7524/outcome.json').read_text())
    assert outcome['status']=='COMPUTE_BUDGET_STOP'
    result={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'native_current_zero_GPU_allocation':out,'sacct':native,'jobs':jobs,'locks':locks,
        'GPU_ledger':ledger,'GPU_used_seconds':sum(ledger.values()),
        'GPU_remaining_seconds':18000-sum(ledger.values()),'GPU_cap_seconds':18000,
        'partials':partials,'outcome':outcome,'complete_models':0,
        'held_out_evaluated':False,'fresh9137_opened':False,'phase1_go':False,
        'no_model_forward_or_training':True,'script_sha256':sha(__file__)}
    dest=ROOT/'native_budget_stop.json'
    with dest.open('x') as f: json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'file':str(dest),'sha256':sha(dest),'GPU_ledger':ledger,
                      'GPU_used_seconds':sum(ledger.values()),'GPU_remaining_seconds':18000-sum(ledger.values()),
                      'all4_actual_partials_pass':True,'phase1_go':False}),flush=True)


if __name__=='__main__': main()
