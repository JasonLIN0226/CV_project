"""Read-only original-data CPU execution equivalence; no neural forward."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import numpy as np

BASE=Path('/scratch/xl4358/phase1_predictor_repair/repair_execution')
ROOT=Path('/scratch/xl4358/phase1_predictor_repair/execution_acceleration')
sys.path.insert(0,str(BASE/'control/src'))
from phase1.proposal_trial import verify_control
from phase1.exact_runtime_cache import CachedFeatureStore
from phase1.model import fit_basis
from phase1.training import paired_rows, TRAIN_CONDITIONS
from phase1.utils import file_hash, write_json
from packed_inputs import PackedTrainInputs, compare_original


def main():
    began=time.perf_counter()
    current=os.environ['SLURM_JOB_ID']
    native=subprocess.run(['/opt/slurm/bin/scontrol','show','job','-o',current],check=True,capture_output=True,text=True).stdout
    f=dict(x.split('=',1) for x in native.split() if '=' in x)
    assert f['JobState']=='RUNNING' and f['Account']=='csci_ga_2271-2026fa'
    assert 'gres/gpu' not in f['AllocTRES'] and os.environ['OMP_NUM_THREADS']=='4'
    settle=json.loads((BASE/'native_budget_stop.json').read_text())
    assert settle['GPU_used_seconds']<18000 and settle['jobs']['7524']['state']=='FAILED'
    lock=json.loads((ROOT/'source_lock.json').read_text())
    assert all(file_hash(ROOT/k)==v for k,v in lock['files'].items())
    protocol,_,configs=verify_control(BASE/'control')
    results={}
    for task in protocol['tasks']:
        store=CachedFeatureStore(configs[task],max_cached=1216)
        rows=[r for r in store.manifest['records'] if r['task']==task and r['split'] in ('train','validation')]
        train=paired_rows([r for r in rows if r['split']=='train'])
        basis=fit_basis(configs[task],store,task);store.bases={task:basis}
        for row in rows:store.features(row);store.projected_features(row)
        packed=PackedTrainInputs(store,train,'cpu')
        samples=0;batches=0
        # Original schedule, full training inputs, all seeds and early/middle/last epochs.
        for seed in (17,29,43):
            for epoch in (1,59,119):
                rng=np.random.default_rng(seed+100003*epoch)
                order=rng.permutation(len(train))
                for start in range(0,len(order),32):
                    selected=[train[i] for i in order[start:start+32]]
                    for condition in rng.permutation(TRAIN_CONDITIONS):
                        seeds=[int(rng.integers(2**32)) for _ in selected]
                        state=json.dumps(rng.bit_generator.state,sort_keys=True)
                        samples+=compare_original(store,packed,selected,str(condition),seeds,'cpu')
                        assert json.dumps(rng.bit_generator.state,sort_keys=True)==state
                        batches+=1
        store.verify_boundaries();packed.verify_unchanged()
        results[task]={'exact_samples_targets_metadata':samples,'exact_batches':batches,
                       'outer_RNG_unchanged':True,'metadata_fingerprint':store.metadata_fingerprint,
                       'no_model_forward_or_training':True}
        del packed,store
    result={'job':int(current),'native':native,'source_lock_sha256':file_hash(ROOT/'source_lock.json'),
            'settled_partial_proof_sha256':file_hash(BASE/'native_budget_stop.json'),
            'tasks':results,'all_exact':True,'GPU':0,'seconds':time.perf_counter()-began,
            'fresh9137_opened':False,'phase1_go':False}
    write_json(ROOT/'CPU_preflight.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
