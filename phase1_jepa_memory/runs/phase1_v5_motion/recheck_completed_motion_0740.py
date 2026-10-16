"""Root read-only audit of completed Motion artifacts; never evaluates a model."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import socket
import subprocess
import sys

base = Path('/scratch/xl4358')
output = base/'phase1_v5_motion_evaluation'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
command = lambda *args: subprocess.check_output(args, text=True)
queue = command('/opt/slurm/bin/squeue','-h','-j','7050','-o','%i|%a|%T|%N')
assert '7050|csci_ga_2271-2026fa|RUNNING|'+socket.gethostname() in queue
controls = {
    'phase1_jepa_memory_v5_gate':'9e2fec4cc7a2b526a275b53f6a96c6180cfe470004f805310e6075ffdb5f9830',
    'phase1_jepa_memory_v5_gate_evaluation':'2055b466634d59adcd951088eab03d6ac9d87d811f49c48d51d4d34958bd14a5',
    'phase1_jepa_memory_v5_motion':'f230af5ce3b0f2bf85d88eeecca886db218970547975e34629b61eabc843896c',
    'phase1_jepa_memory_v5_motion_evaluation':'1a55fe081690326d441009aa9ab893ce2ae9204c61fd6ad81ea525d8b84e2bdb',
    'phase1_jepa_memory_v5_encoding':'384ef961b067c6c0d7189e02ca370a13d680c5d8e11e13df67d1fdf8b3537ca2',
    'phase1_jepa_memory_v5_encoding_l4':'39fe05f2e7ddb65bf9f144b13e0cd327cf2cdf289c3c204d25ee52334280d88c',
    'phase1_v5_l4_cache_audit_control':'dac4ffc6ad5c4d2a73887d32f578163604be9773ff0475c043f1da2785651f5e',
}
source_proofs = {}
for name, expected in controls.items():
    control=base/name
    assert sha(control/'source_lock.json')==expected,name
    files=json.loads((control/'source_lock.json').read_text())['files']
    for rel, digest in files.items():
        p=control/rel
        assert p.is_file() and not p.is_symlink() and sha(p)==digest,(name,rel)
    source_proofs[name]={'files':len(files),'source_lock_sha256':expected}
saved=json.loads((output/'all_models_completion.json').read_text())
assert saved['complete'] and set(saved['models'])=={'motion/17','motion/29','motion/43'}
train_root=base/'phase1_v5_motion_development'
for seed in (17,29,43):
    proof=saved['models'][f'motion/{seed}']
    for name in ('best','latest'):
        path=train_root/f'predictor/motion/seed_{seed}/{name}.pt'
        assert path.is_file() and not path.is_symlink() and sha(path)==proof[name+'_sha256']
    assert sha(train_root/f'predictor/motion/seed_{seed}/training.json')==proof['training_json_sha256']
sys.path.insert(0,str(base/'phase1_jepa_memory_v5_motion_evaluation/src'))
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from phase1.v5_motion_evaluation import all_models_completion
current=all_models_completion(base/'phase1_jepa_memory_v5_motion',train_root)
assert current['complete'] and current==saved,'Actual six weights/source/cache/basis/RNG/boundary proof differs'
comparison=json.loads((output/'development_comparison.json').read_text())
metrics=json.loads((output/'evaluation/metrics.json').read_text())
assert sha(output/'evaluation/raw_records.csv')==metrics['raw_sha256']
assert comparison['current_models']==saved
sacct=command('/opt/slurm/bin/sacct','-j','6990,6994,7000,7001,7004,7005,7050','-n','-P','-o','JobID,JobName,Account,Partition,State,ExitCode,Start,End,Elapsed,NodeList,ReqTRES,AllocTRES')
rows=[line.split('|') for line in sacct.splitlines()]
for job in ('7000','7001'):
    row=next(r for r in rows if r[0]==job)
    assert row[2]=='csci_ga_2271-2026fa' and row[4]=='COMPLETED' and row[5]=='0:0',(job,row)
checks=comparison['fixed_reference_checks']
counts={kind:{'passed':sum(x['passed'] for x in checks if x['kind']==kind),'total':sum(x['kind']==kind for x in checks)} for kind in ('candidate_vs_reference_recent','critical_absolute_mse_reduction')}
proof={'status':'ROOT_ACTUAL_SIX_WEIGHTS_RECHECK_EXACTLY_EQUALS_COMPLETED_EVALUATION_NOT_NEW_SCIENCE',
    'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':socket.gethostname(),'control_connection_job':7050,
    'sacct':sacct,'squeue':command('/opt/slurm/bin/squeue','-h','-j','7004,7005,7050','-o','%i|%j|%a|%P|%T|%N|%R'),
    'six_actual_best_latest_proof_exact':True,'models':current,'all_frozen_source_controls_unchanged':source_proofs,
    'adoption':comparison['status'],'candidate_original_task_rules':comparison['candidate_original_task_rules']['status'],'fixed_counts':counts,
    'development_comparison_sha256':sha(output/'development_comparison.json'),'raw_sha256':metrics['raw_sha256'],
    'new_model_inference_or_scientific_training':False,'new_independent_delegated_reviewer':False,'phase1_go':False,
    'stdout_stderr':{str(p):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in (
        base/'phase1_jepa_memory_v5_motion/logs/train-7000.out',base/'phase1_jepa_memory_v5_motion/logs/train-7000.err',
        base/'phase1_jepa_memory_v5_motion_evaluation/logs/eval-7001.out',base/'phase1_jepa_memory_v5_motion_evaluation/logs/eval-7001.err')},
}
path=output/'root_current_weights_recheck_20261013T0740.json'
assert not path.exists();path.write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps({'captured_utc':proof['captured_utc'],'adoption':proof['adoption'],'original_rules':proof['candidate_original_task_rules'],'fixed_counts':counts,'best_epochs':{k:v['best_epoch'] for k,v in current['models'].items()},'proof_sha256':sha(path)}))
