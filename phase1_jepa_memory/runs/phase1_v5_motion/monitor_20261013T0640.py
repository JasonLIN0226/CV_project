"""Read-only current training-boundary audit. No held-out inference or fitting."""
import ast
import datetime
import hashlib
import io
import json
from pathlib import Path
import socket
import subprocess
import sys

base = Path('/scratch/xl4358')
controls = {
    'phase1_jepa_memory_v5_gate': '9e2fec4cc7a2b526a275b53f6a96c6180cfe470004f805310e6075ffdb5f9830',
    'phase1_jepa_memory_v5_gate_evaluation': '2055b466634d59adcd951088eab03d6ac9d87d811f49c48d51d4d34958bd14a5',
    'phase1_jepa_memory_v5_motion': 'f230af5ce3b0f2bf85d88eeecca886db218970547975e34629b61eabc843896c',
    'phase1_jepa_memory_v5_motion_evaluation': '1a55fe081690326d441009aa9ab893ce2ae9204c61fd6ad81ea525d8b84e2bdb',
    'phase1_jepa_memory_v5_encoding': '384ef961b067c6c0d7189e02ca370a13d680c5d8e11e13df67d1fdf8b3537ca2',
    'phase1_jepa_memory_v5_encoding_l4': '39fe05f2e7ddb65bf9f144b13e0cd327cf2cdf289c3c204d25ee52334280d88c',
    'phase1_v5_l4_cache_audit_control': 'dac4ffc6ad5c4d2a73887d32f578163604be9773ff0475c043f1da2785651f5e',
}
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
command = lambda *args: subprocess.check_output(args, text=True)
queue = command('/opt/slurm/bin/squeue', '-h', '-j', '7000,7001,7004,7005', '-o', '%i|%j|%a|%P|%T|%N|%R')
assert '7000|cv-v5-motion|csci_ga_2271-2026fa|n2c48m24|RUNNING|' + socket.gethostname() + '|' in queue
assert socket.gethostname() == 'b-10-164'
frozen = {}
for name, want in controls.items():
    root = base / name
    lock = root / 'source_lock.json'
    assert sha(lock) == want, name
    files = json.loads(lock.read_text())['files']
    for rel, digest in files.items():
        path = root / rel
        assert not path.is_symlink() and sha(path) == digest, (name, rel)
    frozen[name] = {'files': len(files), 'source_lock_sha256': want}
control = base / 'phase1_jepa_memory_v5_motion'
sys.path.insert(0, str(base/'phase1_jepa_memory_v5_motion_evaluation/src'))
import numpy as np
import torch
from phase1.v5_motion_training import verify_contract
from phase1.v5_motion_evaluation import validate_checkpoint_pair
from phase1.inputs import FeatureStore
torch.set_num_threads(1)
c, protocol, lock = verify_contract(control, control/'configs/phase1_v5_motion_training_protocol.json')
store = FeatureStore(c)
output = base / 'phase1_v5_motion_development'
contract = json.loads((output/'run_contract.json').read_text())
expected = {
    'protocol_sha256': sha(control/'configs/phase1_v5_motion_training_protocol.json'),
    'source_lock_sha256': controls[control.name],
    'data_config_sha256': protocol['data_source_config_sha256'],
    'cache_signature': store.ix['signature'],
    'basis_sha256': protocol['basis_sha256'],
    'normalization_sha256': sha(Path(c['output'])/'features/normalization.npz'),
    'reader': 'SyntheticCausalReader', 'task': 'motion',
    'selection': 'equal_four_condition_two_horizon_mse_earliest_minimum',
}
assert contract['signature'] == expected and contract['protocol'] == protocol and contract['source_files'] == lock['files']
basis = np.load(Path(c['output'])/'predictor/motion/basis.npz', allow_pickle=False)
train_ids = {r['id'] for r in store.manifest['records'] if r['task']=='motion' and r['split']=='train'}
assert set(basis['train_ids'].tolist()) == train_ids
def finite(value):
    if isinstance(value, torch.Tensor): return bool(torch.isfinite(value).all())
    if isinstance(value, dict): return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)): return all(finite(v) for v in value)
    if isinstance(value, float): return np.isfinite(value)
    return True
progress, complete = {}, {}
for seed in (17, 29, 43):
    root = output/'predictor/motion'/f'seed_{seed}'
    path = root/'latest.pt'
    if not path.exists():
        progress[str(seed)] = {'epoch': 0, 'started': False}
        continue
    data = path.read_bytes()
    ck = torch.load(io.BytesIO(data), map_location='cpu', weights_only=False)
    epoch = ck['epoch']
    assert ck['signature']==expected and ck['seed']==seed and ck['task']=='motion'
    assert ck['horizons']==[3,6] and ck['feature_dim']==5120
    assert ck['resume_granularity']=='epoch_boundary' and ck['checkpoint_selection']==expected['selection']
    assert finite(ck['model']) and finite(ck['optimizer'])
    assert set(ck['rng'])=={'python','numpy','torch','cuda'}
    assert ck['step']==epoch*64 and len(ck['logs'])==epoch
    assert ck['scheduler']['last_epoch']==epoch
    expected_lr=c['training']['lr']*c['training']['lr_gamma']**(epoch//c['training']['lr_step'])
    assert all(np.isclose(g['lr'],expected_lr,rtol=0,atol=1e-15) for g in ck['optimizer']['param_groups'])
    if epoch: assert ck['optimizer']['state']
    assert all(float(s['step'])==ck['step'] for s in ck['optimizer']['state'].values())
    for name in ('input_mean','projection','target_mean','output_basis'):
        assert torch.equal(ck['model'][name],torch.as_tensor(basis[name]).float()),name
    for i,row in enumerate(ck['logs'],1):
        assert row['epoch']==i and row['step']==64*i and finite(row)
        counts=ast.literal_eval(row['mix_counts'])
        assert counts=={k:1024 for k in protocol['conditions']}
        assert np.isclose(np.mean([row['val_'+k+'_mse'] for k in protocol['conditions']]), row['validation_mse'],rtol=0,atol=1e-14)
    scores=[r['validation_mse'] for r in ck['logs']]
    progress[str(seed)]={'epoch':epoch,'step':ck['step'],'best_epoch':scores.index(min(scores))+1 if scores else None,
        'best_validation_mse':min(scores) if scores else None,
        'snapshot_sha256':hashlib.sha256(data).hexdigest(),'partial_or_complete_boundary_proved':True,
        'recent10_mean_epoch_seconds':float(np.mean([r['epoch_seconds'] for r in ck['logs'][-10:]])) if scores else None}
    if epoch==120:
        best_bytes=(root/'best.pt').read_bytes()
        best=torch.load(io.BytesIO(best_bytes),map_location='cpu',weights_only=False)
        proof=validate_checkpoint_pair(seed,ck,best,json.loads((root/'training.json').read_text()),expected)
        expected_best_lr=c['training']['lr']*c['training']['lr_gamma']**(best['epoch']//c['training']['lr_step'])
        assert all(np.isclose(g['lr'],expected_best_lr,rtol=0,atol=1e-15) for g in best['optimizer']['param_groups'])
        for name in ('input_mean','projection','target_mean','output_basis'):
            assert torch.equal(best['model'][name],torch.as_tensor(basis[name]).float())
        proof.update(latest_sha256=hashlib.sha256(data).hexdigest(),best_sha256=hashlib.sha256(best_bytes).hexdigest())
        complete[str(seed)]=proof
proof={
    'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'job_id':7000,'host':socket.gethostname(),'ip':command('hostname','-I').strip(),
    'squeue':queue,'sacct':command('/opt/slurm/bin/sacct','-j','6990,6994,7000,7001,7004,7005','-n','-P','-o','JobID,State,ExitCode,Start,End,NodeList'),
    'progress':progress,'completed_seed_proofs':complete,
    'actual_source_cache_basis_config_rng_optimizer_scheduler_boundary_proved':True,
    'all_frozen_sources_unchanged':frozen,'stderr_bytes':(control/'logs/train-7000.err').stat().st_size,
    'first_epoch_budget':json.loads((output/'first_epoch_budget.json').read_text()),
    'stdout_tail':'\n'.join((control/'logs/train-7000.out').read_text().splitlines()[-12:]),
    'root_read_only_audit_not_independent_delegated_review':True,'held_out_evaluation_executed':False,'phase1_go':False,
}
target=output/'monitor_20261013T0640.json'
assert not target.exists()
target.write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps({'captured_utc':proof['captured_utc'],'progress':progress,'completed_seeds':list(complete),'stderr_bytes':proof['stderr_bytes'],'sha256':sha(target)}))
