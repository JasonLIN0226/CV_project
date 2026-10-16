"""Read-only current allocation audit; no prediction, fit, or training."""
import datetime
import gc
import hashlib
import io
import json
from pathlib import Path
import subprocess
import torch
from memory_jepa.utils.hashing import canonical_hash, state_dict_hash
from memory_jepa.utils.modeling import build_predictor

ROOT = Path('/scratch/xl4358/phase1_space_paper_training')
OUT = Path('/scratch/xl4358/phase1_space_predictor_results/job_7313')
RESULT = OUT / 'root_current_partial_proof_20261014T0857.json'
EXPECTED = {
    'phase1_space_paper_training': '46090226dc4ca7aaf50a30606922c4bacb829fa9bc0b218bf97b359f3d17b8c1',
    'phase1_space_paper_artifacts': '43f3fa19795205288d39a347daaa80a71ca701f5cf5e91da20c7ec34fcb951fc',
    'phase1_space_paper_benchmark': 'caf632d26d7d2c4f7598621433f92de3af53e30ab4473dc33f7782b6d2ec63cc',
    'phase1_space_postartifact_handoff': 'd1d1e6984abd0fa0c8c31505d5f6bdae011d7fbde6ddc3f1a4a95f4de550e014',
    'phase1_space_postpredictor_complete_handoff': '33142a0689a1152dff454845d098a66e5440cc9fd80f7f928459d07a0eefc319',
}

def sha(p):
    assert not p.is_symlink(), 'symlink not allowed'
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def sources():
    result = {}
    for name, expected in EXPECTED.items():
        root = ROOT.parent / name
        lockpath = root / ('software_lock.json' if name in
                           ('phase1_space_paper_artifacts', 'phase1_space_paper_benchmark') else 'source_lock.json')
        assert sha(lockpath) == expected, 'source lock differs'
        lock = json.loads(lockpath.read_text())
        for relative, digest in lock['files'].items():
            assert sha(root / relative) == digest, 'frozen source differs'
        result[name] = {'files': len(lock['files']), 'lock_sha256': expected,
                        'all_current_hashes_match': True}
    return result

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
actual = subprocess.check_output(['/opt/slurm/bin/scontrol', 'show', 'job', '-o', '7313'], text=True)
job = dict(p.split('=', 1) for p in actual.split() if '=' in p)
assert job['JobState'] == 'RUNNING' and job['Account'] == 'csci_ga_2271-2026fa'
assert job['Partition'] == 'c12m85-a100-1' and 'gres/gpu=1' in job['AllocTRES']
assert subprocess.check_output(['hostname'], text=True).strip() == job['BatchHost']
assert not RESULT.exists(), 'fixed output already exists; do not repeat audit'
protocol_path = ROOT / 'paper_training_protocol.json'
protocol = json.loads(protocol_path.read_text())
assert sha(protocol_path) == 'fdbf80fac1845b4cd08acd872e261ebeb0b84945d9acd97d2938d87184d69498'
data_path = Path(protocol['data_report'])
assert sha(data_path) == protocol['data_report_sha256']
data = json.loads(data_path.read_text())
manifest = Path(data['manifest'])
assert sha(manifest) == protocol['manifest_sha256']
assert len(data['features']) == 294 and not data['test_payload_deserialized']

def features():
    for row in data['features']:
        p = manifest.parent / (row['video_id'] + '.pt')
        assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], 'feature differs'
    return True

proof = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'job_id': 7313,
         'actual_native_jobs': subprocess.check_output(['/opt/slurm/bin/sacct', '-X', '-j', '7313,7323',
                  '-n', '-P', '--format=JobID,State,ExitCode,Start,End,Account'], text=True),
         'source_groups': sources(), 'all294_current_features_before_match': features(),
         'models': [], 'boundary_rejections': [], 'heldout_inference': False,
         'snapshot_method': 'single byte snapshot plus afterSHA and unchanged history boundary'}
for seed in (0, 1, 2):
    folder = OUT / f'seed_{seed}'
    hp = folder / 'predictor_history.json'
    if not hp.exists():
        proof['models'].append({'seed': seed, 'stage': 'not yet started'})
        continue
    history_bytes = hp.read_bytes()
    history = json.loads(history_bytes)
    assert [r['epoch'] for r in history] == list(range(1, len(history) + 1))
    earliest = min(range(len(history)), key=lambda i: history[i]['validation']['loss']) + 1
    for tag in ('best', 'latest'):
        p = folder / f'predictor_{tag}.pt'
        assert not p.is_symlink()
        payload = p.read_bytes()
        before = hashlib.sha256(payload).hexdigest()
        ck = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
        del payload
        if sha(p) != before or hp.read_bytes() != history_bytes:
            proof['boundary_rejections'].append({'seed': seed, 'tag': tag, 'reason': 'active atomic update'})
            del ck
            gc.collect()
            continue
        wanted = earliest if tag == 'best' else len(history)
        if ck['epoch'] != wanted:
            proof['boundary_rejections'].append({'seed': seed, 'tag': tag, 'reason': 'checkpoint/history transition'})
            del ck
            gc.collect()
            continue
        assert ck['seed'] == seed and ck['global_step'] == wanted * 1233 == history[wanted - 1]['steps']
        assert ck['source_lock_sha256'] == EXPECTED['phase1_space_paper_training']
        assert ck['protocol_sha256'] == sha(protocol_path) and ck['manifest_sha256'] == sha(manifest)
        assert ck['config_hash'] == canonical_hash(ck['config'])
        model = build_predictor(ck['config'], int(ck['d_in']))
        model.load_state_dict(ck['model'], strict=True)
        assert state_dict_hash(model) == ck['predictor_hash']
        assert ck['optimizer']['state'] and ck['optimizer']['param_groups'][0]['weight_decay'] == .05
        assert ck['scheduler']['last_epoch'] == ck['global_step']
        assert set(ck['rng']) == {'python', 'numpy', 'torch_cpu', 'torch_cuda'}
        tensors = list(ck['model'].values()) + [v for state in ck['optimizer']['state'].values()
                   for v in state.values() if isinstance(v, torch.Tensor)]
        assert all(torch.isfinite(t).all() for t in tensors)
        assert bool(ck['complete']) == (wanted == 30)
        if sha(p) != before or hp.read_bytes() != history_bytes:
            proof['boundary_rejections'].append({'seed': seed, 'tag': tag, 'reason': 'active update during numerical proof'})
        else:
            proof['models'].append({'seed': seed, 'tag': tag, 'sha256': before, 'epoch': wanted,
                  'step': ck['global_step'], 'complete': ck['complete'], 'earliest_best_epoch': earliest,
                  'finite_model_optimizer': True, 'RNG_schedule_source_boundary_proved': True})
        del model, ck, tensors
        gc.collect()
assert sources() == proof['source_groups']
proof['all294_current_features_after_match'] = features()
proof['stderr_bytes'] = (ROOT / 'logs/train-7313.err').stat().st_size
proof['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
with RESULT.open('x') as f:
    json.dump(proof, f, indent=2, sort_keys=True)
    f.write('\n')
print(json.dumps({'result': str(RESULT), 'sha256': sha(RESULT), 'models': proof['models'],
                  'boundary_rejections': proof['boundary_rejections'], 'stderr_bytes': proof['stderr_bytes']}))
