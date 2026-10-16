"""Independent audit and analysis; never launches or repeats GPU science."""
from pathlib import Path
import csv
import datetime
import hashlib
import itertools
import json
import os
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SCIENCE = Path('/scratch/xl4358/phase1_far_phase1')
ROOT = Path('/scratch/xl4358/phase1_far_reference')
UP = ROOT / 'upstream'
sys.path.insert(0, str(ROOT / 'vendor'))
import numpy as np


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda: f.read(8 << 20), b''):
            h.update(x)
    return h.hexdigest()


def window(values, tail):
    if tail:
        # Integer frame rule: the final ceil(n/4) predictions, including end.
        return float(values[np.arange(1, len(values)+1)/len(values) > .75].mean())
    return float(values.mean())


def paired_stats(delta):
    delta = np.asarray(delta, dtype=float)
    assert delta.shape == (8,) and np.isfinite(delta).all()
    rng = np.random.default_rng(0)
    means = delta[rng.integers(0, 8, size=(10000, 8))].mean(1)
    signs = np.asarray(list(itertools.product([-1, 1], repeat=8)))
    exact = (signs * delta).mean(1)
    return {'mean_improvement': float(delta.mean()),
            'paired_scene_bootstrap95': np.percentile(means, [2.5, 97.5]).tolist(),
            'exact_one_sided_signflip_p': float((exact >= delta.mean()-1e-12).mean()),
            'independent_units': 8, 'seeds_averaged_within_scene': True}


def audit_trace(trace, policy):
    assert trace
    seen, deleted, previous = set(), set(), set()
    last_current = trace[0]['offered']
    seen.add(last_current)
    arrivals = [last_current]
    for row in trace:
        assert row['offered'] == last_current
        assert row['arrived'] not in seen
        candidates = previous | {row['offered']}
        kept, removed = set(row['kept']), set(row['deleted'])
        assert len(row['kept']) == len(kept)
        assert kept <= candidates and removed == candidates-kept
        assert not kept & deleted
        assert row['arrived'] not in kept
        if policy != 'Archive_reference':
            assert len(kept) == min(16, len(seen))
        else:
            assert kept == candidates and not removed
        if policy == 'FIFO':
            assert kept == set(arrivals[-16:])
        deleted.update(removed)
        seen.add(row['arrived'])
        arrivals.append(row['arrived'])
        previous, last_current = kept, row['arrived']
    observed = [int(x[2:]) for x in arrivals if x.startswith('o:')]
    generated = [int(x[2:]) for x in arrivals if x.startswith('g:')]
    assert all(x.startswith(('o:', 'g:')) for x in arrivals)
    assert observed == sorted(set(observed)) and generated == sorted(set(generated))
    assert len(generated) > 0
    return {'updates': len(trace), 'observed_arrivals': len(observed),
            'generated_arrivals': len(generated), 'evicted_IDs': len(deleted),
            'causal_transition_and_no_reentry': True, 'logical_capacity_after_each_update': True,
            'evaluator_tensors_not_claimed_to_be_physically_deleted': True}


def main():
    out = HERE / 'output' / ('job_' + os.environ['SLURM_JOB_ID'])
    out.mkdir(parents=True, exist_ok=False)
    def save(n, d):
        (out/n).write_text(json.dumps(d, indent=2)+'\n')
    spec = json.loads((HERE/'analysis_protocol.json').read_text())
    protocol = json.loads((SCIENCE/'protocol.json').read_text())
    proof = json.loads((SCIENCE/'input_proof.json').read_text())
    data = SCIENCE / 'output/job_7426'
    state = subprocess.check_output(['/opt/slurm/bin/sacct', '-j', '7426', '-n', '-P',
                                    '-o', 'JobID,State,ExitCode,Account,Start,End,Elapsed'], text=True)
    audit = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'sacct': state,
             'analysis_protocol_sha256': sha(HERE/'analysis_protocol.json'),
             'science_protocol_sha256': sha(SCIENCE/'protocol.json'),
             'input_proof_sha256': sha(SCIENCE/'input_proof.json'), 'files': {}, 'sources': {},
             'no_model_forward_or_GPU_allocation': True}
    assert audit['science_protocol_sha256'] == spec['science_protocol_sha256']
    assert audit['input_proof_sha256'] == spec['input_proof_sha256']
    for lock, base, expected in [
        (SCIENCE/'source_lock.json', SCIENCE, spec['science_source_lock_sha256']),
        (ROOT/'upstream_source_lock.json', UP, spec['upstream_source_lock_sha256'])]:
        assert sha(lock) == expected
        entries = json.loads(lock.read_text())['files']
        if isinstance(entries, list): entries = {x['path']: x['sha256'] for x in entries}
        assert all(sha(base/n) == h for n, h in entries.items())
        audit['sources'][str(lock)] = {'sha256': expected, 'files': len(entries), 'all_match': True}
    for e in proof['files']:
        before = (UP/e['path']).stat()
        assert sha(UP/e['path']) == e['sha256']
        after = (UP/e['path']).stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    for e in proof['models']:
        assert sha(UP/'models'/e['path']) == e['sha256']
    audit['selected16_files_and11_models_current_SHA_match'] = True
    certificate = Path('/scratch/xl4358/phase1_far_metadata_keys/output/job_7421/certificate.json')
    assert sha(certificate) == proof['source_certificate_sha256']
    cert = json.loads(certificate.read_text())
    for e in cert['metrics']:
        assert sha(UP/e['path']) == e['sha256'], e['path']
    audit['known_metric_cache_files_current_SHA_match'] = len(cert['metrics'])
    for n in ['progress.json', 'failure.json', 'completion.json', 'first_clip_budget.json']:
        p = data/n
        if p.exists(): audit[n] = json.loads(p.read_text())
    for p in sorted(data.glob('*.npz')):
        audit['files'][p.name] = {'bytes': p.stat().st_size, 'sha256': sha(p)}
    complete = any(line.split('|')[:3] == ['7426', 'COMPLETED', '0:0'] for line in state.splitlines())
    audit['actual_complete0_0'] = complete
    if not complete or len(audit['files']) != 96 or not (data/'completion.json').exists():
        audit['status'] = 'PARTIAL_OR_FAILED_NO_POPULATION_AGGREGATION'
        save('audit.json', audit)
        return
    model = json.loads((data/'executed_model_proof.json').read_text())
    assert model['all_actual_loaded_floating_tensors_finite'] and model['threads'] == 4
    assert model['input_proof_sha256'] == audit['input_proof_sha256']
    audit['executed_model_proof_sha256'] = sha(data/'executed_model_proof.json')
    rows, scores, axes = [], {}, {}
    trace_audits = {}
    for idx, steps, video in proof['selected_clips']:
        for seed in protocol['seeds']:
            for policy in protocol['policies']:
                n = f'{idx}_{seed}_{policy}'
                with np.load(data/(n+'.npz'), allow_pickle=False) as record:
                    assert str(record['video_id']) == video and int(record['seed']) == seed
                    frame = record['frame_idx']
                    assert frame.shape == (steps,) and np.issubdtype(frame.dtype, np.integer)
                    assert (np.diff(frame) > 0).all()
                    if idx not in axes: axes[idx] = frame.copy()
                    else: assert np.array_equal(frame, axes[idx]), 'Paired target axis mismatch'
                    row = {'clip_index': idx, 'scene': video.split('/')[4],
                           'route': video.split('/')[3], 'seed': seed, 'policy': policy, 'steps': steps}
                    for metric in ['dreamsim', 'lpips', 'psnr']:
                        x = record[metric]
                        assert x.shape == (steps,) and np.isfinite(x).all()
                        for tail in [False, True]:
                            key = metric + ('_tail' if tail else '_full')
                            row[key] = scores[idx, seed, policy, key] = window(x, tail)
                    rows.append(row)
                trace_audits[n] = audit_trace(json.loads((data/('trace_'+n+'.json')).read_text()), policy)
                assert trace_audits[n]['generated_arrivals'] == steps
    assert len(rows) == 96 and len(axes) == 8
    audit['all96_metric_records_finite_and_exact_paired_axes'] = True
    audit['all96_storage_trace_transitions_pass'] = True
    save('storage_audit.json', trace_audits)
    with (out/'record_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {'status': 'COMPLETE_SMALL_PHASE1_PILOT_NOT_FULL_PAPER_OR_ORIGINAL_GO',
               'independent_scenes': 8, 'records': 96, 'policies': {}, 'paired_comparisons': {},
               'interpretation_limits': ['same pretrained FAR model; no I-JEPA claim',
                   '8-scene pilot, not full LoopNav test population', 'Archive reference has greater storage',
                   'seeds are not independent population units', 'no original Phase1 GO threshold changed']}
    keys = ['dreamsim_tail', 'dreamsim_full', 'lpips_tail', 'lpips_full', 'psnr_tail', 'psnr_full']
    for policy in protocol['policies']:
        summary['policies'][policy] = {}
        for key in keys:
            seed_means = [float(np.mean([scores[idx, seed, policy, key] for idx, _, _ in proof['selected_clips']]))
                          for seed in protocol['seeds']]
            summary['policies'][policy][key] = {'mean': float(np.mean(seed_means)),
                'seed_SD': float(np.std(seed_means, ddof=1)), 'seed_means': seed_means}
    for baseline in ['FIFO', 'Reservoir', 'Archive_reference']:
        summary['paired_comparisons'][baseline] = {}
        for key in keys:
            sign = -1 if key.startswith('psnr') else 1
            delta = [np.mean([sign*(scores[idx, seed, baseline, key]-scores[idx, seed, 'FAR_current', key])
                             for seed in protocol['seeds']]) for idx, _, _ in proof['selected_clips']]
            summary['paired_comparisons'][baseline][key] = paired_stats(delta)
    summary['primary_two_comparisons_positive_CI'] = all(
        summary['paired_comparisons'][p]['dreamsim_tail']['paired_scene_bootstrap95'][0] > 0
        for p in ['FIFO', 'Reservoir'])
    summary['route_means'] = {route: {policy: float(np.mean([r['dreamsim_tail'] for r in rows
        if r['route'] == route and r['policy'] == policy])) for policy in protocol['policies']}
        for route in ['ABA', 'ABCA']}
    audit['status'] = 'COMPLETE_CURRENT_SOURCE_MODEL_DATA_SCALAR_STORAGE_AUDIT'
    save('summary.json', summary); save('audit.json', audit)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        q = HERE / 'output' / ('job_' + os.environ.get('SLURM_JOB_ID', 'unknown'))
        q.mkdir(parents=True, exist_ok=True)
        (q/'failure.json').write_text(json.dumps({'type': type(error).__name__, 'error': str(error), 'no_GPU_or_science_retry': True}, indent=2)+'\n')
        raise
