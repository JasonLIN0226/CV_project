"""Read-only final evidence replay on course CPU; no model import or inference."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import csv
import datetime
import hashlib
import itertools
import json
import random
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path('/scratch/xl4358/phase1_far_reference')
UP = ROOT / 'upstream'
SCI = Path('/scratch/xl4358/phase1_far_phase1')
ANA = Path('/scratch/xl4358/phase1_far_phase1_analysis')
DATA = SCI / 'output/job_7426'
RESULT = ANA / 'output/job_7435'
OUT = Path('/scratch/xl4358/phase1_far_phase1_final_evidence_7451')
sys.path.insert(0, str(ROOT / 'vendor'))
import numpy as np

def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def load(p):
    return json.loads(p.read_text())

def near(a, b):
    assert np.allclose(a, b, atol=1e-12, rtol=1e-12), (a, b)

def main():
    OUT.mkdir(exist_ok=False)
    proof = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'host': socket.gethostname(), 'verifier_source_sha256': sys.argv[1],
             'scope': 'Independent actual-data integrity, statistics and storage replay; no model forward',
             'new_science_or_GPU_allocation': False, 'source_groups': {}, 'files': {}}
    proof['sacct'] = subprocess.check_output([
        '/opt/slurm/bin/sacct', '-j', '7426,7435,7451', '-n', '-P', '-o',
        'JobID,State,ExitCode,Account,Partition,Start,End,Elapsed,AllocTRES,NodeList'], text=True)
    for job in ('7426', '7435'):
        row = next(x.split('|') for x in proof['sacct'].splitlines() if x.startswith(job+'|'))
        assert row[1:4] == ['COMPLETED', '0:0', 'csci_ga_2271-2026fa']
        assert ('gres/gpu=1' in row[8]) if job == '7426' else ('gres/gpu' not in row[8])
    proof['squeue'] = subprocess.check_output([
        '/opt/slurm/bin/squeue', '-u', 'xl4358', '-h', '-o', '%i|%j|%a|%P|%T|%b'], text=True)
    assert all(x.startswith('7451|') for x in proof['squeue'].splitlines())
    for base, lock, expected in [
        (SCI, SCI/'source_lock.json', 'e53cf22195b277ef2be12c87151632ee706f19d0eb16a41819861628fad1fcbb'),
        (UP, ROOT/'upstream_source_lock.json', 'f5dee7da4dc3eb14eacf2ab0595c1303cd13176e5a3bc913e3f19881846fe11f'),
        (ANA, ANA/'source_lock.json', '765def947f04d9f2a9a78d4d0659acb7f2764473c07f09488d7143c6d5bde7dc')]:
        assert digest(lock) == expected
        entries = load(lock)['files']
        if isinstance(entries, list): entries = {e['path']: e['sha256'] for e in entries}
        assert all(digest(base/n) == sha for n, sha in entries.items())
        proof['source_groups'][str(lock)] = {'lock_sha256': expected, 'files': len(entries), 'all_match': True}
    binding, protocol = load(SCI/'input_proof.json'), load(SCI/'protocol.json')
    assert digest(SCI/'protocol.json') == '3845db102ca99b4f1649a40bc23db4123a8227a18087b61361248de8bfe2c4d2'
    assert digest(SCI/'input_proof.json') == 'c0e98d64a10aff9248c05d17c5f7894534039452955ba12e3b09bb8b7752b4c1'
    assets = [(UP/e['path'], e['sha256']) for e in binding['files']]
    assets += [(UP/'models'/e['path'], e['sha256']) for e in binding['models']]
    certificate = Path('/scratch/xl4358/phase1_far_metadata_keys/output/job_7421/certificate.json')
    assert digest(certificate) == binding['source_certificate_sha256']
    assets += [(UP/e['path'], e['sha256']) for e in load(certificate)['metrics']]
    for p, expected in assets:
        a = p.stat(); assert digest(p) == expected; b = p.stat()
        assert (a.st_size, a.st_mtime_ns) == (b.st_size, b.st_mtime_ns)
    proof['current_asset_full_SHA'] = {'selected_inputs': 16, 'release_models': 11, 'metric_cache': len(assets)-27, 'all_match_and_stable': True}
    model = load(DATA/'executed_model_proof.json')
    assert digest(DATA/'executed_model_proof.json') == 'b1c8bf4a5e89275401876e093a6a5f0093a4aa77001775c759120324349e6deb'
    assert model['all_actual_loaded_floating_tensors_finite'] and model['threads'] == 4
    audit, summary = load(RESULT/'audit.json'), load(RESULT/'summary.json')
    assert audit['status'] == 'COMPLETE_CURRENT_SOURCE_MODEL_DATA_SCALAR_STORAGE_AUDIT'
    assert len(list(DATA.glob('*.npz'))) == len(audit['files']) == 96
    assert load(DATA/'completion.json')['completed_steps'] == 6996
    keys = [m+'_'+w for m in ('dreamsim','lpips','psnr') for w in ('full','tail')]
    values, traces, rows, retention = {}, {}, [], []
    with (RESULT/'record_summary.csv').open() as f:
        csvrows = {(int(r['clip_index']),int(r['seed']),r['policy']):r for r in csv.DictReader(f)}
    for idx, count, video in binding['selected_clips']:
        axis = None
        for seed in protocol['seeds']:
            for policy in protocol['policies']:
                stem = f'{idx}_{seed}_{policy}'; path = DATA/(stem+'.npz')
                assert digest(path) == audit['files'][path.name]['sha256']
                with np.load(path, allow_pickle=False) as z:
                    frame = z['frame_idx']
                    assert frame.shape == (count,) and np.issubdtype(frame.dtype,np.integer) and np.all(np.diff(frame)>0)
                    if axis is None: axis=frame.copy()
                    else: assert np.array_equal(axis,frame)
                    assert str(z['video_id']) == video and int(z['seed']) == seed
                    for metric in ('dreamsim','lpips','psnr'):
                        x=z[metric]; assert x.shape==(count,) and np.isfinite(x).all()
                        for window,cut in [('full',0),('tail',count-(count+3)//4)]:
                            k=metric+'_'+window; value=float(x[cut:].mean())
                            near(value,float(csvrows[idx,seed,policy][k]));values[idx,seed,policy,k]=value
                trace=load(DATA/('trace_'+stem+'.json'));traces[idx,seed,policy]=trace
                rng=random.Random(seed+idx); past=[]; erased=set(); arrivals=[trace[0]['offered']]
                observed_fraction=[]
                for number,event in enumerate(trace,1):
                    assert event['offered']==arrivals[-1] and event['arrived'] not in arrivals
                    candidates=set(past)|{event['offered']};keep=set(event['kept'])
                    assert len(keep)==len(event['kept']) and keep<=candidates
                    assert set(event['deleted'])==candidates-keep and not keep&erased and event['arrived'] not in keep
                    if policy=='Archive_reference': assert keep==candidates
                    else: assert len(keep)==min(16,number)
                    if policy=='FIFO': assert keep==set(arrivals[-16:])
                    if policy=='Reservoir':
                        if len(past)<16:past.append(event['offered'])
                        else:
                            j=rng.randrange(number)
                            if j<16:past[j]=event['offered']
                        assert set(past)==keep
                    else:past=list(keep)
                    erased.update(event['deleted']);arrivals.append(event['arrived'])
                    if event['arrived'].startswith('g:'):observed_fraction.append(sum(i.startswith('o:') for i in keep)/len(keep))
                assert len(observed_fraction)==count
                retention.append({'clip':idx,'seed':seed,'policy':policy,
                                  'observed_past_fraction_return_mean':float(np.mean(observed_fraction)),
                                  'observed_past_fraction_return_end':observed_fraction[-1]})
    for policy in protocol['policies']:
        for k in keys:
            means=[np.mean([values[idx,s,policy,k] for idx,_,_ in binding['selected_clips']]) for s in protocol['seeds']]
            target=summary['policies'][policy][k]
            near(means,target['seed_means']);near(np.mean(means),target['mean']);near(np.std(means,ddof=1),target['seed_SD'])
    signs=np.array(list(itertools.product([-1,1],repeat=8)))
    for baseline in ('FIFO','Reservoir','Archive_reference'):
        for k in keys:
            direction=-1 if k.startswith('psnr') else 1
            delta=np.array([np.mean([direction*(values[idx,s,baseline,k]-values[idx,s,'FAR_current',k]) for s in protocol['seeds']]) for idx,_,_ in binding['selected_clips']])
            rng=np.random.default_rng(0); ci=np.percentile(delta[rng.integers(0,8,(10000,8))].mean(axis=1),[2.5,97.5])
            target=summary['paired_comparisons'][baseline][k]
            near(delta.mean(),target['mean_improvement']);near(ci,target['paired_scene_bootstrap95'])
            near(np.mean((signs*delta).mean(axis=1)>=delta.mean()-1e-12),target['exact_one_sided_signflip_p'])
    proof.update(status='PASS_INDEPENDENT_FINAL_NATIVE_SOURCE_ASSET_RAW_STATISTICS_STORAGE_REPLAY',
                 records=96,target_steps=6996,source_file_count=169,
                 exact_algorithm_R_trace_replay=True,all_raw_metrics_and_CSV_match=True,
                 all_18_paired_comparisons_match=True,analysis_own4_source_files_verified=True,
                 retention_diagnostic='Post hoc ID-only descriptive audit; no new model evaluation or changed primary test')
    (OUT/'retention_diagnostic.json').write_text(json.dumps(retention,indent=2)+'\n')
    for p in list(RESULT.iterdir())+[DATA/n for n in ('completion.json','progress.json','executed_model_proof.json','first_clip_budget.json')]:
        proof['files'][str(p)]={'bytes':p.stat().st_size,'sha256':digest(p)}
    (OUT/'final_native_audit.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({'status':proof['status'],'records':96,'steps':6996,'output':str(OUT),'audit_sha256':digest(OUT/'final_native_audit.json')},indent=2))

if __name__=='__main__':main()
