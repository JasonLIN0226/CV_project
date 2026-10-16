"""Dependency-only recovery; reuse the already verified official releases."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

OWN = Path(__file__).resolve().parent
ROOT = Path('/scratch/xl4358/phase1_far_reference')
OUT = OWN / 'output' / ('job_' + os.environ.get('SLURM_JOB_ID', 'UNSET'))
START = time.monotonic()


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(16 * 1024**2), b''): h.update(b)
    return h.hexdigest()


def save(name, value):
    p = OUT / name; t = p.with_suffix('.tmp')
    t.write_text(json.dumps(value, indent=2) + '\n'); t.replace(p)


def source():
    for n, h in json.loads((OWN/'recovery_source_lock.json').read_text())['files'].items():
        if sha(OWN/n) != h: raise RuntimeError('recovery source differs: '+n)
    if sha(ROOT/'evaluation_source_lock.json') != 'a941ab106917c3efdb7516806b1f1a21faa870dc47d5e229e2fec10d512df0f9':
        raise RuntimeError('original evaluation source lock differs')
    for lock, base in [('evaluation_source_lock.json', ROOT), ('upstream_source_lock.json', ROOT/'upstream')]:
        for n, h in json.loads((ROOT/lock).read_text())['files'].items():
            if sha(base/n) != h: raise RuntimeError('original frozen source differs: '+n)


def require_failure_boundary(rows):
    d = {r.split('|')[0]:r.split('|') for r in rows.strip().splitlines()}
    if d['7385'][1:4] != ['FAILED','1:0','csci_ga_2271-2026fa']:
        raise RuntimeError('original native preparation failure differs')
    if not d['7386'][1].startswith('CANCELLED') or d['7386'][3] != 'csci_ga_2271-2026fa':
        raise RuntimeError('invalid dependency cancellation not proved')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if os.environ.get('SLURM_JOB_ACCOUNT') != 'csci_ga_2271-2026fa': raise RuntimeError('course required')
    if Path(sys.executable).resolve() != Path('/scratch/xl4358/phase1_env/bin/python').resolve(): raise RuntimeError('approved Python required')
    source()
    old = subprocess.check_output(['/opt/slurm/bin/sacct','-j','7385,7386','-X','-n','-P','-o','JobID,State,ExitCode,Account'],text=True)
    require_failure_boundary(old)
    for n in ['evaluation_binding.json','evaluation_submission_attempt.json','evaluation_submission.json']:
        if (ROOT/n).exists(): raise RuntimeError('unknown/existing evaluation marker: '+n)
    progress = json.loads((ROOT/'preparation/job_7385/progress.json').read_text())
    if (progress['model_files'],progress['test_shards']) != (11,5): raise RuntimeError('public releases incomplete')
    released = []
    datasets = json.loads((ROOT/'datasets.json').read_text())
    models = json.loads((ROOT/'bundles.json').read_text())['bundles']
    groups = [(ROOT/'public_release_downloads', [e for e in datasets['shards'] if e['tier']=='test']),
              (ROOT/'upstream/models', [e for e in models if e.get('corpus')=='loopnav' or e['path']=='oasis_500m_vit_vae.pth'])]
    for base, entries in groups:
        for e in entries:
            p=base/e['path']
            if p.stat().st_size != e['size_bytes'] or sha(p) != e['sha256']: raise RuntimeError('existing release mismatch: '+str(p))
            released.append({'path':str(p),'sha256':e['sha256'],'bytes':p.stat().st_size})
    index=json.loads((ROOT/'upstream/results/indices/loopnav/loopnav_latent_test.json').read_text())['sample_ids']
    if len(index)!=3840 or len(set(index))!=3840: raise RuntimeError('official test split differs')
    for n in index:
        p=ROOT/'upstream'/n
        if not p.is_file() or not p.with_suffix('.json').is_file(): raise RuntimeError('missing extracted test pair')
    vendor=ROOT/'vendor'
    if vendor.exists(): raise RuntimeError('prior dependency target exists; inspect rather than overwrite')
    save('progress.json',{'status':'VERIFIED_EXISTING_RELEASES_INSTALLING_ORIGINAL_PINS','test_pairs':3840,'released_files':released})
    req=ROOT/'upstream/requirements.txt'
    with (OUT/'dependency_install.log').open('w') as f:
        subprocess.run([sys.executable,'-m','pip','install','--target',str(vendor),'--prefer-binary','--no-cache-dir','--report',str(OUT/'pip_report.json'),'-r',str(req)],stdout=f,stderr=subprocess.STDOUT,check=True)
    pip=json.loads((OUT/'pip_report.json').read_text())
    dream=[e for e in pip['install'] if e['metadata']['name'].lower()=='dreamsim']
    if len(dream)!=1 or dream[0]['metadata']['version']!='0.2.1' or dream[0]['download_info']['archive_info']['hashes']['sha256']!='36c655ee1bb5dbbf1730f03a59ac0d0180922f2151dfbdfb4f058b1756903d14':
        raise RuntimeError('official DreamSim source release differs')
    distributions=list(importlib.metadata.distributions(path=[str(vendor)]))
    versions={d.metadata['Name'].lower().replace('_','-'):d.version for d in distributions}
    for line in req.read_text().splitlines():
        if '==' not in line or line.startswith('#'):continue
        name,version=line.strip().split('==')
        if versions.get(name.lower().replace('_','-'))!=version:raise RuntimeError('original dependency pin differs: '+name)
    source()
    package_files=[{'path':str(p.relative_to(vendor)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(vendor.rglob('*')) if p.is_file()]
    save('vendor_files_proof.json',package_files)
    report={'status':'PUBLIC_RELEASE_FILES_AND_DEPENDENCIES_PREPARED','job_id':int(os.environ['SLURM_JOB_ID']),'original_failed_job':7385,'original_cancelled_metadata_job':7386,'native_failure_boundary':old,'test_clips_declared':3840,'existing_release_files_verified':released,'no_redownload_or_reextract':True,'vendor':str(vendor),'versions':versions,'requirements_sha256':sha(req),'pip_report_sha256':sha(OUT/'pip_report.json'),'vendor_proof_sha256':sha(OUT/'vendor_files_proof.json'),'recovery_source_sha256':sha(OWN/'recovery_source_lock.json'),'model_inference_or_training':False,'data_numeric_audit_and_GPU_compatibility_pending':True,'elapsed_seconds':time.monotonic()-START}
    save('preparation_report.json',report); save('progress.json',report); print(json.dumps(report),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as e:
        save('failure.json',{'status':'DEPENDENCY_RECOVERY_INCOMPLETE','type':type(e).__name__,'error':str(e),'elapsed_seconds':time.monotonic()-START,'model_inference_or_training':False})
        raise
