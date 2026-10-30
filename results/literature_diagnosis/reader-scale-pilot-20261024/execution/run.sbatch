#!/bin/bash
set -euo pipefail
umask 077
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
/scratch/xy3257/cv-phase2-20261020/venv/bin/python /scratch/xy3257/cv-phase2-20261020/reader-scale-probe-20261024/bundle/experiments/phase2/run_reader_scale_probe.py run --inputs /scratch/xy3257/cv-phase2-20261020/reader-scale-probe-20261024/bundle/inputs --cache-dir /scratch/xy3257/cv-phase2-20261020/cache/ijepa --output /scratch/xy3257/cv-phase2-20261020/reader-scale-probe-20261024/run --steps 512 --validate-every 64 --model-seed 17
/scratch/xy3257/cv-phase2-20261020/venv/bin/python - <<'PACK_READER_PILOT'

import hashlib, json, os, zipfile
from pathlib import Path
work = Path.cwd()
run = work / 'run'
completion = json.loads((run / 'completion.json').read_text())
assert completion['status'] == 'completed'
assert str(completion['hardware']['slurm_job_id']) == os.environ['SLURM_JOB_ID']
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
packed, omitted = {}, {}
destination = work / 'reader-scale-results.zip'
with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
    for p in sorted(run.rglob('*')):
        if not p.is_file(): continue
        relative=str(p.relative_to(run))
        expected=completion['files'].get(relative)
        actual=sha(p)
        if expected is not None and expected != actual: raise RuntimeError('Output changed: '+relative)
        heavy=relative.startswith('data/features/') or relative in ('data/unique_rgb.npz','data/unique_features.npz')
        (omitted if heavy else packed)[relative]={'sha256':actual,'bytes':p.stat().st_size}
        if not heavy: z.write(p, 'run/'+relative)
    for name in ('submission_receipt.json','run.sbatch','submission_driver.py'):
        z.write(work/name, 'execution/'+name)
    z.writestr('archive_manifest.json',json.dumps({'packed':packed,'remote_cache_only':omitted},indent=2))
receipt={'status':'completed','job_id':os.environ['SLURM_JOB_ID'],'zip_sha256':sha(destination),
         'zip_bytes':destination.stat().st_size,'remote_cache_retained':True,
         'actual_gpu_uuid':completion['hardware']['actual_gpu_uuid']}
(work/'zip_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)

PACK_READER_PILOT
