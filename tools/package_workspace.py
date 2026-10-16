#!/usr/bin/env python3
"""Publish local research source and result records without caches or credentials."""
from pathlib import Path
from collections import Counter
import argparse, hashlib, json, re, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT=Path(__file__).resolve().parents[1]
SKIP_DIRS={'.git','__pycache__','.pytest_cache','.venv','node_modules','.matplotlib','.fontconfig','simple_report_fonts'}
SKIP_SUFFIX={'.pyc','.pyo','.npz','.npy','.pt','.pth','.ckpt','.safetensors','.whl','.zip','.otf','.ttf'}
UPSTREAM={'far_reproduction/upstream','paper_reproduction/SPACE'}
PATTERNS={
 'private_key':re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----'),
 'github_token':re.compile(rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})'),
 'huggingface_token':re.compile(rb'\bhf_[A-Za-z0-9]{25,}'),
 'openai_key':re.compile(rb'\bsk-(?:proj-)?[A-Za-z0-9_-]{30,}'),
}

def sha(data):return hashlib.sha256(data).hexdigest()
def exclusion(path):
    s=path.as_posix()
    if any(s==v or s.startswith(v+'/') for v in UPSTREAM):return 'Pinned upstream checkout; fetch from its original repository'
    if any(v in SKIP_DIRS for v in path.parts):return 'Local cache, runtime, fonts, or Git internals'
    if path.name in {'.DS_Store','.gitignore'}:return 'Local filesystem or ignore settings'
    if path.suffix.lower() in SKIP_SUFFIX:return 'Generated arrays, weights, runtime binaries, fonts, or rebuildable upload archive'
    if re.fullmatch(r'(?:native-)?page[-_]\d+\.png',path.name):return 'Rendered per-page QA image; report and scientific figures are retained'
    if path.name.endswith('.cache-9'):return 'Font cache'
    if path.name.startswith('.env') or path.suffix.lower() in {'.pem','.key'}:return 'Private local configuration'
    return None

def main(source, resume=False):
    source=Path(source).resolve();dest=ROOT/'phase1_jepa_memory'
    assert source!=dest and source.name=='phase1_jepa_memory'
    included=[];excluded=[];reasons=Counter();eligible=[]
    files=sorted(f for f in source.rglob('*') if f.is_file() and '.git' not in f.relative_to(source).parts)
    for f in files:
        rel=f.relative_to(source);reason=exclusion(rel)
        if f.is_symlink():reason='Symlink outside the portable source snapshot'
        if reason:
            excluded.append({'path':rel.as_posix(),'reason':reason,'bytes':f.stat().st_size});reasons[reason]+=1;continue
        eligible.append((f,rel))
    out=ROOT/'publication';out.mkdir(exist_ok=True);dest.mkdir(exist_ok=True)
    filelist=out/'included_paths.txt'
    filelist.write_text(''.join(rel.as_posix()+'\n' for _,rel in eligible))
    print(f'Copying {len(eligible)} selected source/result files',flush=True)
    def copy_one(item):
        f,rel=item;target=dest/rel
        if resume and target.exists() and target.stat().st_size==f.stat().st_size:return None
        target.parent.mkdir(parents=True,exist_ok=True)
        try:
            data=subprocess.check_output(['cat',str(f)],timeout=30,stderr=subprocess.PIPE)
            assert len(data)==f.stat().st_size
            target.write_bytes(data)
            if f.stat().st_mode & 0o111:target.chmod(target.stat().st_mode|0o111)
            return None
        except (OSError,subprocess.SubprocessError,AssertionError) as exc:
            return {'path':rel.as_posix(),'error_type':type(exc).__name__}
    errors=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for number,result in enumerate(as_completed([pool.submit(copy_one,x) for x in eligible]),1):
            if result.result():errors.append(result.result())
            if number%500==0:print(f'Copy checked {number} / {len(eligible)}',flush=True)
    if errors:
        (out/'unreadable_local_files.json').write_text(json.dumps(errors,indent=2)+'\n')
        raise RuntimeError(f'{len(errors)} local files need recovery; see publication/unreadable_local_files.json')
    missing=out/'unreadable_local_files.json'
    if missing.exists():missing.unlink()
    def hash_one(item):
        f,rel=item
        target=dest/rel;data=target.read_bytes()
        assert len(data)==f.stat().st_size
        if f.suffix.lower() not in {'.png','.pdf','.jpg','.jpeg','.webp','.docx'}:
            for label,pattern in PATTERNS.items():
                if pattern.search(data):raise RuntimeError(f'Possible {label} in {rel}; no content printed')
            if b'sig=' in data and b'oaiusercontent.com' in data:raise RuntimeError(f'Expiring signed download link in {rel}')
        return {'path':'phase1_jepa_memory/'+rel.as_posix(),'bytes':len(data),'sha256':sha(data)}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for result in as_completed([pool.submit(hash_one,x) for x in eligible]):
            included.append(result.result())
            if len(included)%300==0:print(f'Packaged {len(included)} / {len(eligible)} files',flush=True)
    included.sort(key=lambda x:x['path'])
    (dest/'.gitignore').write_text('__pycache__/\n*.py[cod]\n.pytest_cache/\n.venv/\n.DS_Store\n*.egg-info/\n')
    manifest={'source_snapshot':'phase1_jepa_memory','snapshot_date':'2026-10-16','copied_files':len(included),'copied_bytes':sum(f['bytes'] for f in included),'byte_identical_copies':True,'new_scientific_execution':False,'files':included}
    (out/'source_and_results_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'excluded_artifacts.json').write_text(json.dumps({'counts_by_reason':dict(reasons),'excluded':excluded},indent=2)+'\n')
    print(json.dumps({'files':len(included),'MB':round(manifest['copied_bytes']/1e6,2),'excluded':len(excluded),'reasons':dict(reasons)}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True)
    ap.add_argument('--resume',action='store_true',help='Continue an interrupted export into its existing destination')
    args=ap.parse_args();main(args.source,args.resume)
