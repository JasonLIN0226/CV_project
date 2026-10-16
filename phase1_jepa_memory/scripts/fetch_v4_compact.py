#!/usr/bin/env python3
"""Native, checksum-verified transfer of scalar results; never follow payload links."""
import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

SSH_CONFIG = "/private/tmp/phase1-hpc-control/config"
REMOTE_PYTHON = "/scratch/xl4358/phase1_env/bin/python"
MANIFEST_PROGRAM = r'''
import hashlib,json,os,sys
from pathlib import Path
root=Path(sys.argv[1])
allowed={'.json','.csv','.md','.png','.out','.err','.txt'}
results={}
for folder,dirs,files in os.walk(root,followlinks=False):
    dirs[:]=[d for d in dirs if not (Path(folder)/d).is_symlink() and d not in {'dataset','features','observations','evaluation_only','__pycache__'}]
    for name in files:
        p=Path(folder)/name
        if p.is_symlink() or p.suffix not in allowed or p.stat().st_size>64*1024*1024: continue
        key=p.relative_to(root).as_posix()
        with p.open('rb') as f:
            h=hashlib.sha256()
            for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
        results[key]={'sha256':h.hexdigest(),'bytes':p.stat().st_size}
print(json.dumps(results,sort_keys=True))
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--remote',required=True)
    p.add_argument('--local',type=Path,required=True)
    p.add_argument('--verification-name',default='compact_transfer.json')
    a=p.parse_args()
    remote=Path(a.remote)
    if not str(remote).startswith('/scratch/xl4358/phase1_') or any(c in str(remote) for c in "'\n\r\t "):
        p.error('An explicit Phase1 output directory is required')
    ssh=['ssh','-F',SSH_CONFIG,'-o','BatchMode=yes','-o','ConnectTimeout=8',
         '-o','ServerAliveInterval=5','-o','ServerAliveCountMax=1','phase1-cloud']
    def manifest():
        r=subprocess.run(ssh+[f"{REMOTE_PYTHON} - '{remote}'"],input=MANIFEST_PROGRAM,
                         text=True,capture_output=True,check=True)
        return json.loads(r.stdout)
    before=manifest()
    if not before: raise ValueError('No compact actual files exist yet')
    a.local.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',suffix='.files') as f:
        for name in before:
            if name.startswith('/') or '..' in Path(name).parts or '\n' in name:
                raise ValueError('Invalid artifact path')
            f.write(name+'\n')
        f.flush()
        subprocess.run(['rsync','-r','--no-links','--files-from='+f.name,
                        '-e','ssh -F '+SSH_CONFIG+' -o BatchMode=yes -o ConnectTimeout=8',
                        f'phase1-cloud:{remote}/',str(a.local.resolve())+'/'],check=True)
    after=manifest()
    checks={}
    for name,record in before.items():
        local=a.local/name
        h=hashlib.sha256(local.read_bytes()).hexdigest()
        checks[name]={'remote_before':record['sha256'],'local':h,
                     'remote_after':after.get(name,{}).get('sha256'),
                     'matched':record['sha256']==h==after.get(name,{}).get('sha256')}
    result={'remote':str(remote),'local':str(a.local.resolve()),'files':checks,
            'all_matched':all(c['matched'] for c in checks.values()),
            'no_links_followed':True,'no_weights_arrays_or_datasets_transferred':True}
    (a.local/a.verification_name).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'files':len(checks),'all_matched':result['all_matched']}))
    return 0 if result['all_matched'] else 3


if __name__=='__main__':
    raise SystemExit(main())
