"""Explicit SCP-O retrieval with native-before/local/native-after SHA checks.

Use only a currently authenticated RUNNING allocation, never an Ended alias.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--alias',required=True);p.add_argument('--route-job',type=int,required=True)
    p.add_argument('--remote-root',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('files',nargs='+');a=p.parse_args()
    for f in a.files:
        path=PurePosixPath(f)
        if path.is_absolute() or '..' in path.parts or path.suffix not in ('.json','.csv','.out','.err'):
            raise ValueError('Explicit relative compact file only: '+f)
    cfg='/private/tmp/phase1-hpc-control/config'
    ssh=['ssh','-F',cfg,'-o','BatchMode=yes',a.alias]
    code='''import pathlib,json,hashlib,subprocess
root=pathlib.Path(REMOTE)
r=subprocess.run(['/opt/slurm/bin/scontrol','show','job','-o',str(JOB)],capture_output=True,text=True,check=True,timeout=15)
f=dict(x.split('=',1) for x in r.stdout.split() if '=' in x)
assert f['JobState']=='RUNNING' and f['Account']=='csci_ga_2271-2026fa'
x={}
for name in FILES:
 p=root/name
 assert p.stat().st_size<=1500000,name
 x[name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
print(json.dumps(x))
'''.replace('REMOTE',repr(a.remote_root)).replace('JOB',repr(a.route_job)).replace('FILES',repr(a.files))
    command=shlex.join(['/scratch/xl4358/phase1_env/bin/python','-c',code])
    def native():
        return json.loads(subprocess.run(ssh+[command],capture_output=True,text=True,check=True,timeout=30).stdout)
    before=native();a.output.mkdir(parents=True,exist_ok=True)
    for name in a.files:
        dest=a.output/name;dest.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['scp','-O','-F',cfg,'-o','BatchMode=yes',a.alias+':'+a.remote_root+'/'+name,str(dest)],check=True,timeout=30)
    local={name:{'sha256':hashlib.sha256((a.output/name).read_bytes()).hexdigest(),'bytes':(a.output/name).stat().st_size} for name in a.files}
    after=native()
    integrity={'native_before':before,'local':local,'native_after':after,'all_match':before==local==after}
    (a.output/'transfer_integrity.json').write_text(json.dumps(integrity,indent=2)+'\n')
    assert integrity['all_match'],'Transfer changed; retain partial proof, no silent retry'
    print(json.dumps({'triple_SHA_match':True,'files':len(a.files)}))


if __name__=='__main__':main()
