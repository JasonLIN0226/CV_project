"""Explicit compact evidence only; native before/local/native after SHA match."""
from pathlib import Path
import argparse,datetime,hashlib,json,shlex,subprocess
p=argparse.ArgumentParser();p.add_argument('alias');p.add_argument('destination');p.add_argument('files',nargs='+');p.add_argument('--root',default='/scratch/xl4358/phase1_far_phase1_controls');a=p.parse_args()
ROOT=a.root;D=Path(a.destination);D.mkdir(parents=True,exist_ok=True)
base=['-F','/private/tmp/phase1-hpc-control/config','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ForwardAgent=no']
assert all('..' not in Path(n).parts and not n.startswith('/') and Path(n).suffix in ['.json','.csv','.out','.err'] for n in a.files)
code="""from pathlib import Path
import json,hashlib
root=Path(%r);names=%r;out={}
for n in names:
 p=root/n;s=p.stat();assert s.st_size<1500000
 out[n]={'bytes':s.st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
print(json.dumps(out))
"""%(ROOT,a.files)
def remote():
 r=subprocess.run(['ssh',*base,a.alias,'/scratch/xl4358/phase1_env/bin/python -'],input=code,capture_output=True,text=True,check=True);return json.loads(r.stdout)
before=remote()
for n in a.files:
 dest=D/n;dest.parent.mkdir(parents=True,exist_ok=True)
 subprocess.run(['scp','-O',*base,a.alias+':'+ROOT+'/'+n,str(dest)],capture_output=True,text=True,check=True)
after=remote();assert before==after
for n,v in before.items():assert hashlib.sha256((D/n).read_bytes()).hexdigest()==v['sha256']
proof={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':ROOT,'alias':a.alias,'files':before,'native_before_local_native_after_all_match':True}
(D/'transfer_integrity.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({'files':len(before),'all_three_SHA_match':True,'destination':str(D)}))
