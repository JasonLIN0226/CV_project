import csv,hashlib,json,socket,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
AGE=Path('/scratch/xl4358/phase1_age_control_a1')
HV=Path('/scratch/xl4358/phase1_motion_history_reference_validation/output/job_8449')
GV=Path('/scratch/xl4358/phase1_motion_geometry_validation/output/job_8336')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def source_check():
 assert all(sha(ROOT/k)==v for k,v in read(ROOT/'source_lock.json')['files'].items())
def native(job,gpu):
 s=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',str(job)],text=True);n=dict(v.split('=',1) for v in s.split() if '=' in v)
 assert n['JobState']=='RUNNING' and n['Account']=='csci_ga_2271-2026fa' and n['NodeList']==socket.gethostname().split('.')[0]
 assert n['Partition']==('c12m85-a100-1' if gpu else 'n2c48m24') and ('gres/gpu' in n['AllocTRES'])==gpu
 return s,n
def x_features(projected_current,actions):
 p=np.asarray(projected_current);a=np.asarray(actions)
 return np.concatenate([p,a[:,0],a.sum(1),a[:,-1]],axis=1).astype(np.float64)
def apply_numpy(X,f):
 X=np.asarray(X,dtype=np.float64);Z=np.concatenate([(X-f['mean'])/f['scale'],np.ones((len(X),1))],1)
 return (Z@f['coef'])@f['N'].T
