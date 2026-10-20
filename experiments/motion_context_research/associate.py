"""Read-only pre-fit association check; existing training pairs only."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import hashlib,json,socket,subprocess,sys
from pathlib import Path
import numpy as np
from context import context
BASE=Path('/scratch/xl4358/phase1_motion_scene_calibration')
OUT=Path('/scratch/xl4358/phase1_motion_context_research')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as f:return {k:f[k] for k in f.files}
def distance(x):
 x=np.asarray(x,float);x=(x-x.mean(0))/np.maximum(x.std(0),1e-6);d=np.maximum(np.sum(x*x,1)[:,None]+np.sum(x*x,1)[None,:]-2*x@x.T,0);np.fill_diagonal(d,np.inf);return d
def main(job):
 n=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',str(job)],text=True);v=dict(x.split('=',1) for x in n.split() if '=' in x);assert v['JobState']=='RUNNING' and v['Account']=='csci_ga_2271-2026fa' and 'gres/gpu' not in v['AllocTRES'] and v['NodeList']==socket.gethostname().split('.')[0]
 OUT.mkdir(exist_ok=False);prep=json.loads((BASE/'preparation.json').read_text());assert sha(BASE/'train_pack.npz')==prep['train_pack_sha256'];d=load(BASE/'train_pack.npz');r=load(BASE/'gpu_output/training_residuals.npz');m=json.loads((BASE/'gpu_output/manifest.json').read_text());assert sha(BASE/'gpu_output/training_residuals.npz')==m['files']['training_residuals.npz'];cells=[]
 # Sibling averages are retrospective descriptors only, never deployment inputs.
 geom=d['geometry'].reshape(512,2,2,2).mean(1)
 for hi,h in enumerate((3,6)):
  current=d['X'][::2,hi,:128];actions=d['X'][::2,hi,128:];cc=context(d['z'],d['past_actions'],d['actions'],d['projected_current'],d['A'],d['b'],h);assert np.array_equal(cc[::2],cc[1::2]),'Causal scene-centre must be exactly sibling-invariant'
  contexts={'current':current,'current_actions':d['X'][::2,hi],'paired_causal_geometry':geom[:,hi],'causal_scene_centre':cc[::2,:2]};neighbors={k:np.argsort(distance(x),axis=1,kind='stable')[:,:10] for k,x in contexts.items()}
  for si,seed in enumerate((17,29,43)):
   y=r['residuals'][si,:,hi]@r['N'];dist=np.maximum(np.sum(y*y,1)[:,None]+np.sum(y*y,1)[None,:]-2*y@y.T,0);den=(dist.sum()/512/511);cell={'seed':seed,'horizon':h,'pairs':512,'reference_pair_distance':float(den),'contexts':{}}
   for name,ix in neighbors.items():
    value=float(dist[np.arange(512)[:,None],ix].mean());perms=[];rng=np.random.default_rng(0)
    for z in range(1000):
     p=rng.permutation(512);perms.append(float(dist[p[:,None],p[ix]].mean()))
    cell['contexts'][name]={'neighbor_distance':value,'relative_reduction_vs_random':float(1-value/den),'permutation_p':float((1+sum(x<=value for x in perms))/1001)}
   cells.append(cell)
 result={'stage':'READ_ONLY_TRAIN_CONTEXT_ASSOCIATION','native':n,'source_sha256':sha(__file__),'cells':cells,'GPU':0,'new_fitting':False,'new_model_forward':False,'validation_or_test_access':False,'fresh9137':False,'gates_changed':False,'paired_geometry_not_a_deployment_input':True,'interpretation':'Association supports or rejects omitted-context hypothesis; not a candidate or a passing gate.'}
 (OUT/'association.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main(int(sys.argv[1]))
