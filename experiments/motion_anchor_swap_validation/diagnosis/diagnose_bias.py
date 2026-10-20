"""Descriptive frozen-cache error decomposition; never modifies a prediction or fits an adapter."""
import csv,hashlib,json,os,socket,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
V=Path('/scratch/xl4358/phase1_motion_history_reference_validation/output/job_8449')
REF=Path('/scratch/xl4358/phase1_v3_dev/motion/reference')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def main(job):
 native=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',str(job)],text=True);n=dict(x.split('=',1) for x in native.split() if '=' in x);assert n['JobState']=='RUNNING' and n['Account']=='csci_ga_2271-2026fa' and n['Partition']=='n2c48m24' and n['NodeList']==socket.gethostname().split('.')[0] and 'gres/gpu' not in n['AllocTRES']
 assert all(sha(ROOT/k)==v for k,v in read(ROOT/'source_lock.json')['files'].items())
 with (ROOT/'execution_attempt.json').open('x') as f:json.dump(dict(job=job,allocation=native,no_retry=True),f)
 assert sha(V/'summary.json')=='3a221a8dec7eb371efddbbca2fa1777c7d29f247606abeff011b116b2c77870c' and sha(V/'vectors.npz')==read(V/'summary.json')['vectors_sha256'];assert read(V/'independent_audit.json')['status']=='PASS_INDEPENDENT_QR_AND_SCALAR_AUDIT'
 with np.load(V/'vectors.npz',allow_pickle=False) as f:z=f['candidate'];target=f['target']
 with (Path('/scratch/xl4358/phase1_motion_geometry_validation/output/job_8336')/'validation_geometry.csv').open() as f:rows=list(csv.DictReader(f))
 assert z.shape==target.shape==(1152,5120)
 with np.load(REF/'predictor/motion/basis.npz',allow_pickle=False) as f:B=f['output_basis'].astype(float)
 with np.load(REF/'probes/motion_state_decoder.npz',allow_pickle=False) as f:A=f['weight'][:,1:]/f['std'][:,None]
 Q=np.linalg.qr(B)[0];F=Q.T@A;U=np.linalg.qr(F)[0];assets={str(p):sha(p) for p in (V/'summary.json',V/'vectors.npz',REF/'predictor/motion/basis.npz',REF/'probes/motion_state_decoder.npz')};results=[];maxidentity=0.
 for seed in (17,29,43,'mean'):
  for h in (3,6):
   pairs=sorted({r['pair'] for r in rows if int(r['horizon'])==h});assert len(pairs)==96;E=[]
   for pair in pairs:
    siblings=sorted({r['episode'] for r in rows if r['pair']==pair});assert len(siblings)==2
    ee=[]
    for sibling in siblings:
     ix=[i for i,r in enumerate(rows) if r['episode']==sibling and int(r['horizon'])==h and (seed=='mean' or int(r['seed'])==seed)];assert len(ix)==(3 if seed=='mean' else 1)
     # Mean-seed outputs are not a deployed ensemble: summarize seed-wise quadratic energies below instead.
     ee.append(np.stack([z[i]-target[i] for i in ix]))
    E.append(np.stack(ee))
   E=np.stack(E);assert E.shape==(96,2,3 if seed=='mean' else 1,5120)
   common=(E[:,0]+E[:,1])/2;contrast=(E[:,0]-E[:,1])/2
   total=float(np.mean(E**2));ce=float(np.mean(common**2));de=float(np.mean(contrast**2));identity=abs(total-ce-de);maxidentity=max(maxidentity,identity);assert identity<1e-12
   mu=common.mean(0);centered=common-mu;bias=float(np.mean(mu**2));variance=float(np.mean(centered**2));assert abs(ce-bias-variance)<1e-12
   cq=common@Q;null=cq-(cq@U)@U.T;nullmu=null.mean(0);nullenergy=float(np.mean(np.sum(null**2,-1))/5120);nullbias=float(np.mean(np.sum(nullmu**2,-1))/5120)
   draw=np.random.default_rng(0).permutation(96);m1=common[draw[:48]].mean(0);m2=common[draw[48:]].mean(0);dot=float(np.mean(m1*m2));cos=float(np.sum(m1*m2)/np.sqrt(np.sum(m1*m1)*np.sum(m2*m2)))
   results.append(dict(seed=seed,horizon=h,pairs=96,full_feature_mse=total,common_energy=ce,contrast_energy=de,common_fraction=ce/total,descriptive_mean_bias_energy=bias,descriptive_mean_bias_fraction=bias/total,common_centered_variance=variance,common_basis_null_energy=nullenergy,common_basis_null_mean_energy=nullbias,common_basis_null_mean_fraction=nullbias/total,cross_half_mean_dot=dot,cross_half_mean_cosine=cos,cross_half_dot_fraction=dot/total,interpretation='Descriptive repeatability, not an adapter or a new passing criterion; mean biases are optimistic in-sample summaries.'))
 # Independent reconstruction using direct sibling sum/difference quadratic form, no primary common/contrast arrays.
 checks=[]
 for cell in results:
  seed,h=cell['seed'],cell['horizon'];seeds=(17,29,43) if seed=='mean' else (seed,);vals=[]
  for pair in sorted({r['pair'] for r in rows if int(r['horizon'])==h}):
   for ss in seeds:
    ix=[i for i,r in enumerate(rows) if r['pair']==pair and int(r['seed'])==ss and int(r['horizon'])==h];assert len(ix)==2;a,b=z[ix]-target[ix];v=(np.dot(a,a)+np.dot(b,b))/(2*5120);c=(np.dot(a,a)+np.dot(b,b)+2*np.dot(a,b))/(4*5120);d=(np.dot(a,a)+np.dot(b,b)-2*np.dot(a,b))/(4*5120);vals.append((v,c,d))
  got=np.mean(vals,0);want=np.array([cell[k] for k in ('full_feature_mse','common_energy','contrast_energy')]);checks.append(float(np.max(abs(got-want))))
 assert max(checks)<1e-12 and all(sha(k)==v for k,v in assets.items()) and all(sha(ROOT/k)==v for k,v in read(ROOT/'source_lock.json')['files'].items())
 result=dict(status='DESCRIPTIVE_CACHE_DECOMPOSITION_AUDITED',source_rows=1152,cells=results,independent_quadratic_energy_checks=24,independent_max_difference=max(checks),energy_identity_max=maxidentity,source_lock_sha256=sha(ROOT/'source_lock.json'),assets=assets,GPU=0,training=False,model_forward=False,fitting=False,predictions_modified=False,original_gates_changed=False,test_opened=False,fresh9137=False,phase1_go=False,no_bias_calibration_candidate_created=True)
 (ROOT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(status=result['status'],checks=24,max_difference=max(checks))))
if __name__=='__main__':main(int(sys.argv[1]))
