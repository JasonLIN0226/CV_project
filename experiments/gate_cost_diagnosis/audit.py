"""Independent scalar reconstruction and primitive vector checks; no model import."""
import csv,hashlib,json,os,subprocess,sys
from pathlib import Path
import numpy as np
METRICS=('total','parallel','orthogonal','common','contrast','basis_inside','basis_outside','task_error','margin_loss','score_error','label_score_error','prediction_pair_signal','signal_gain')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check_rows(rows):
 ids=set();cells={}
 for r in rows:
  k=(int(r['seed']),r['split'],r['pool_group'],r['pair'],int(r['horizon']));assert k not in ids;ids.add(k)
  assert k[0] in (17,29,43) and k[1] in ('test_id','test_long_gap','test_distractor_shift') and k[2] in ('cue_oldest','cue_middle','cue_newest') and k[-1] in (3,6)
  cells.setdefault(k[:3]+(k[-1],),set()).add(k[3])
  for prefix in ('critical_','other_'):
   total=float(r[prefix+'total'])
   for a,b in (('parallel','orthogonal'),('common','contrast'),('basis_inside','basis_outside')):
    assert abs(total-float(r[prefix+a])-float(r[prefix+b]))<1e-10
   assert abs(total-float(r[prefix+'original_feature_mse']))<1e-7
  assert abs(float(r['critical_basis_outside'])-float(r['other_basis_outside']))<1e-6
 assert len(rows)==6912 and len(cells)==54 and all(len(v)==128 for v in cells.values())
 return True

def audit_witness(f):
 p=f['prediction'].astype(np.float64);t=f['target'].astype(np.float64);e=p-t;w=f['w'];Q=f['Q'];D=p.shape[-1];u=w/np.linalg.norm(w)
 assert p.shape==t.shape==(162,2,5120) and np.max(np.abs(Q.T@Q-np.eye(Q.shape[1])))<1e-10
 total=np.sum(e*e,axis=(1,2))/(2*D);parallel=np.sum(np.einsum('nsd,d->ns',e,u)**2,axis=1)/(2*D)
 meanerr=(e[:,0]+e[:,1])/2;differr=(e[:,0]-e[:,1])/2
 inside=np.sum(np.einsum('nsd,dk->nsk',e,Q)**2,axis=(1,2))/(2*D)
 target_half=(t[:,0]-t[:,1])/2;pred_half=(p[:,0]-p[:,1])/2
 values=dict(total=total,parallel=parallel,orthogonal=total-parallel,common=np.mean(meanerr**2,axis=1),contrast=np.mean(differr**2,axis=1),basis_inside=inside,basis_outside=total-inside,
  target_pair_signal=np.mean(target_half**2,axis=1),prediction_pair_signal=np.mean(pred_half**2,axis=1),signal_gain=np.sum(pred_half*target_half,axis=1)/np.sum(target_half**2,axis=1),prediction_sibling_maxabs=np.max(np.abs(p[:,0]-p[:,1]),axis=1))
 for j,name in enumerate(f['metric_names'].tolist()):np.testing.assert_allclose(values[name],f['metrics'][:,j],atol=1e-10,rtol=1e-10)
 return values

def main(folder):
 folder=Path(folder);root=folder.parents[1];summary=json.loads((folder/'summary.json').read_text());protocol=json.loads((root/'protocol.json').read_text())
 assert sha(folder/'geometry_pairs.csv')==summary['paired_csv_sha256']
 with (folder/'geometry_pairs.csv').open() as f:rows=list(csv.DictReader(f))
 check_rows(rows)
 oldfolder=Path('/scratch/xl4358/phase1_age_control_a1/output/job_8275')
 parent=json.loads((oldfolder/'summary.json').read_text());assert sha(oldfolder/'summary.json')==protocol['parent_summary_sha256'] and sha(oldfolder/'raw_deletions.csv')==protocol['parent_raw_sha256']
 with (oldfolder/'paired_policy_losses.csv').open() as f:old={(int(r['seed']),r['split'],r['pool_group'],r['pair'],int(r['horizon'])):r for r in csv.DictReader(f) if r['task']=='gate'}
 for r in rows:
  prior=old[(int(r['seed']),r['split'],r['pool_group'],r['pair'],int(r['horizon']))]
  for role in ('critical','other'):
   assert abs(float(r[role+'_original_feature_mse'])-float(prior[role+'_feature_mse']))<1e-12
   assert float(r[role+'_task_error'])==float(prior[role+'_task_error'])
 data={(int(r['seed']),r['split'],r['pool_group'],r['pair'],int(r['horizon'])):r for r in rows}
 count=0
 for cell in summary['cells']:
  ss=(17,29,43) if cell['seed']=='mean' else (cell['seed'],);sp=cell['split'];g=cell['pool_group'];h=cell['horizon']
  pairs=sorted(k[3] for k in data if k[0]==17 and k[1]==sp and k[2]==g and k[-1]==h);assert len(pairs)==128
  for metric in METRICS:
   cr=np.array([[float(data[(seed,sp,g,pair,h)]['critical_'+metric]) for pair in pairs] for seed in ss]).mean(0)
   ot=np.array([[float(data[(seed,sp,g,pair,h)]['other_'+metric]) for pair in pairs] for seed in ss]).mean(0);delta=cr-ot
   rng=np.random.default_rng(0);bs=delta[rng.integers(0,128,(10000,128))].mean(1)
   target=cell['metrics'][metric];np.testing.assert_allclose([cr.mean(),ot.mean(),delta.mean(),np.quantile(bs,.025),np.quantile(bs,.975)],[target['critical_mean'],target['other_mean'],target['difference']['mean'],target['difference']['low'],target['difference']['high']],atol=1e-10,rtol=1e-10);count+=1
 assert count==936 and sha(folder/'vector_witnesses.npz')==summary['witness_sha256']
 with np.load(folder/'vector_witnesses.npz',allow_pickle=False) as f:
  audit_witness(f);witness_ids=f['witness_ids'].copy()
 with (oldfolder/'raw_deletions.csv').open() as f:goldgate=[r for r in csv.DictReader(f) if r['task']=='gate']
 rawforks={}
 for r in goldgate:
  k=(int(r['seed']),r['split'],r['pool'],r['pair'],int(r['horizon']),int(r['deleted']))
  rawforks.setdefault(k,[]).append((float(r['feature_mse']),float(r['task_error'])))
 for r in rows:
  k=(int(r['seed']),r['split'],r['pool_group'],r['pair'],int(r['horizon']));parent_times=list(map(int,r['parent'].split(';')));cue=int(r['cue'])
  fork={d:np.mean(rawforks[k+(d,)],axis=0) for d in parent_times};cf,ct=fork[cue]
  candidates=[v for d,v in fork.items() if d!=cue and v[1]<ct-1e-8]
  expected=dict(task_help_available=int(bool(candidates)),joint_help_available=int(any(v[0]<cf-1e-8 for v in candidates)),all_task_help_increases_feature=int(bool(candidates) and all(v[0]>cf+1e-8 for v in candidates)))
  assert all(int(r[name])==v for name,v in expected.items())
 first={}
 for r in goldgate:first.setdefault((r['seed'],r['split']),r['pair'])
 assert all(first[(seed,sp)]==pair for seed,sp,pair,g,h,d in witness_ids)
 assets=json.loads((folder/'asset_ledger.json').read_text());assert sha(folder/'asset_ledger.json')==summary['asset_ledger_sha256'] and all(sha(k)==v for k,v in assets.items())
 lock=json.loads((root/'source_lock.json').read_text());assert all(sha(root/k)==v for k,v in lock['files'].items())
 state=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',str(summary['job'])],text=True);assert 'JobState=RUNNING ' in state and 'Account=csci_ga_2271-2026fa ' in state
 result=dict(status='INDEPENDENT_GATE_GEOMETRY_AUDIT_PASS',reproduced_raw_rows=41472,paired_rows=6912,independent_metric_intervals=count,vector_witness_pairs=162,all_old_loss_and_class_checks=True,all_energy_identities=True,assets_stable=True,summary_sha256=sha(folder/'summary.json'),paired_csv_sha256=sha(folder/'geometry_pairs.csv'),source_lock_sha256=sha(root/'source_lock.json'),GPU=0,no_model_forward_or_training=True)
 (folder/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main(sys.argv[1])
