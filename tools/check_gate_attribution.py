"""Minimal post-hoc scalar controls, no torch/model/data generation/remote calls."""
import collections,csv,datetime,hashlib,importlib.util,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/gate_attribution_check';SCI=ROOT/'experiments/gate_attribution_check'
spec=importlib.util.spec_from_file_location('gate_attribution_id',SCI/'identities.py');math=importlib.util.module_from_spec(spec);spec.loader.exec_module(math)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def verify_transfer(directory,relative):
 p=directory/relative;h=sha(p);ledger=read(directory/'transfer_integrity.json')['files'];entries=[r for r in ledger if Path(r['remote']).as_posix().endswith('/'+relative)]
 assert len(entries)==1 and entries[0]['native_before_sha256']==entries[0]['local_sha256']==entries[0]['native_after_sha256']==h
 return h

def main():
 OUT.mkdir(exist_ok=True)
 lock=read(SCI/'source_lock.json');assert all(sha(SCI/k)==v for k,v in lock['files'].items())
 g=ROOT/'results/gate_cost_diagnosis/native_completed';c=ROOT/'results/gate_cost_diagnosis/native_context';cal=ROOT/'results/gate_calibration/native_completed';inputs={}
 for base,rel in [(g,'output/job_8282/geometry_pairs.csv'),(g,'output/job_8282/summary.json'),(g,'output/job_8282/independent_audit.json'),(c,'output/job_8282/context_pairs.csv'),(c,'output/job_8282/context_scalar_diagnosis.json'),(cal,'output/job_8289/paired_selected.csv'),(cal,'output/job_8289/independent_audit.json')]:inputs[str(base/rel)]=verify_transfer(base,rel)
 parent=read(g/'output/job_8282/summary.json');audit=read(g/'output/job_8282/independent_audit.json');assert audit['summary_sha256']==sha(g/'output/job_8282/summary.json');assert parent['paired_csv_sha256']==sha(g/'output/job_8282/geometry_pairs.csv')
 with (g/'output/job_8282/geometry_pairs.csv').open() as f:rows=list(csv.DictReader(f))
 with (cal/'output/job_8289/paired_selected.csv').open() as f:old=list(csv.DictReader(f))
 key=lambda r:(int(r['seed']),r['split'],r['pool_group'],r['pair'],int(r['horizon']))
 oi={key(r):r for r in old};assert len(oi)==len(rows)==6912
 grouped=collections.defaultdict(dict);max_roundoff=0.;identity_error=0.
 for r in rows:
  k=key(r);v=oi[k]
  assert float(r['critical_prediction_pair_signal'])==float(r['critical_prediction_sibling_maxabs'])==0
  assert abs(float(r['critical_contrast'])-float(r['critical_target_pair_signal']))<1e-12
  for side in ('critical','other'):
   max_roundoff=max(max_roundoff,abs(float(r[side+'_total'])-float(v[side+'_feature_mse'])))
   assert float(r[side+'_original_feature_mse'])==float(v[side+'_feature_mse'])
   identity_error=max(identity_error,abs(float(r[side+'_total'])-float(r[side+'_common'])-float(r[side+'_contrast'])))
  grouped[(r['split'],r['pool_group'],int(r['horizon']))].setdefault(int(r['seed']),{})[r['pair']]=r
 assert max_roundoff<1e-7 and identity_error<1e-12
 cells=[]
 for (sp,group,h),ss in sorted(grouped.items()):
  ids=sorted(ss[17]);assert len(ids)==128 and set(ss)=={17,29,43} and all(set(ss[s])==set(ids) for s in ss)
  for seed in (17,29,43,'mean'):
   def vec(field):
    a=np.array([[float(ss[s][i][field]) for i in ids] for s in (17,29,43)]);return a.mean(0) if seed=='mean' else a[(17,29,43).index(seed)]
   critical=vec('critical_total');other=vec('other_total');controlled_other=vec('critical_common')+vec('other_contrast')
   actual=math.feature_check(critical,other);controlled=math.feature_check(critical,controlled_other)
   # Analytic positive control uses future targets; never a deployable model.
   ideal=math.feature_check(vec('critical_target_pair_signal'),np.zeros(128))
   r=dict(seed=seed,split=sp,pool_group=group,horizon=h,actual_feature=actual,paired_mean_control_feature=controlled,ideal_truth_control_feature=ideal,
       class_parallel_delta=float((vec('critical_parallel')-vec('other_parallel')).mean()),binary_task_delta=float((vec('critical_task_error')-vec('other_task_error')).mean()),contrast_delta=float((vec('critical_contrast')-vec('other_contrast')).mean()),common_delta=float((vec('critical_common')-vec('other_common')).mean()))
   cells.append(r)
 original=read(cal/'output/job_8289/independent_audit.json');baseline={(str(r['seed']),r['split'],r['pool_group'],r['horizon']):r for r in original['cells']}
 for r in cells:
  old=baseline[(str(r['seed']),r['split'],r['pool_group'],r['horizon'])]['metrics']['feature_mse'];assert r['actual_feature']['passed']==old['passed']
 def count(rr):
  wrong=[r for r in rr if r['actual_feature']['delta']<=0]
  return dict(cells=len(rr),actual_feature_pass=sum(r['actual_feature']['passed'] for r in rr),controlled_feature_pass=sum(r['paired_mean_control_feature']['passed'] for r in rr),controlled_positive_mean=sum(r['paired_mean_control_feature']['delta']>0 for r in rr),actual_wrong_mean=len(wrong),wrong_mean_repaired_sign=sum(r['paired_mean_control_feature']['delta']>0 for r in wrong),wrong_mean_controlled_feature_pass=sum(r['paired_mean_control_feature']['passed'] for r in wrong),wrong_mean_also_class_parallel_improves=sum(r['class_parallel_delta']>0 for r in wrong),wrong_mean_binary_task_improves=sum(r['binary_task_delta']>0 for r in wrong),ideal_truth_positive_control_pass=sum(r['ideal_truth_control_feature']['passed'] for r in rr))
 indiv=[r for r in cells if r['seed']!='mean'];mean=[r for r in cells if r['seed']=='mean'];overall=count(indiv);assert overall['actual_feature_pass']==11 and count(mean)['actual_feature_pass']==1
 # Condition summaries use one base pair as unit after averaging all repeated axes.
 ag=collections.defaultdict(list)
 for r in rows:ag[(r['split'],r['pair'])].append(r)
 condition={};pairsummary=[]
 for sp in ('test_id','test_long_gap','test_distractor_shift'):
  ids=sorted(i for s,i in ag if s==sp);assert len(ids)==128
  for pair in ids:
   rr=ag[(sp,pair)];assert len(rr)==18
   v={field:float(np.mean([float(r[field]) for r in rr])) for field in ('critical_total','other_total','critical_common','other_common','critical_contrast','other_contrast')}
   pairsummary.append(dict(split=sp,pair=pair,actual_delta=v['critical_total']-v['other_total'],mean_control_delta=v['critical_contrast']-v['other_contrast'],common_delta=v['critical_common']-v['other_common']))
  pr=[r for r in pairsummary if r['split']==sp]
  def ci(field):
   a=np.array([r[field] for r in pr]);ix=np.random.default_rng(0).integers(0,128,(10000,128));lo,hi=np.quantile(a[ix].mean(1),[.025,.975]);return dict(mean=float(a.mean()),low=float(lo),high=float(hi))
  condition[sp]=dict(feature_checks=count([r for r in indiv if r['split']==sp]),actual_delta=ci('actual_delta'),mean_control_delta=ci('mean_control_delta'),common_delta=ci('common_delta'))
 with (OUT/'pair_controls.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(pairsummary[0]),lineterminator='\n');w.writeheader();w.writerows(pairsummary)
 context=read(c/'output/job_8282/context_scalar_diagnosis.json')['overall']
 result=dict(status='COMPLETE_LOCAL_SCALAR_ATTRIBUTION_DIAGNOSIS',scope='Posthoc revealed6301; paired oracle-assisted feature control, not scientific reacceptance',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_lock_sha256=sha(SCI/'source_lock.json'),analysis_sha256=sha(__file__),inputs=inputs,actual_outcomes_reused=41472,paired_rows=6912,independent_base_pairs_per_condition=128,max_float32_geometry_roundoff=max_roundoff,max_energy_identity_error=identity_error,
  individual_seed=overall,seedmean=count(mean),by_condition=condition,context=context,existing_true_feature_decoder=dict(correct=1536,total=1536,not_forecasting_accuracy=True),ideal_control_future_truth=True,negative_control_zero_by_identity=True,new_model_calls=0,new_GPU_seconds=0,new_training=False,new_scalar_fitting=False,original_feature_pass=11,original_joint_pass=10,original_joint_total=54,original_seedmean_joint_pass=1,original_seedmean_total=18,original_Phase1_GO=False,original_gates_changed=False,fresh9137_opened=False,github_updated=False,cells=cells)
 write(OUT/'summary.json',result)
 print(json.dumps({k:result[k] for k in ('individual_seed','seedmean','by_condition','max_float32_geometry_roundoff','max_energy_identity_error')},indent=2))
if __name__=='__main__':main()
