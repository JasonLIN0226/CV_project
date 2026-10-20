"""Read existing paired Gate losses only; never changes registered decisions."""
from pathlib import Path
from collections import Counter,defaultdict
import csv,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'results/age_control/native_completed/output/job_8275'
def main():
 summary=json.loads((P/'summary.json').read_text());audit=json.loads((P/'independent_audit.json').read_text())
 assert hashlib.sha256((P/'summary.json').read_bytes()).hexdigest()==audit['summary_sha256']
 assert hashlib.sha256((P/'paired_policy_losses.csv').read_bytes()).hexdigest()==audit['paired_sha256']
 checks=[r for r in summary['gate_critical_contrasts'] if r['seed']!='mean'];counts=Counter();cells=[]
 for r in checks:
  cell={k:r[k] for k in ('seed','split','pool_group','horizon')};cell['original_passed']=r['passed']
  for metric in ('feature_mse','task_error'):
   v=r['metrics'][metric];reason='pass' if v['passed'] else ('nonpositive_mean' if v['difference']['mean']<=0 else ('positive_below_10pct' if v['relative_reduction']<.1 else 'CI_includes_zero'))
   counts[metric+':'+reason]+=1;cell[metric]=dict(mean=v['difference']['mean'],CI_low=v['difference']['low'],CI_high=v['difference']['high'],relative_reduction=v['relative_reduction'],reason=reason)
  cells.append(cell)
 with (P/'paired_policy_losses.csv').open() as f:rows=[r for r in csv.DictReader(f) if r['task']=='gate']
 def signs(rr):
  f=np.array([float(r['critical_feature_mse'])-float(r['other_feature_mse']) for r in rr]);t=np.array([float(r['critical_task_error'])-float(r['other_task_error']) for r in rr])
  return dict(pair_seed_group_horizon_rows=len(rr),both_positive=int(((f>1e-8)&(t>1e-8)).sum()),task_positive_feature_nonpositive=int(((t>1e-8)&(f<=1e-8)).sum()),task_positive=int((t>1e-8).sum()),task_tied=int((np.abs(t)<=1e-8).sum()),task_negative=int((t< -1e-8).sum()),pearson_delta_correlation=float(np.corrcoef(f,t)[0,1]))
 by={}
 for g in ('cue_oldest','cue_middle','cue_newest'):
  by[g]=signs([r for r in rows if r['pool_group']==g])
 metrics=json.loads((ROOT/'results/reference/gate/probes/metrics.json').read_text())
 decoder=[r for r in metrics['results'] if r['variant']=='state_decoder_on_true_target' and r['split'].startswith('test_')]
 result=dict(scope='Post-hoc scalar diagnosis of revealed6301, no new model calls',old_summary_sha256=audit['summary_sha256'],old_pair_csv_sha256=audit['paired_sha256'],original_checks=54,original_passed=sum(r['passed'] for r in checks),failure_breakdown=dict(counts),pair_signs=signs(rows),by_cue_position=by,original_real_target_decoder=decoder,cells=cells,no_new_acceptance_rule=True,original_Phase1_GO=False,GPU_seconds_added=0)
 out=ROOT/'results/gate_cost_diagnosis';out.mkdir(exist_ok=True);(out/'existing_scalar_diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:result[k] for k in ('failure_breakdown','pair_signs','by_cue_position')},indent=2))
if __name__=='__main__':main()
