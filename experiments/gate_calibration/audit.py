"""Independent scalar reconstruction of selection and unchanged deletion rules."""
import csv,hashlib,json,sys,collections
from pathlib import Path
import numpy as np
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def compute(rr,repeats=10000):
 grouped=collections.defaultdict(list)
 for r in rr:grouped[(int(r['seed']),r['split'],r['pool_group'],r['pair'],int(r['horizon']))].append(r)
 pairrows=[]
 for (seed,split,group,pair,h),rs in sorted(grouped.items()):
  assert len(rs)==6 and len({r['episode'] for r in rs})==2
  cue=int(rs[0]['cue']);cr=[r for r in rs if int(r['deleted'])==cue];ot=[r for r in rs if int(r['deleted'])!=cue];assert len(cr)==2 and len(ot)==4
  d=dict(seed=seed,split=split,pool_group=group,pair=pair,horizon=h)
  for metric in ('feature_mse','task_error'):
   d['critical_'+metric]=float(np.mean([float(r[metric]) for r in cr]));d['other_'+metric]=float(np.mean([float(r[metric]) for r in ot]))
  pairrows.append(d)
 by=collections.defaultdict(dict)
 for r in pairrows:by[(r['split'],r['pool_group'],r['horizon'])].setdefault(r['seed'],{})[r['pair']]=r
 cells=[]
 for (split,g,h),ss in sorted(by.items()):
  ids=sorted(ss[17]);assert len(ids)==128 and set(ss)=={17,29,43} and all(set(ss[s])==set(ids) for s in ss)
  ix=np.random.default_rng(0).integers(0,128,(repeats,128))
  for seed in (17,29,43,'mean'):
   cell=dict(seed=seed,split=split,pool_group=g,horizon=h,metrics={})
   for m in ('feature_mse','task_error'):
    a=np.array([[ss[s][i]['critical_'+m] for i in ids] for s in (17,29,43)]);b=np.array([[ss[s][i]['other_'+m] for i in ids] for s in (17,29,43)])
    a=a.mean(0) if seed=='mean' else a[(17,29,43).index(seed)];b=b.mean(0) if seed=='mean' else b[(17,29,43).index(seed)];delta=a-b;boot=delta[ix].mean(1);lo,hi=np.quantile(boot,[.025,.975]);rel=float(delta.mean()/max(a.mean(),1e-8))
    cell['metrics'][m]=dict(critical_mean=float(a.mean()),other_mean=float(b.mean()),delta=float(delta.mean()),low=float(lo),high=float(hi),relative=rel,passed=bool(a.mean()>=1e-8 and rel>=.1 and lo>0))
   cell['passed']=all(x['passed'] for x in cell['metrics'].values());cells.append(cell)
 return pairrows,cells

def main(folder):
 root=Path(__file__).parent;out=Path(folder);summary=json.loads((out/'summary.json').read_text());selection=json.loads((out/'selection_lock.json').read_text())
 assert sha(out/'raw_selected.csv')==summary['raw_sha256'] and sha(out/'selection_lock.json')==summary['selection_sha256']
 with (out/'validation_scores.csv').open() as f:val=list(csv.DictReader(f))
 assert len(val)==23040 and sha(out/'validation_scores.csv')==selection['validation_csv_sha256']
 keys={(r['seed'],r['condition'],r['pair'],r['episode'],r['horizon'],r['alpha']) for r in val};assert len(keys)==23040
 assert {r['condition'] for r in val}=={'recent_only','full_history','privileged_small_budget','random_small_budget'}
 scores={a:float(np.mean([float(r['feature_mse']) for r in val if float(r['alpha'])==a])) for a in (0.,.25,.5,.75,1.)};chosen=sorted(scores,key=lambda a:(scores[a],-a))[0]
 assert chosen==summary['selected_alpha']==selection['selected_alpha']
 assert all(scores[a]==selection['score_by_alpha'][str(a)] for a in scores)
 with (out/'raw_selected.csv').open() as f:raw=list(csv.DictReader(f))
 assert len(raw)==41472 and all(float(r['alpha'])==chosen and int(r['history_count'])==2 for r in raw)
 assert len({(r['seed'],r['split'],r['episode'],r['pool_group'],r['deleted'],r['horizon']) for r in raw})==41472
 for r in raw:
  parent=list(map(int,r['parent'].split(';')));assert len(parent)==3 and len(set(parent))==3 and int(r['deleted']) in parent and int(r['cue']) in parent
 assert np.isfinite([[float(r[k]) for k in ('feature_mse','task_error','old_feature_mse','old_task_error')] for r in raw]).all()
 pairs,cells=compute(raw)
 oldraw=[dict(r,feature_mse=r['old_feature_mse'],task_error=r['old_task_error']) for r in raw];_,baseline=compute(oldraw)
 parent=json.loads(Path('/scratch/xl4358/phase1_age_control_a1/output/job_8275/summary.json').read_text());original={(str(x['seed']),x['split'],x['pool_group'],x['horizon']):x for x in parent['gate_critical_contrasts']}
 for x in baseline:
  old=original[(str(x['seed']),x['split'],x['pool_group'],x['horizon'])];assert x['passed']==old['passed']
  for m,v in x['metrics'].items():
   target=old['metrics'][m];assert abs(v['low']-target['difference']['low'])<1e-12 and abs(v['high']-target['difference']['high'])<1e-12 and v['passed']==target['passed']
 assert all(sha(p)==v for p,v in json.loads((out/'asset_ledger.json').read_text()).items())
 assert all(sha(root/p)==v for p,v in json.loads((root/'source_lock.json').read_text())['files'].items())
 def counts(cc,mean):
  rr=[r for r in cc if (r['seed']=='mean')==mean];return dict(both=sum(r['passed'] for r in rr),feature=sum(r['metrics']['feature_mse']['passed'] for r in rr),task=sum(r['metrics']['task_error']['passed'] for r in rr),total=len(rr))
 result=dict(status='PASS_INDEPENDENT_SCALAR_AUDIT',summary_sha256=sha(out/'summary.json'),raw_sha256=sha(out/'raw_selected.csv'),selected_alpha=chosen,baseline=counts(baseline,False),selected=counts(cells,False),baseline_seedmean=counts(baseline,True),selected_seedmean=counts(cells,True),validation_rows=len(val),raw_rows=len(raw),pair_rows=len(pairs),cells=cells,original_gates_changed=False,original_Phase1_GO=False,no_test_selection=True,fresh9137_opened=False,GPU=0)
 with (out/'paired_selected.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(pairs[0]),lineterminator='\n');w.writeheader();w.writerows(pairs)
 (out/'independent_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='cells'}),flush=True)
if __name__=='__main__':main(sys.argv[1])
