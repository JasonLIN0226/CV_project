"""Independent reconstruction from primitive scalar geometry; no analysis helper."""
import collections,csv,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/gate_attribution_check'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 result=json.loads((OUT/'summary.json').read_text());assert all(sha(Path(p))==v for p,v in result['inputs'].items())
 source=ROOT/'results/gate_cost_diagnosis/native_completed/output/job_8282/geometry_pairs.csv'
 with source.open() as f:rr=list(csv.DictReader(f))
 groups=collections.defaultdict(dict)
 for r in rr:groups[(r['split'],r['pool_group'],int(r['horizon']))].setdefault(int(r['seed']),{})[r['pair']]=r
 counts=collections.Counter();percond=collections.Counter();maxci=0.;n=0
 for cc in result['cells']:
  g=groups[(cc['split'],cc['pool_group'],cc['horizon'])];ids=sorted(g[17]);seeds=(17,29,43) if cc['seed']=='mean' else (cc['seed'],)
  get=lambda field:np.mean([[float(g[s][p][field]) for p in ids] for s in seeds],axis=0)
  ca=get('critical_total');oa=get('other_total')
  # Independently obtain the mean-adjustment loss by subtracting the change in common energy.
  adjusted=oa+(get('critical_common')-get('other_common'))
  positive=get('critical_contrast')
  for name,c,o in [('actual_feature',ca,oa),('paired_mean_control_feature',ca,adjusted),('ideal_truth_control_feature',positive,np.zeros(128))]:
   diff=c-o;ix=np.random.default_rng(0).choice(128,size=(10000,128),replace=True);lo,hi=np.percentile(np.mean(diff[ix],axis=1),[2.5,97.5]);relative=(np.mean(c)-np.mean(o))/max(np.mean(c),1e-8);passed=bool(np.mean(c)>=1e-8 and relative>=.1 and lo>0);saved=cc[name]
   assert saved['passed']==passed;maxci=max(maxci,abs(saved['low']-lo),abs(saved['high']-hi));n+=1
   if cc['seed']!='mean':counts[name]+=passed;percond[(cc['split'],name)]+=passed
 assert maxci<1e-12 and counts['actual_feature']==11 and counts['paired_mean_control_feature']==34 and n==216
 assert result['original_joint_pass']==10 and result['original_Phase1_GO'] is False
 audit=dict(status='PASS_INDEPENDENT_LOCAL_SCALAR_CONTROLS',summary_sha256=sha(OUT/'summary.json'),audit_source_sha256=sha(__file__),checks_reconstructed=n,max_interval_difference=maxci,actual_feature_checks=11,offline_control_feature_checks=34,these_are_not_new_model_passes=True,new_model_calls=0,new_training=False,new_GPU_seconds=0,original_rules_changed=False)
 (OUT/'independent_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit))
if __name__=='__main__':main()
