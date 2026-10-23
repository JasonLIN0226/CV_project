"""Independent 32-interval audit of compact cached validation scalars."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
def main(root):
 root=Path(root);s=json.loads((root/'validation/summary.json').read_text());rows=list(csv.DictReader((root/'validation/scalar_rows.csv').open()));assert len(rows)==3456;index={(r['seed'],r['horizon'],r['pair'],r['episode'],r['arm']):r for r in rows};assert len(index)==3456;maximum=0.;n=0
 for c in s['cells']:
  seed,h=c['seed'],str(c['horizon']);chosen=[r for r in rows if r['horizon']==h and r['arm']=='candidate' and (seed=='mean' or int(r['seed'])==seed)];pairs=sorted({r['pair'] for r in chosen});assert len(pairs)==96;draw=np.random.default_rng(0).integers(0,96,(2000,96))
  for name,arm,metric,floor,absolute in [('feature_vs_recent','recent','feature_mse',.02,.01),('position_vs_recent','recent','position_mse',1e-5,1e-6),('feature_vs_current','current','feature_mse',.02,.01),('position_vs_current','current','position_mse',1e-5,1e-6)]:
   matrix=[]
   for pair in pairs:
    rr=[r for r in chosen if r['pair']==pair];matrix.append([np.mean([float(index[(r['seed'],h,pair,r['episode'],arm)][metric]) for r in rr]),np.mean([float(r[metric]) for r in rr])])
   x=np.array(matrix);ab=x[:,0].mean()<floor
   def effect(z):
    b=z[...,0].mean(-1);a=z[...,1].mean(-1);return b-a if ab else (b-a)/np.maximum(b,floor)
   val=float(effect(x));lo,hi=np.quantile(effect(x[draw]),[.025,.975]);want=c[name];delta=max(abs(val-want['mean']),abs(lo-want['low']),abs(hi-want['high']));assert delta<1e-11 and want['numeric_rule_met']==bool(val>=(absolute if ab else .1) and lo>0);maximum=max(maximum,delta);n+=1
 eligible=all(c['feature_vs_recent']['numeric_rule_met'] and c['position_vs_recent']['numeric_rule_met'] and c['feature_vs_current']['low']>0 for c in s['cells']);assert eligible==s['validation_eligible']
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();out={'status':'PASS_LOCAL32CI_AND_NATIVE_VECTOR_IDENTITY','origin':'Independent local rawCSV interval implementation; native algebraic vector identity separately reported','source_sha256':sha(__file__),'comparisons':n,'max_difference':maximum,'validation_eligible':eligible,'validation_summary_sha256':sha(root/'validation/summary.json'),'scalar_rows_sha256':sha(root/'validation/scalar_rows.csv'),'native_vector_identity_max':s.get('vector_replay_max',s.get('vector_independent_max')),'new_GPU':0,'original_gates_changed':False,'phase1_go':False};(root/'validation/independent_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root');main(p.parse_args().root)
