from pathlib import Path
import ast,importlib.util,sys
import numpy as np
R=Path(__file__).resolve().parents[1];S=R/'experiments/gate_locked_test'
def module(name):
 spec=importlib.util.spec_from_file_location(name,S/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_locked_intervention_identical_and_original_scoring_loop_preserved():
 assert (S/'intervention.py').read_bytes()==(R/'experiments/gate_transport_ablation/intervention.py').read_bytes()
 before=ast.parse((R/'experiments/predictor/control/src/phase1/repair_evaluation.py').read_text());after=ast.parse((S/'original_evaluation.py').read_text())
 def score_loop(tree):return next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple) and [getattr(x,'id',None) for x in n.target.elts]==['rec','sample','prediction'])
 assert ast.dump(score_loop(before),include_attributes=False)==ast.dump(score_loop(after),include_attributes=False)

def test_complete_critical_statistics_independently_reconstructed():
 sys.path.insert(0,str(R/'experiments/predictor/control/src'));a=module('deletion_primary');b=module('deletion_stats')
 rows=[]
 for s in (17,29,43):
  for sp in ('test_id','test_long_gap','test_distractor_shift'):
   for p in range(128):
    for h in (3,6):
     for gi,g in enumerate(('cue_oldest','cue_middle','cue_newest')):
      for ep in (0,1):
       for d in (2,7,12):
        critical=d==(2,7,12)[gi];v=1.5+p/1000 if critical else .4
        rows.append(dict(seed=s,split=sp,pair=str(p),episode=str(ep),pool_group=g,horizon=h,cue=(2,7,12)[gi],deleted=d,feature_mse=v,task_error=float(critical)))
 pairs,cells=a.analyze(rows);pp,cc=b.compute(rows)
 assert len(pairs)==len(pp)==6912 and len(cells)==len(cc)==72 and all(x['passed'] for x in cells)
 ix={(str(c['seed']),c['split'],c['pool_group'],c['horizon']):c for c in cc}
 for c in cells:
  other=ix[(str(c['seed']),c['split'],c['pool_group'],c['horizon'])];assert c['passed']==other['passed']
  for f in ('feature_mse','task_error'):assert abs(c['metrics'][f]['difference']['low']-other['metrics'][f]['low'])<1e-12
