"""Controlled renderer information check; no encoder, fitting or heldout data."""
import copy,hashlib,json,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(BASE/'phase1_jepa_memory/src'))
from phase1.config import load_config
from phase1.envs import scene,episode
c=load_config(BASE/'phase1_jepa_memory/configs/phase1.yaml');rows=[]
for i in range(32):
 s=scene(c,'motion','train',i);q=s['query']
 for variable,values in [('xref',(.39,.61)),('y',(.44,.56)),('speed',(.0051,.0069))]:
  ss=[]
  for v in values:
   z=copy.deepcopy(s);z[variable]=v;ss.append([episode(z,label) for label in (0,1)])
  a,b=ss;recent_equal=all(np.array_equal(a[l]['rgb'][q-3:q+1],b[l]['rgb'][q-3:q+1]) for l in (0,1));history_diff=any(not np.array_equal(a[l]['rgb'][s['start']:s['end']+1],b[l]['rgb'][s['start']:s['end']+1]) for l in (0,1))
  for h in (3,6):
   common=[np.mean([ep['rgb'][q+h].astype(float) for ep in eps],axis=0) for eps in (a,b)]
   rows.append(dict(index=i,variable=variable,horizon=h,recent_RGB_exact=recent_equal,actions_exact=np.array_equal(a[0]['actions'],b[0]['actions']),history_RGB_different=history_diff,pair_common_future_RGB_MSE=float(np.mean((common[0]-common[1])**2))))
assert all(r['recent_RGB_exact'] and r['actions_exact'] and r['history_RGB_different'] and r['pair_common_future_RGB_MSE']>0 for r in rows)
out=BASE/'CV_project/results/motion_scene_diagnosis';out.mkdir(exist_ok=True,parents=True)
(out/'observability.json').write_text(json.dumps(dict(status='192_OF_192_CONTROLLED_RGB_OBSERVABILITY_WITNESSES',checks=rows,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Simulator information witness only; no claim about encoded-null residual predictability or Phase1 success',GPU=0),indent=2)+'\n')
print('192/192 exact same recent RGB and actions; different history and pair-common future RGB')
