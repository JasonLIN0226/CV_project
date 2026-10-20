import csv,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];D=R/'results/motion_history_reference_validation';N=D/'native_completed';O=N/'output/job_8449'
s=json.loads((O/'summary.json').read_text());a=json.loads((O/'independent_audit.json').read_text());proof=json.loads((N/'transfer_integrity.json').read_text());assert proof['native_before_local_native_after_all_match']
for n,v in proof['files'].items():assert hashlib.sha256((N/n).read_bytes()).hexdigest()==v['sha256']
assert a['status']=='PASS_INDEPENDENT_QR_AND_SCALAR_AUDIT'
with (O/'scalar_rows.csv').open() as f:raw=list(csv.DictReader(f))
ix=np.random.default_rng(0).integers(0,96,(2000,96));diff=[]
for c in s['cells']:
 seeds=(17,29,43) if c['seed']=='mean' else (c['seed'],);rr=[r for r in raw if int(r['seed']) in seeds and int(r['horizon'])==c['horizon']];pairs=sorted({r['pair'] for r in rr});assert len(pairs)==96
 for name,arm,metric,floor,absolute in [('feature_vs_recent','recent','feature_mse',.02,.01),('position_vs_recent','recent','position_mse',1e-5,1e-6),('feature_vs_current','current','feature_mse',.02,.01),('position_vs_current','current','position_mse',1e-5,1e-6)]:
  def vec(arm):return np.array([np.mean([float(r[metric]) for r in rr if r['arm']==arm and r['pair']==p]) for p in pairs])
  b,z=vec(arm),vec('candidate');useabs=b.mean()<floor
  effect=lambda x,y:(x-y if useabs else (x-y)/np.maximum(x,floor))
  v=float(effect(b.mean(),z.mean()));lo,hi=np.quantile(effect(b[ix].mean(1),z[ix].mean(1)),[.025,.975]);d=max(abs(v-c[name]['mean']),abs(lo-c[name]['low']),abs(hi-c[name]['high']));assert d<1e-11;diff.append(d)
(D/'local_review.json').write_text(json.dumps(dict(status='PASS',intervals=32,max_difference=max(diff)),indent=2)+'\n')
cs=[c for c in s['cells'] if c['seed']=='mean'];fig,ax=plt.subplots(figsize=(6,3.5),layout='constrained');v=[100*c['feature_vs_recent']['mean'] for c in cs];ax.bar(['3 steps','6 steps'],v,color='.4',edgecolor='black');ax.axhline(10,color='black',ls='--',label='Original10%floor');ax.set(ylabel='Feature error reduction vs recent (%)',title='Frozen historical reference + current position');ax.legend()
for i,x in enumerate(v):ax.text(i,x+.4,f'{x:.2f}%',ha='center')
fig.savefig(D/'validation_result.png',dpi=160);plt.close(fig)
lines=['# Frozen historical-reference combination: validation','','One fixed candidate, coefficient1. Use the frozen v3 model on the same legitimate two historical observations as the visual-output base; apply the minimum fixed-basis correction preserving the current guarded model decoded position. No fitting, training, encoder call, newGPU, test or fresh9137access at this validation stage. Original96pairs/3seeds/h3,h6 reuse1,152complete current/recent/target vectors; one frozen CPU pass fills missing v3 reference-history outputs. Source replay maximum discrepancy:'+f"{s['reference_replay_max']:.3g}",'', '![Validation](../results/motion_history_reference_validation/validation_result.png)','','| Check | Met / total |','|---|---:|',f"|Original feature numeric rule|{s['feature_checks']}/8|",f"|Original position numeric rule|{s['position_checks']}/8|",f"|Incremental feature lowerCI above0|{s['incremental_feature_positive']}/8|",'','The third row is an incremental diagnostic, not a new original pass threshold. Eight cells contain3seed estimates and their shared-data mean at2horizons. All original effect floors and paired confidence rules remain unchanged.','','| Horizon | Feature gain vs recent,95%CI | Feature gain vs current | Position gain vs recent |','|---|---|---:|---:|']
for c in cs:
 f=c['feature_vs_recent'];lines.append(f"|{c['horizon']}|{100*f['mean']:.2f}%[{100*f['low']:.2f},{100*f['high']:.2f}]|{100*c['feature_vs_current']['mean']:.2f}%|{100*c['position_vs_recent']['mean']:.2f}%|")
lines+=['','Independent QR reconstruction and all5,760scalar outcomes/32intervals passed; local32intervals match. All11compact native-before/local/native-after hashes match. The earlier inherited extra-screen preparation was clarified before upload or experimental outputs; no original numeric criterion changed, and all8incremental diagnostics pass regardless.','','**Validation supports locking this exact composite for the separately recorded full original recheck. It is not Phase1_GO.** No coefficient search or later test-driven switching. See[complete review](MOTION_HISTORY_REFERENCE_LOCKED_TEST.md). The fixed decoder/RGB guard remain task/renderer-specific. GPU remains12,723/28,800seconds; no GitHub update.']
(R/'docs/MOTION_HISTORY_REFERENCE_VALIDATION.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(status='VALIDATION_REPORT_COMPLETE',feature=s['feature_checks'],position=s['position_checks'],intervals=32)))
