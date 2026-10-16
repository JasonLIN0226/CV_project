from pathlib import Path
import json,hashlib,statistics,datetime
BASE=Path(__file__).resolve().parents[1]
p=BASE/'native_diagnosis7580/diagnostic_results_v2.json'
d=json.loads(p.read_text());groups={}
for r in d['slices']:
 key=(r['task'],r['split'],r['condition'],r['horizon']);groups.setdefault(key,[]).append(r)
summary=[]
for key,rows in groups.items():
 summary.append(dict(zip(('task','split','condition','horizon'),key),**{k:statistics.mean(r[k] for r in rows) for k in rows[0] if k not in {'task','seed','split','condition','horizon','pairs'}}))
print(json.dumps(summary,indent=2))
