"""Explicit compact-only SCP-O with native-before/local/native-after SHA equality."""
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEST = HERE / 'native_completed7426_7435'
ALIAS = 'phase1-far-finalevidence7451'
CONFIG = '/private/tmp/phase1-hpc-control/config'
SSH = ['ssh','-F',CONFIG,'-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ForwardAgent=no',ALIAS]
items = {}
for filename in ['audit.json','summary.json','record_summary.csv','storage_audit.json']:
    items['analysis_outputs/'+filename]='/scratch/xl4358/phase1_far_phase1_analysis/output/job_7435/'+filename
for filename in ['completion.json','progress.json','executed_model_proof.json','first_clip_budget.json']:
    items['science_outputs/'+filename]='/scratch/xl4358/phase1_far_phase1/output/job_7426/'+filename
for filename in ['protocol.json','input_proof.json','source_lock.json','online_memory.py','run_phase1.py','phase1.sbatch']:
    items['science_source/'+filename]='/scratch/xl4358/phase1_far_phase1/'+filename
for filename in ['source_lock.json','analysis_protocol.json','aggregate.py','analysis.sbatch','checks.json']:
    items['analysis_source/'+filename]='/scratch/xl4358/phase1_far_phase1_analysis/'+filename
for filename in ['final_native_audit.json','retention_diagnostic.json']:
    items['final_evidence/'+filename]='/scratch/xl4358/phase1_far_phase1_final_evidence_7451/'+filename
for root, filenames in [
    ('/scratch/xl4358/phase1_far_phase1/logs',['phase1-7426.out','phase1-7426.err']),
    ('/scratch/xl4358/phase1_far_phase1_analysis/logs',['audit-7435.out','audit-7435.err'])]:
    for filename in filenames: items['logs/'+filename]=root+'/'+filename

def native_manifest():
    program = """import hashlib,json,socket,subprocess
from pathlib import Path
assert socket.gethostname()=='b-9-10'
state=subprocess.check_output(['/opt/slurm/bin/squeue','-h','-j','7451','-o','%i|%T|%a|%N'],text=True).strip()
assert state=='7451|RUNNING|csci_ga_2271-2026fa|b-9-10',state
files=ITEMS
out={}
for name,p in files.items():
    q=Path(p); b=q.read_bytes()
    assert len(b)<=1000000,(name,len(b))
    out[name]={'remote':p,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
print(json.dumps(out))
""".replace('ITEMS',repr(items))
    r=subprocess.run(SSH+['/scratch/xl4358/phase1_env/bin/python','-'],input=program,text=True,capture_output=True,check=True)
    return json.loads(r.stdout)

DEST.mkdir(exist_ok=False)
before=native_manifest()
(DEST/'native_before.json').write_text(json.dumps(before,indent=2)+'\n')
for relative,remote in items.items():
    p=DEST/relative;p.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['scp','-O','-F',CONFIG,'-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
                    '-o','ForwardAgent=no',ALIAS+':'+remote,str(p)],check=True,capture_output=True)
local={name:{'bytes':(DEST/name).stat().st_size,'sha256':hashlib.sha256((DEST/name).read_bytes()).hexdigest()} for name in items}
after=native_manifest()
assert before==after,'Remote compact outputs changed during transfer'
assert all(v['sha256']==local[n]['sha256'] and v['bytes']==local[n]['bytes'] for n,v in before.items())
record={'status':'ALL_EXPLICIT_COMPACT_NATIVE_BEFORE_LOCAL_NATIVE_AFTER_SHA_EQUAL',
        'allocation':7451,'alias':ALIAS,'large_weights_arrays_traces_downloaded':False,
        'files':{name:{'native_before':before[name],'local':local[name],'native_after':after[name]} for name in items}}
(DEST/'transfer_integrity.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'status':record['status'],'files':len(items),'bytes':sum(x['bytes'] for x in local.values()),'destination':str(DEST)},indent=2))
