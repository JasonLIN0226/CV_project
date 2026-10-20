"""Read-only native metadata, to run only on an actual course CPU allocation."""
from pathlib import Path
import hashlib,json,os,socket,subprocess,sys
ROOT=Path('/scratch/xl4358/phase1_independent_confirmation_execution_v2')
def run(args):
 p=subprocess.run(args,text=True,capture_output=True,env={**os.environ,'TZ':'UTC'})
 return dict(arguments=args,exit=p.returncode,stdout=p.stdout,stderr=p.stderr)
job=sys.argv[1]
current=run(['/opt/slurm/bin/scontrol','show','job','-o',job]);text=current['stdout']
assert current['exit']==0 and 'JobState=RUNNING ' in text and 'Account=csci_ga_2271-2026fa ' in text and 'gres/gpu=' not in text
assert 'NodeList='+socket.gethostname().split('.')[0]+' ' in text
lock=json.loads((ROOT/'source_lock.json').read_text())
source={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in lock['files']}
assert source==lock['files']
receipts={str(p.relative_to(ROOT)):json.loads(p.read_text()) for p in sorted((ROOT/'receipts').glob('*.json'))}
small={}
for p in [ROOT/'preflight_cpu.json',ROOT/'preflight_cpu_terminal.json',ROOT/'preflight_gpu.json',ROOT/'fresh/sequence_budget.json',ROOT/'fresh/last_budget_forecast.json']:
 if p.exists():small[str(p.relative_to(ROOT))]=json.loads(p.read_text())
proof=dict(current_CPU_native=current,source77_current_match=True,receipts=receipts,small=small,
 metadata=[run(['/opt/slurm/bin/sacct','-D','-n','-P','-j','8533,8536','--format=JobID,JobIDRaw,State,ExitCode,ElapsedRaw,Start,End,Submit,Eligible,Account,AllocTRES']),
 run(['/opt/slurm/bin/scontrol','show','job','-o','8536'])],
 stage_attempts=[p.name for p in (ROOT/'stage_attempts').glob('*.json')],
 failures={p.name:json.loads(p.read_text()) for p in (ROOT/'stage_failures').glob('*.json')},
 fresh_permit_exists=(ROOT/'permit.json').exists(),no_model_or_encoder_forward_by_this_reader=True,
 official_preflight_cpu_sha256=hashlib.sha256((ROOT/'preflight_cpu.json').read_bytes()).hexdigest())
(ROOT/'startup_native_settlement.json').write_text(json.dumps(proof,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(saved=str(ROOT/'startup_native_settlement.json'),source_files=len(source),receipts=list(receipts),metadata=proof['metadata']),indent=2))
