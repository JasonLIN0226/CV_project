from pathlib import Path
import json,hashlib,shutil,zipfile,datetime
ROOT=Path(__file__).resolve().parents[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=ROOT/'runs/phase1_v5_encoding/control';control=ROOT/'runs/phase1_v5_encoding_l4/control';control.mkdir(exist_ok=False)
old_lock=json.loads((old/'source_lock.json').read_text())
for rel,want in old_lock['files'].items():
 assert sha(old/rel)==want,rel
 dst=control/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(old/rel,dst)
shutil.copyfile(ROOT/'configs/phase1_v5_real_encoding_l4_protocol.json',control/'protocol.json')
shutil.copyfile(old/'protocol.json',control/'parent_a100_protocol.json')
extras=['src/phase1/v5_encoding_l4.py','scripts/v5_encode_real_l4.py','tests/test_v5_encoding_l4.py','tests/test_v5_causal_frame_stream.py']
for rel in extras:
 dst=control/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dst)
files=list(old_lock['files'])+['parent_a100_protocol.json']+extras
lock={'schema_version':1,'files':{rel:sha(control/rel) for rel in files},'scope':'Only explicit one-L4 runtime guard; old encoder and original encoding/decode core unchanged. Hardware/protocol/cache are independent.'}
(control/'source_lock.json').write_text(json.dumps(lock,indent=2)+'\n')
locked=sha(control/'source_lock.json')
batch=(old/'submit_v5_encoding.sh').read_text().replace('cv-v5-encode','cv-v5-encode-l4').replace('c12m85-a100-1','g2-standard-12').replace('--gres=gpu:1','--gres=gpu:l4:1').replace('--mem=60G','--mem=32G').replace('phase1_jepa_memory_v5_encoding','phase1_jepa_memory_v5_encoding_l4').replace('phase1_v5_real_encoding/job_','phase1_v5_real_encoding_l4/job_').replace('scripts/v5_encode_real.py','scripts/v5_encode_real_l4.py').replace('384ef961b067c6c0d7189e02ca370a13d680c5d8e11e13df67d1fdf8b3537ca2',locked)
(control/'submit_v5_encoding_l4.sh').write_text(batch)
with zipfile.ZipFile(ROOT/'runs/phase1_v5_encoding_l4/source.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(control.rglob('*')):
  if p.is_file():z.write(p,str(p.relative_to(control)))
proof={'status':'PREREGISTERED_PREPARED_NOT_SUBMITTED_NOT_ENCODED','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'source_files':len(files),'source_lock_sha256':locked,'protocol_sha256':sha(control/'protocol.json'),'batch_sha256':sha(control/'submit_v5_encoding_l4.sh'),
 'bundle_sha256':sha(ROOT/'runs/phase1_v5_encoding_l4/source.zip'),'old_all13_encoding_files_unchanged':True,
 'local_software_tests_passed':20,'tests_are_not_actual_gpu_encode_or_science':True,
 'only_core_diff':'Strict one-L4/partition runtime guard; all other main AST and eight encoding helper sources identical to old frozen A100 preparation.',
 'first_preflight_failure':'Test AST body index erroneously counted past the leading import; fixed before deploy. Production encoder/data were not changed.',
 'planned_dependency':'afterok:7001','same_frames':50688,'new_weights_trained':False,'gpu_count':1,'course':'csci_ga_2271-2026fa','partition':'g2-standard-12','cpus':4,'memory_gb':32,'time_limit':'04:00:00','first512_budget_rule_unchanged':True,'gpu_attempt_limit':1,'phase1_go':False}
(ROOT/'runs/phase1_v5_encoding_l4/preparation.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof))
