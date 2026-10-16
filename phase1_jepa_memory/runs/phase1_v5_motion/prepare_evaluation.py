from pathlib import Path
import json,hashlib,shutil,zipfile,datetime
ROOT=Path(__file__).resolve().parents[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
training=ROOT/'runs/phase1_v5_motion/control'
protocol=json.loads((ROOT/'configs/phase1_v5_gate_evaluation_protocol.json').read_text())
protocol.update(task='motion',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),training_job_id=7000,
 training_control='/scratch/xl4358/phase1_jepa_memory_v5_motion',training_root='/scratch/xl4358/phase1_v5_motion_development',
 training_source_lock_sha256=sha(training/'source_lock.json'),training_protocol_sha256=sha(training/'configs/phase1_v5_motion_training_protocol.json'),
 output='/scratch/xl4358/phase1_v5_motion_evaluation',
 prerequisite='Successful7000 completion plus independent all6 actual Motion current best/latest120epochs/7680steps finite/source/cache/basis/config/RNG/scheduler/optimizer/balancedcounts/earliestminimum proof before any new held-out prediction.',
 dependency='afterok:7000, with7000 itself afterok:6994. Single sequential science; no planned jobs run concurrently.',
 critical_absolute_slice='test_long_gap',evidence_policy='Same original Motion pre-occlusion start/end timestamps. This reader-only trial has no wide-span/two-vs-wide or pure-velocity claim.',
 source_code_unchanged_during_active_training=True,phase1_go=False)
(ROOT/'configs/phase1_v5_motion_evaluation_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
control=ROOT/'runs/phase1_v5_motion/evaluation_control';control.mkdir(exist_ok=False)
files=list(json.loads((training/'source_lock.json').read_text())['files'])
files+=['src/phase1/v5_motion_evaluation.py','scripts/v5_evaluate_motion.py','tests/test_v5_motion_evaluation.py','configs/phase1_v5_motion_evaluation_protocol.json']
for rel in files:
    dst=control/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dst)
lock={'files':{rel:sha(control/rel) for rel in files},'scope':'Independent Motion evaluator; explicit current reader; original start/end data and rules; no old trainer signature relabelling.'}
(control/'source_lock.json').write_text(json.dumps(lock,indent=2)+'\n')
locked=sha(control/'source_lock.json')
batch=(ROOT/'runs/phase1_v5_gate/evaluation_control/submit_v5_gate_evaluation.sh').read_text().replace('cv-v5-gate-eval','cv-v5-motion-eval').replace('phase1_jepa_memory_v5_gate_evaluation','phase1_jepa_memory_v5_motion_evaluation').replace('v5_evaluate_gate.py','v5_evaluate_motion.py').replace('2055b466634d59adcd951088eab03d6ac9d87d811f49c48d51d4d34958bd14a5',locked)
(control/'submit_v5_motion_evaluation.sh').write_text(batch)
with zipfile.ZipFile(ROOT/'runs/phase1_v5_motion/evaluation_source.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(control.rglob('*')):
        if p.is_file():z.write(p,str(p.relative_to(control)))
proof={'status':'PREREGISTERED_SOFTWARE_PREPARED_NOT_SUBMITTED','source_files':len(files),'source_lock_sha256':locked,
 'protocol_sha256':sha(control/'configs/phase1_v5_motion_evaluation_protocol.json'),'batch_sha256':sha(control/'submit_v5_motion_evaluation.sh'),
 'bundle_sha256':sha(ROOT/'runs/phase1_v5_motion/evaluation_source.zip'),'planned_dependency':'afterok:7000','local_software_tests':18,
 'phase1_go':False,'new_model_evaluation_executed':False}
(ROOT/'runs/phase1_v5_motion/evaluation_preparation.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof))
