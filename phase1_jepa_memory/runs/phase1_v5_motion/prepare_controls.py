from pathlib import Path
import hashlib, json, shutil, zipfile, datetime
ROOT=Path(__file__).resolve().parents[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
control=ROOT/'runs/phase1_v5_motion/control'
control.mkdir(exist_ok=False)
old=json.loads((ROOT/'runs/phase1_v5_gate/control/source_lock.json').read_text())
replace={'src/phase1/v5_synthetic_training.py':'src/phase1/v5_motion_training.py',
 'scripts/v5_train_synthetic.py':'scripts/v5_train_motion.py',
 'tests/test_v5_synthetic_training.py':'tests/test_v5_motion_training.py',
 'configs/phase1_v5_gate_training_protocol.json':'configs/phase1_v5_motion_training_protocol.json'}
files=[]
for rel,want in old['files'].items():
    if rel in replace:
        rel=replace[rel]
    elif sha(ROOT/rel)!=want:
        raise ValueError('Frozen shared source changed: '+rel)
    files.append(rel)
files.append('runs/phase1_v3_dev/motion/reference/resolved_config.json')
for rel in files:
    dst=control/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dst)
lock={'files':{rel:sha(control/rel) for rel in files},'scope':'Independent Motion reader-only trial; Gate active source and original Motion data/inputs/basis/loss/gates unchanged.'}
(control/'source_lock.json').write_text(json.dumps(lock,indent=2)+'\n')
locked=sha(control/'source_lock.json')
batch=(ROOT/'runs/phase1_v5_gate/control/submit_v5_gate.sh').read_text().replace('cv-v5-gate','cv-v5-motion').replace('phase1_jepa_memory_v5_gate','phase1_jepa_memory_v5_motion').replace('phase1_v5_gate_development','phase1_v5_motion_development').replace('v5_train_synthetic.py','v5_train_motion.py').replace('9e2fec4cc7a2b526a275b53f6a96c6180cfe470004f805310e6075ffdb5f9830',locked)
(control/'submit_v5_motion.sh').write_text(batch)
with zipfile.ZipFile(ROOT/'runs/phase1_v5_motion/source.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(control.rglob('*')):
        if p.is_file():z.write(p,str(p.relative_to(control)))
review={'status':'PREREGISTERED_SOFTWARE_PREPARED_NOT_SUBMITTED','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'task':'motion','new_models':3,'seeds':[17,29,43],'epochs':120,'reader_sha256':sha(ROOT/'src/phase1/v5_synthetic_reader.py'),
 'source_lock_sha256':locked,'protocol_sha256':sha(control/'configs/phase1_v5_motion_training_protocol.json'),
 'batch_sha256':sha(control/'submit_v5_motion.sh'),'bundle_sha256':sha(ROOT/'runs/phase1_v5_motion/source.zip'),
 'source_files':len(files),'local_software_tests_passed':18,'software_tests_are_not_science':True,
 'original_two_frame_policy':True,'basis_refit':False,'source_output_separate':True,'planned_dependency':'afterok:6994',
 'first_epoch_fixed_budget_gate':True,'original_critical_slice':'test_long_gap','critical_absolute_checks':8,'fixed_reference_recent_checks':24,
 'initial_preflight_findings':'Static diff caught an unconverted reference dictionary key; repaired before tests/deployment. Default local Python3.14 lacks pytest; tests passed in existing Anaconda runtime. No HPC science affected.',
 'phase1_go':False,'gate_failed_results_cannot_be_rescued_by_motion':True}
(ROOT/'runs/phase1_v5_motion/preparation.json').write_text(json.dumps(review,indent=2)+'\n')
print(json.dumps(review))
