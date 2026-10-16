"""Prepare isolated GPU reader source guards; no formal cache binding or fit."""
from pathlib import Path
import ast
import hashlib
import json
import shutil
import datetime

root = Path(__file__).resolve().parents[2]
out = root/'runs/phase1_v5_encoding_l4/real_frame_training_software'
if out.exists():
    raise ValueError('Refuse overwriting prepared reader software')
files = ['src/phase1/__init__.py', 'src/phase1/v5_real_reader.py',
         'src/phase1/v5_real_training.py', 'scripts/v5_retention.py',
         'scripts/v5_train_real.py', 'tests/test_v5_real_reader.py',
         'tests/test_v5_real_training.py']
for rel in files:
    target=out/rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root/rel, target)
path=out/'src/phase1/v5_real_training.py'
original=path.read_text()
updated=original.replace('import socket\n', 'import socket\nimport subprocess\n')
updated=updated.replace('384ef961b067c6c0d7189e02ca370a13d680c5d8e11e13df67d1fdf8b3537ca2',
                        'cd667fc4f029660c198353570686a8ed8f069a0e2dab1434e912776f5521985e')
guard='''def verify_external_cache_audit(protocol, encoding_root, expected_job_id):
    """Require actual completed7187/7188 and source-bound full-cache audit."""
    path=Path(encoding_root)/'full_cache_audit.json'
    check_regular(path)
    expected=protocol.get('actual_cache_audit_sha256')
    if not isinstance(expected,str) or len(expected)!=64 or sha_file(path)!=expected:
        raise ValueError('Actual complete cache audit must be hashed and bound before real training')
    proof=json.loads(path.read_text())
    if (expected_job_id!=7187 or protocol.get('actual_cache_audit_job_id')!=7188
            or proof.get('audit_job_id')!='7188'
            or proof.get('status')!='ACTUAL_FULL_CACHE_VERIFIED_NOT_READER_TRAINING_NOT_PHASE1_GO'
            or proof.get('complete') is not True or proof.get('encoding_job_id')!=expected_job_id
            or proof.get('source_lock_sha256')!='fc02564863f8af84a79eed2bf0861b210252667c2f70411518a32a9eeb219d00'
            or proof.get('protocol_sha256')!='51c0b25cffb63a9f7ea2e8f3a059c4ab99cc020b976d1a400d6ac167ecf78b26'
            or proof.get('encoding_source_lock_sha256')!='cd667fc4f029660c198353570686a8ed8f069a0e2dab1434e912776f5521985e'
            or proof.get('encoding_protocol_sha256')!='c456f06f572df47e66e7863ca752e22214c44846ab9be9934955f31ffa79ac7c'
            or proof.get('encoding_report_sha256')!=protocol.get('actual_encoding_report_sha256')
            or proof.get('encoder_metadata_sha256')!=protocol.get('actual_encoder_metadata_sha256')
            or proof.get('reader_fit_or_training') is not False or proof.get('labels_read') is not False
            or proof.get('sealed_data_read') is not False or proof.get('frames')!=50688
            or proof.get('recordings')!=78 or proof.get('episodes')!=198
            or len(proof.get('actual_feature_files',[]))!=78):
        raise ValueError('Actual cache audit is incomplete or incompatible')
    account=subprocess.check_output(['/opt/slurm/bin/sacct','-X','-j','7187,7188',
        '-n','-P','--format=JobIDRaw,State,ExitCode,Account,Partition,AllocTRES'],text=True)
    rows={line.split('|')[0]:line.split('|') for line in account.splitlines() if line.strip()}
    if (set(rows)!={'7187','7188'}
            or rows['7187'][1:5]!=['COMPLETED','0:0','csci_ga_2271-2026fa','c12m85-a100-1']
            or rows['7188'][1:5]!=['COMPLETED','0:0','csci_ga_2271-2026fa','n2c48m24']):
        raise ValueError('Actual encoding/audit Slurm completion differs')
    tres={part.split('=',1)[0]:part.split('=',1)[1] for part in rows['7187'][5].split(',') if '=' in part}
    if tres.get('gres/gpu')!='1':
        raise ValueError('Actual parent did not contain exactly one GPU')
    return proof


'''
updated=updated.replace('def main(control, encoding_root, encoding_control, expected_job_id, output):\n',
                        guard+'def main(control, encoding_root, encoding_control, expected_job_id, output):\n')
updated=updated.replace("or 'A100' not in torch.cuda.get_device_name(0)):",
                        "or 'A100' not in torch.cuda.get_device_name(0)\n            or os.getenv('SLURM_JOB_PARTITION') != 'c12m85-a100-1'):")
updated=updated.replace("    try:\n        episodes, cache_proof = load_verified_features(",
                        "    try:\n        verify_external_cache_audit(protocol, encoding_root, expected_job_id)\n        report['external_cache_audit_sha256']=protocol['actual_cache_audit_sha256']\n        episodes, cache_proof = load_verified_features(")
path.write_text(updated)
old_ast={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(original).body if isinstance(n,ast.FunctionDef)}
new_ast={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(updated).body if isinstance(n,ast.FunctionDef)}
changed=[name for name in old_ast if old_ast[name]!=new_ast[name]]
assert set(changed)=={'main','load_verified_features'}
assert 'verify_external_cache_audit' in new_ast
record={'status':'ISOLATED_SOFTWARE_ONLY_NOT_FORMAL_PROTOCOL_OR_SUBMISSION',
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'reason':'Original reader hard-coded old encoder source lock; FRAME cache source and actual completed audit must be guarded before future fit',
        'changes':['Strict expected encoder source lock updated to actual7187 FRAME source',
                   'Strict actual7188 audit/source/report/metadata and both completed Slurm jobs before fit',
                   'Original A100 guard additionally requires actual A100 partition'],
        'all_original_math_model_fit_loss_batch_optimizer_schedule_epochs_checkpoint_functions_AST_equal':True,
        'only_changed_original_functions':changed,'new_function':'verify_external_cache_audit',
        'original_active_and_pending_source_edited':False,'actual_features_read_fit_or_training':False,
        'formal_protocol_written':False,'new_science_submitted':False}
(out/'preparation.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
