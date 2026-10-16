from pathlib import Path
import json,hashlib,shutil,zipfile,datetime
ROOT=Path(__file__).resolve().parents[2];base=ROOT/'runs/phase1_v5_encoding_l4';control=base/'cache_audit_control'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p={'schema_version':1,'status':'PREREGISTERED_FUTURE_COMPLETE_CACHE_AUDIT_NO_FIT_OR_TRAINING',
 'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'encoding_job_id':7004,
 'encoding_control':'/scratch/xl4358/phase1_jepa_memory_v5_encoding_l4','encoding_root':'/scratch/xl4358/phase1_v5_real_encoding_l4/job_7004',
 'encoding_source_lock_sha256':'39fe05f2e7ddb65bf9f144b13e0cd327cf2cdf289c3c204d25ee52334280d88c',
 'encoding_protocol_sha256':'ec4a27b8d2a5f6a6442ddb9d1731501b5016194832f0f7d93b92e15c1e147345',
 'output':'/scratch/xl4358/phase1_v5_real_encoding_l4/job_7004/full_cache_audit.json',
 'required_actual_parent':'COMPLETED0:0 on declared course g2-standard-12 with actual one GPU; report success label alone insufficient',
 'checks':['All78file SHA/size/dtype/shape/finite; each before-after SHA stable','Actual source/protocol/encoder/official cached weights/metadata binding','All episode descriptors and timestamp hashes independently replayed','Exact complete train/validation participants/names/counts and no partial/extra files','No labels, basis fitting, model training or GPU inference'],
 'root_numerical_audit_not_independent_delegated_review':True,'resource':{'course':'csci_ga_2271-2026fa','partition':'n2c48m24','cpus':4,'memory_gb':16,'gpus':0,'time_limit':'00:30:00'},
 'dependency':'afterok:7004; no science overlaps','reader_submission_blocked_until_actual_cache_audit':True,'phase1_go':False}
(control/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
files=['src/phase1/__init__.py','src/phase1/utils.py','src/phase1/encoders.py','src/phase1/v5_encoding_l4.py','src/phase1/v5_real_reader.py','src/phase1/v5_real_training.py','scripts/v5_audit_l4_cache.py','scripts/v5_retention.py','tests/test_v5_cache_audit.py']
for rel in files:
 dst=control/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dst)
files+=['protocol.json']
(control/'source_lock.json').write_text(json.dumps({'files':{rel:sha(control/rel) for rel in files},'scope':'Future complete cache source/numerical attestation only, no reader fit/training'},indent=2)+'\n')
locked=sha(control/'source_lock.json')
batch=f'''#!/usr/bin/env bash
#SBATCH --job-name=cv-v5-cache-audit
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/xl4358/phase1_v5_l4_cache_audit_control/logs/audit-%j.out
#SBATCH --error=/scratch/xl4358/phase1_v5_l4_cache_audit_control/logs/audit-%j.err
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HOME=/scratch/xl4358/phase1_hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
TASK_CONTROL=/scratch/xl4358/phase1_v5_l4_cache_audit_control
cd "$TASK_CONTROL"
hostname
hostname -I
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
exec /scratch/xl4358/phase1_env/bin/python scripts/v5_audit_l4_cache.py --control "$TASK_CONTROL" --source-lock-sha {locked}
'''
(control/'submit_cache_audit.sh').write_text(batch)
with zipfile.ZipFile(base/'cache_audit_source.zip','w',zipfile.ZIP_DEFLATED) as z:
 for f in sorted(control.rglob('*')):
  if f.is_file():z.write(f,str(f.relative_to(control)))
proof={'status':'PREPARED_NOT_SUBMITTED_NOT_RUN','source_files':len(files),'source_lock_sha256':locked,
 'protocol_sha256':sha(control/'protocol.json'),'batch_sha256':sha(control/'submit_cache_audit.sh'),
 'bundle_sha256':sha(base/'cache_audit_source.zip'),'local_software_tests_passed':10,'actual_real_features_loaded':False,'dependency':'afterok:7004','phase1_go':False}
(base/'cache_audit_preparation.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof))
