import argparse, datetime, hashlib, json, os, socket, subprocess, sys, time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--lock-sha',required=True)
args=p.parse_args()
root=Path(__file__).resolve().parent
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
lock=root/'source_lock.json'
assert sha(lock)==args.lock_sha
files=json.loads(lock.read_text())['files']
for name,want in files.items():
    path=root/name
    assert not any(x.is_symlink() for x in (path,*path.parents))
    assert sha(path)==want,name
job=subprocess.check_output(['/opt/slurm/bin/squeue','-h','-j','7000','-o','%i|%a|%T|%N'],text=True).strip()
assert job=='7000|csci_ga_2271-2026fa|RUNNING|'+socket.gethostname()
sys.path.insert(0,str(root/'src'));sys.path.insert(0,str(root/'scripts'))
import torch,pytest
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
start=time.monotonic()
code=pytest.main(['-q','-p','no:cacheprovider','--basetemp',str(root/'artificial_test_temp'),
    str(root/'tests/test_v5_real_reader.py'),str(root/'tests/test_v5_real_training_l4.py')])
for name,want in files.items():assert sha(root/name)==want,name
proof={'status':'HPC_SOFTWARE_CHECK_ONLY_NO_REAL_FEATURES_NO_SCIENCE_SUBMISSION',
    'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'host':socket.gethostname(),'actual_allocation_checked':job,
    'source_lock_sha256':args.lock_sha,'source_files':len(files),
    'test_exit_code':code,'expected_tests':15,'elapsed_seconds':time.monotonic()-start,
    'torch_threads':torch.get_num_threads(),'actual_features_loaded':False,
    'scientific_fit_or_training':False,'new_scientific_protocol_created':False,
    'new_scientific_job_submitted':False,'phase1_go':False,
    'tests_scope':'Artificial CPU fixtures only: future isolation, padding, train-only fit, full-feature gradients, missing/wrong external cache proof'}
(root/'software_test_proof.json').write_text(json.dumps(proof,indent=2)+'\n')
sys.exit(code)
