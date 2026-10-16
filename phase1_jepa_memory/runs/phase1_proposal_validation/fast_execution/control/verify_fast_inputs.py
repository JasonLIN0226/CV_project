"""Full actual CPU data equivalence audit. No model forward or fitting."""
import argparse,time
from pathlib import Path
import torch
from phase1.proposal_trial import verify_control
from phase1.exact_runtime_cache import CachedFeatureStore
from phase1.model import fit_basis
from phase1.training import paired_rows
from phase1.utils import file_hash,write_json

parser=argparse.ArgumentParser();parser.add_argument('--control',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
torch.set_num_threads(1);began=time.perf_counter();p,lock,configs=verify_control(args.control)
proof={'source_lock_sha256':file_hash(Path(args.control)/'source_lock.json'),'tasks':{},'all_exact':False,'no_model_forward':True,'no_held_out_inputs':True,'threads':1}
for task,c in configs.items():
    store=CachedFeatureStore(c,max_cached=64)
    basis=fit_basis(c,store,task);store.bases={task:basis}
    result=store.verify_all_inputs()
    pairs=paired_rows([r for r in store.manifest['records'] if r['task']==task and r['split']=='validation'])
    store.prepare_validation(pairs,'cpu');result['original_validation_batches_exact']=store.verify_validation_inputs(pairs)
    assert result['exact_samples_targets_info_rng']==4864 and result['original_validation_batches_exact']==12
    proof['tasks'][task]=result
    write_json(args.output,proof)
    del store
proof.update(all_exact=True,seconds=time.perf_counter()-began)
write_json(args.output,proof);print(proof)
