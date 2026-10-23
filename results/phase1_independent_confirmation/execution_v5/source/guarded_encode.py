"""Original batch2 encoder with accounting outside the numerical path.

Structural cache audits are deferred to the following CPU stage and remain
mandatory. No complete audit is fabricated; original/staged strict loaders keep
rejecting a cache until those CPU audits actually finish.
"""
import time
from pathlib import Path
from phase1 import cache
from phase1.data import require_audit
from phase1.utils import read_json,write_json,file_hash
from staged_data import bind,require_stage,stage_load_index,verify_permit
from budget_guard import TOTAL_FRAMES,BudgetStop


def encode(c,permit,sequence_guard,allocation_start,*,training_only):
    verify_permit(c,permit,models_locked=not training_only)
    root=Path(c['output']); began=time.perf_counter();frames=0
    last_check=0.;last_report=None;stream_began=None
    def budget_check(force=False):
        nonlocal last_check,last_report
        now=time.time()
        if force or now-last_check>=30:
            p=root/'features/index.json'
            if p.exists():
                ix=read_json(p)
                committed=sum(v['shape'][0] for v in ix['episodes'].values())
                if committed>TOTAL_FRAMES:raise BudgetStop('Encoded complete frame bound exceeded')
                sequence_guard.state['encoded_frames']=committed
            elapsed=0. if stream_began is None else time.perf_counter()-stream_began
            if frames>=64 and elapsed>0:
                measured=frames/elapsed
                sequence_guard.state['encoder_frames_per_second']=min(
                    sequence_guard.state.get('encoder_frames_per_second',measured),measured)
            sequence_guard.save()
            last_report=sequence_guard.check(allocation_start)
            last_check=now
        if last_report is not None and last_report['remaining_cap_seconds']-(now-last_check)<=120:
            raise BudgetStop('Actual shared allocation wall is close to the hard8h cap')
    class Encoder:
        def __init__(self,base):self.base=base;self.metadata=base.metadata
        def encode(self,rgb):
            nonlocal frames,stream_began
            if stream_began is None:stream_began=time.perf_counter()
            budget_check()
            result=self.base.encode(rgb)
            frames+=len(rgb)
            return result
    def build(config,device):
        # First guard before loading the large frozen checkpoint.
        budget_check(force=True)
        import torch
        protocol=read_json(Path(__file__).resolve().parent/'protocol.json')
        if torch.cuda.get_device_name(0)!=protocol['expected_GPU'] or torch.__version__!=protocol['expected_torch']:
            raise ValueError('Inherited encoder rate requires the identical native A100/Torch runtime')
        encoder_cache=read_json(Path(__file__).resolve().parent/'preflight_cpu.json')['encoder_cache']
        for name,info in encoder_cache['files'].items():
            path=Path(name);st=path.stat()
            if (str(path.resolve()),st.st_size,st.st_mtime_ns)!=(info['resolved_path'],info['bytes'],info['mtime_ns']):
                raise ValueError('SHA-verified frozen encoder object changed before load')
        base=cache.build_encoder(config,device)
        if base.metadata!=protocol['encoder_metadata']:
            raise ValueError('Actual frozen encoder/checkpoint/processor metadata differs')
        floating=0
        for tensor in base.model.state_dict().values():
            if tensor.is_floating_point():
                floating+=1
                if not bool(torch.isfinite(tensor).all()):raise ValueError('Actual loaded I-JEPA floating tensor is nonfinite')
        if floating==0:raise ValueError('No actual pretrained floating weights loaded')
        write_json(root/('encoder_loaded_train.json' if training_only else 'encoder_loaded_test.json'),
            dict(actual_float_tensors_verified=floating,all_finite=True,
                official_cache_object_SHA_proof=True,metadata_exact=True,
                actual_GPU=torch.cuda.get_device_name(0),actual_torch=torch.__version__))
        return Encoder(base)
    def defer_audit(config,manifest,index):
        write_json(root/'features'/'pending_CPU_pair_audit.json',dict(
            complete_audit=False,stage='TRAIN_VALIDATION' if training_only else 'FULL_FINAL',
            index_sha256=file_hash(root/'features/index.json'),
            original_pair_audit_required_before_any_store_or_training=True))
    fn=bind(cache.encode,build_encoder=build,audit_features=defer_audit,
        require_audit=require_stage if training_only else require_audit,
        load_index=stage_load_index if training_only else cache.load_index)
    budget_check(force=True)
    result=fn(c)
    budget_check(force=True)
    write_json(root/('train_encoding_complete.json' if training_only else 'test_encoding_complete.json'),
        dict(index_sha256=file_hash(root/'features/index.json'),encoded_this_stage=frames,
             seconds_observed=time.perf_counter()-began,batch_size=2,
             unaltered_original_encoder=True,CPU_audit_pending=True))
    return result
