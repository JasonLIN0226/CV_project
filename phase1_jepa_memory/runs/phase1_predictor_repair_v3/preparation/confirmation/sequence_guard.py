"""Preparation only: conservative full-confirmation cost guard, no submission.

The caller must obtain the ledger from native sacct and freeze its source,
permit, cost measurements and complete12model plan before fresh data access.
"""
import math
from pathlib import Path
from phase1.resource_guard import elapsed_from_allocation
from phase1.utils import file_hash, read_json


class FreshSequenceGuard:
    status='NATIVE_RECONCILED_COMPLETE_CONFIRMATION_GUARD'
    cap_seconds=28800
    def __init__(self, *, settled_seconds, native_ledger, candidate_epoch_seconds,
                 reference_epoch_seconds, encoder_frames_per_second,
                 encoder_setup_seconds, actual_charged_now, permit):
        if (not native_ledger or sum(r['elapsed_raw'] for r in native_ledger)!=settled_seconds
                or any(r['state'] not in ('COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY')
                    or r['account']!='csci_ga_2271-2026fa' or r['gpus']!=1
                    or not isinstance(r['elapsed_raw'],int) or r['elapsed_raw']<0 for r in native_ledger)
                or len({r['job_id'] for r in native_ledger})!=len(native_ledger)):
            raise ValueError('Complete actual terminal GPU ledger required')
        self.settled_seconds=settled_seconds;self.ledger=native_ledger
        self.rates={'candidate':dict(candidate_epoch_seconds),'reference':dict(reference_epoch_seconds)}
        self.throughput=float(encoder_frames_per_second);self.setup=float(encoder_setup_seconds)
        for role in self.rates:
            if set(self.rates[role])!={'gate','motion'}:raise ValueError('Both full task rates required')
        if not all(math.isfinite(x) and x>0 for x in [self.throughput,self.setup,*self.rates['candidate'].values(),*self.rates['reference'].values()]):
            raise ValueError('Finite positive native rates required')
        self.actual_charged_now=actual_charged_now;self.permit=Path(permit)
        self.counts={(task,seed,role):0 for task in ('gate','motion') for seed in (17,29,43) for role in ('reference','candidate')}
        self.encoded_frames=0;self.frame_bound=198656

    def check(self, allocation_start, task, seed, role, retained_epochs):
        # Native callback includes current Slurm wall AND any previous
        # confirmation allocation/proof/failure seconds, not merely epoch time.
        charged=self.actual_charged_now()
        if (not isinstance(charged,(int,float)) or not math.isfinite(charged)
                or charged<elapsed_from_allocation(allocation_start)):
            raise ValueError('Full native new-allocation wall accounting required')
        key=(task,seed,role)
        if key not in self.counts or self.counts[key]!=retained_epochs:
            raise ValueError('Actual retained epoch coverage differs')
        remaining_training=sum((120-n)*self.rates[r][t] for (t,s,r),n in self.counts.items())
        remaining_encoding=(self.frame_bound-self.encoded_frames)/self.throughput
        projected=self.settled_seconds+charged+1.15*(remaining_training+remaining_encoding)+900+2*self.setup
        if projected>self.cap_seconds or self.settled_seconds+charged>=self.cap_seconds:
            raise RuntimeError('Complete confirmation cannot fit authorized cumulative8h; preserve partial, never shorten')
        return {'projected_cumulative_seconds':projected,'cap':self.cap_seconds,'cost_prediction_only':True,'phase1_go':False}

    def record_epoch(self,task,seed,role,epoch,seconds):
        key=(task,seed,role)
        if key not in self.counts or epoch!=self.counts[key]+1 or epoch>120:
            raise ValueError('Incomplete/duplicate/nonsequential actual epoch')
        if not math.isfinite(seconds) or seconds<=0:raise ValueError('Actual full epoch timing required')
        self.counts[key]=epoch;self.rates[role][task]=max(self.rates[role][task],seconds)

    def record_encoded_frames(self,total):
        if not isinstance(total,int) or not self.encoded_frames<=total<=self.frame_bound:
            raise ValueError('Complete bounded unique actual encoder frame ledger required')
        self.encoded_frames=total

    def require_completed_reference(self,path,signature):
        p=read_json(self.permit);lock=read_json(p['references_lock'])
        if (lock['status']!='ALL_SIX_FRESH_REFERENCES_LOCKED' or len(lock['models'])!=6
                or signature['permit_sha256']!=file_hash(self.permit)
                or signature['role']!='reference' or self.counts[(signature['task'],signature['seed'],'reference')]!=120):
            raise ValueError('All original fresh references must be completed before candidates')
        for name,expected in lock['files'].items():
            if file_hash(name)!=expected:raise ValueError('Frozen fresh reference/transform changed')
        if lock['files'].get(str(Path(path)))!=file_hash(path):
            raise ValueError('Candidate received an unregistered fresh reference')
