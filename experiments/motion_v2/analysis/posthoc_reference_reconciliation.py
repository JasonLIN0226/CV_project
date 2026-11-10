"""事后探索性分析，未预登记：audit reported-reference discrepancies.

The authoritative diagnosis keeps the frozen negative-score deletion target.
One explicitly different positive-score retention target is computed only to
check whether the external CE reference may use the opposite target direction.
This is not a replacement result, a student fit, or another scientific run.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

SPEC=importlib.util.spec_from_file_location('posthoc',Path(__file__).with_name('posthoc_label_diagnosis.py'))
D=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(D)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-output',required=True,type=Path)
    args=parser.parse_args()
    root=args.analysis_output.resolve()
    original=json.loads((root/'diagnosis.json').read_text())
    annotations=json.loads((root/'parent_types.json').read_text())['records']
    original_sha=D.sha(root/'diagnosis.json')
    keys={s:[D.candidate_keys(r['types'],r['pool']) for r in annotations if r['split']==s] for s in ['train','validation']}
    q={}
    inputs={}
    for split in ['train','validation']:
        p=D.ROOT/D.GPU_REL/'labels'/split/'labels.npz'
        digest=D.sha(p);inputs[str(p.relative_to(D.ROOT))]=digest
        if digest!=original['provenance']['input_sha256'][str(p.relative_to(D.ROOT))]:
            raise ValueError('Original label input changed')
        with np.load(p,allow_pickle=False) as z:
            value=z['future_attention']
        logits=(value-value.mean(1,keepdims=True))/original['train_rms']['future_attention']
        logits-=logits.max(1,keepdims=True)
        exp=np.exp(logits)
        q[split]=(exp/exp.sum(1,keepdims=True)).astype(np.float32)
    alternative=D.type_reference(keys['train'],keys['validation'],q['train'],q['validation'])
    authoritative=original['hidden_type_reference']['future_attention']
    exact=original['splits']['train']['two_new_frame_alignment']['use_time_position']
    result={'analysis_classification':D.NOTICE,'purpose':'reference-definition audit only; never replaces frozen training-target diagnosis',
        'authoritative_future_attention':{'target':'softmax(-centered_attention/train_RMS), deletion direction used in training',
            'validation_ce':authoritative['validation_soft_ce'],'target_entropy':authoritative['validation_target_entropy'],
            'external_reference':.726,'reference_below_actual_target_entropy':.726<authoritative['validation_target_entropy']},
        'opposite_direction_sensitivity':{'target':'softmax(+centered_attention/train_RMS), retention direction NOT used in training',
            'validation_ce':alternative['validation_soft_ce'],'target_entropy':alternative['validation_target_entropy'],
            'matches_0_726_after_rounding':round(alternative['validation_soft_ce'],3)==.726,
            'not_an_adopted_target_or_rerun':True},
        'train_use_time_position':{'correct':exact['argmin_deletes_nonnew_count'],'parents':exact['parents'],
            'exact_rate':exact['argmin_deletes_nonnew_rate'],'single_round_2dp':round(exact['argmin_deletes_nonnew_rate'],2),
            'double_round_3dp_then_2dp':round(round(exact['argmin_deletes_nonnew_rate'],3),2),
            'external_reference':.64,'reference_rounding_procedure_unknown':True},
        'original_diagnosis_sha256':original_sha,'input_sha256':inputs,'script_sha256':D.sha(__file__),
        'gpu_jobs_submitted':0,'model_calls':0,'training_calls':0,'test_or_confirmation_used':False}
    if D.sha(root/'diagnosis.json')!=original_sha:raise ValueError('Diagnosis modified')
    D.write_new(root/'reference_reconciliation.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
