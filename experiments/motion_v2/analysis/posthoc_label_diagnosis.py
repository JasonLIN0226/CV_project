"""事后探索性分析，未预登记。

CPU-only summaries of exported Motion v2.1 deletion labels. No model imports,
checkpoints, frame rendering, episode files, fitting of retention students, or
held-out scenes. envs_v2.scene reconstructs train/validation hidden factors
solely for this privileged explanatory analysis.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import PIL

NOTICE = "事后探索性分析，未预登记"
ROOT = Path(__file__).resolve().parents[3]
GPU_REL = Path("results/motion_v2/step3_4_recovery1/terminal_evidence/repo/results/motion_v2/step3_4_recovery1/gpu")
MANIFEST_REL = Path("results/motion_v2/v2_2/step1/data/dataset/manifest.json")
MANIFEST_SHA = "9ac43b81eb75456e17f3542e64e8a0ab58282207752ae3712836f6d2ea20032a"
LABELS = ("local_feature", "local_position", "use_time_feature", "use_time_position", "future_attention")
TYPES = ("O", "S", "P", "E")
NEW_TYPES = frozenset(("P", "E"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def frame_type(scene, frame):
    if not 0 <= frame <= scene["query"] - 4:
        raise ValueError("Parent must contain only already departed history frames")
    if not scene["visible"][frame]:
        return "O"
    if frame == scene["end"]:
        return "E"
    if scene["t_c"] <= frame < scene["end"]:
        return "P"
    if frame < scene["t_c"]:
        return "S"
    raise ValueError("Unexpected visible frame outside old/new intervals")


def composition(types):
    return tuple(types.count(kind) for kind in TYPES)


def candidate_keys(types, pool):
    if len(pool) != 3 or len(set(pool)) != 3:
        raise ValueError("Expected three distinct parent timestamps")
    comp = composition(types)
    ranks = {frame: rank for rank, frame in enumerate(sorted(pool))}
    return [(kind, ranks[frame], comp) for kind, frame in zip(types, pool)]


def targets(raw, scales=None):
    """Identical NumPy arithmetic and final FP32 cast to step34_students.py."""
    fitted, probabilities = {}, {}
    for label in LABELS:
        values = np.asarray(raw[label], dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
            raise ValueError("Invalid three-candidate labels")
        centered = values - values.mean(1, keepdims=True)
        fitted[label] = max(float(np.sqrt(np.mean(centered**2))), 1e-8) if scales is None else scales[label]
        logits = -centered / fitted[label]
        logits -= logits.max(1, keepdims=True)
        exp = np.exp(logits)
        probabilities[label] = (exp / exp.sum(1, keepdims=True)).astype(np.float32)
    return probabilities, fitted


def entropy(q):
    q = np.asarray(q, dtype=np.float64)
    return float(-np.sum(q * np.log(np.clip(q, 1e-300, 1.)), axis=1).mean())


def type_reference(train_keys, validation_keys, train_q, validation_q):
    """Train-only target means, then normalize the three retrieved scores."""
    sums, counts = defaultdict(float), Counter()
    for keys, row in zip(train_keys, train_q):
        for key, probability in zip(keys, row):
            sums[key] += float(probability)
            counts[key] += 1
    means = {key: sums[key] / counts[key] for key in sums}
    scores = np.asarray([[means.get(key, 1. / 3.) for key in keys] for keys in validation_keys], dtype=np.float64)
    unseen = np.asarray([[key not in means for key in keys] for keys in validation_keys])
    prediction = scores / scores.sum(1, keepdims=True)
    q = np.asarray(validation_q, dtype=np.float64)
    ce = float(-np.sum(q * np.log(np.clip(prediction, 1e-300, 1.)), axis=1).mean())
    return {
        "validation_soft_ce": ce, "uniform_soft_ce": math.log(3),
        "validation_target_entropy": entropy(q), "train_target_entropy": entropy(train_q),
        "improvement_over_uniform": math.log(3) - ce,
        "excess_over_target_entropy": ce - entropy(q),
        "train_keys": len(means), "unseen_validation_candidate_keys": int(unseen.sum()),
        "validation_rows_with_unseen_key": int(unseen.any(1).sum()),
        "unseen_key_fallback": "1/3 before parentwise renormalization",
        "key_means": [{"type": key[0], "temporal_rank_oldest_0": key[1],
                       "composition_O_S_P_E": list(key[2]), "train_count": counts[key],
                       "mean_train_target_probability": means[key]}
                      for key in sorted(means)],
    }


def two_new_alignment(costs, type_rows):
    new = np.asarray([[kind in NEW_TYPES for kind in row] for row in type_rows])
    mask = new.sum(1) == 2
    selected = np.asarray(costs)[mask]
    nonnew = np.argmax(~new[mask], axis=1)
    if not len(selected):
        raise ValueError("No eligible two-new-frame parents")
    good = selected[np.arange(len(selected)), nonnew]
    other = selected.copy()
    other[np.arange(len(other)), nonnew] = np.inf
    margin = other.min(1) - good
    choices = selected.argmin(1)
    minima = selected == selected.min(1, keepdims=True)
    return {
        "parents": len(selected), "argmin_deletes_nonnew_count": int((choices == nonnew).sum()),
        "argmin_deletes_nonnew_rate": float((choices == nonnew).mean()),
        "random_deletion_reference": 1. / 3.,
        "exact_tie_parents": int((minima.sum(1) > 1).sum()),
        "argmin_tie_rule": "first exported candidate slot (numpy.argmin)",
        "tie_averaged_deletes_nonnew_rate": float((minima[np.arange(len(minima)), nonnew] / minima.sum(1)).mean()),
        "median_best_competing_new_drop_minus_nonnew_drop": float(np.median(margin)),
        "median_mean_new_drop_minus_nonnew_drop": float(np.median((selected.sum(1) - good) / 2 - good)),
        "median_cost_range": float(np.median(np.ptp(selected, axis=1))),
        "positive_margin_meaning": "keeping the new-frame pair costs less than the alternative",
    }


def split_summary(records, types, raw):
    counts = Counter(composition(row) for row in types)
    nnew = np.asarray([sum(kind in NEW_TYPES for kind in row) for row in types])
    informative = (nnew >= 1) & (nnew <= 2)
    result = {
        "parents": len(records), "weighting": "exported parent multiplicity; both siblings and all behavior/clock rows retained",
        "composition_distribution": [{"composition_O_S_P_E": list(key), "display": "".join(k * n for k,n in zip(TYPES,key)),
                                      "count": count, "proportion": count / len(types)} for key,count in sorted(counts.items())],
        "new_frame_count_distribution": {str(n): int((nnew == n).sum()) for n in range(4)},
        "at_least_one_new": {"count": int((nnew >= 1).sum()), "proportion": float((nnew >= 1).mean())},
        "informative": {"definition": "one or two P/E and at least one O/S", "count": int(informative.sum()), "proportion": float(informative.mean())},
        "cost_ranges": {}, "two_new_frame_alignment": {},
    }
    for label in LABELS[:-1]:
        span = np.ptp(raw[label], axis=1)
        result["cost_ranges"][label] = {
            "informative_median": float(np.median(span[informative])),
            "remaining_median": float(np.median(span[~informative])),
            "all_median": float(np.median(span)),
            "exact_zero_range_count": int((span == 0).sum()),
            "unit": "raw feature/position MSE, before train-RMS scaling",
        }
        result["two_new_frame_alignment"][label] = two_new_alignment(raw[label], types)
    return result


def compare_reference(result):
    checks = []
    def add(name, actual, expected, decimals):
        checks.append({"name": name, "actual": actual, "reference": expected,
                       "reference_decimals": decimals, "difference": actual-expected,
                       "matches_reference_rounding": round(actual, decimals) == expected})
    for split, percent, agreements in [("train",33.2,[.68,.69,.69,.64]),("validation",34.7,[.52,.61,.52,.56])]:
        add(split+"_at_least_one_new_percent",100*result["splits"][split]["at_least_one_new"]["proportion"],percent,1)
        for label,value in zip(LABELS,agreements):
            add(split+"_"+label+"_pair_preservation", result["splits"][split]["two_new_frame_alignment"][label]["argmin_deletes_nonnew_rate"],value,2)
    for label,value in zip(LABELS,[1.090,1.094,1.103,1.094,.726]):
        add(label+"_hidden_type_validation_ce",result["hidden_type_reference"][label]["validation_soft_ce"],value,3)
    return {"purpose":"external rounded reference check only; not a selection or scientific gate",
            "all_numeric_values_match_rounding":all(x["matches_reference_rounding"] for x in checks),"checks":checks}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args=parser.parse_args(argv)
    output=args.output.resolve()
    if output.exists():
        raise FileExistsError("Keep earlier exploratory outputs; use a distinct output directory")
    began=time.perf_counter()
    input_hashes={}
    gpu=ROOT/GPU_REL
    inventory=json.loads((gpu/'artifact_sha256.json').read_text())
    def checked(path):
        rel=str(path.relative_to(ROOT));digest=sha(path)
        if path.is_relative_to(gpu):
            expected=inventory[str(path.relative_to(gpu))]
            if digest != expected: raise ValueError("Export hash changed: "+rel)
        input_hashes[rel]=digest
        return path
    def read(path): return json.loads(checked(path).read_text())
    config=read(ROOT/'experiments/motion_v2/config_v2_1.json')
    manifest=read(ROOT/MANIFEST_REL)
    if input_hashes[str(MANIFEST_REL)] != MANIFEST_SHA:
        raise ValueError("Original manifest hash changed")
    env_path=ROOT/'experiments/motion_v2/envs_v2.py'
    checked(env_path)
    if sha(env_path) != manifest['source_hashes']['envs_v2.py']:
        raise ValueError("Original scene implementation changed")
    spec=importlib.util.spec_from_file_location('posthoc_envs_v2',env_path)
    env=importlib.util.module_from_spec(spec);spec.loader.exec_module(env)
    env.validate_config(config)
    index={r['id']:r for r in manifest['records']}
    if len(index)!=1216 or Counter(r['split'] for r in index.values()) != {'train':1024,'validation':192}:
        raise ValueError("Only the original train/validation manifest is allowed")
    scenes={}
    raw, records, type_rows, keys={}, {}, {}, {}
    annotated=[]
    for split, expected_rows in [('train',16384),('validation',3072)]:
        states=read(gpu/'labels'/split/'states.json')
        records[split]=states['records']
        if states['split']!=split or len(records[split])!=expected_rows:
            raise ValueError("Unexpected parent split or count")
        with np.load(checked(gpu/'labels'/split/'labels.npz'),allow_pickle=False) as z:
            if set(z.files)!=set(LABELS):raise ValueError("Expected all five raw label arrays")
            raw[split]={name:z[name].copy() for name in LABELS}
        type_rows[split],keys[split]=[],[]
        for state in records[split]:
            rec=index[state['episode_id']]
            if rec['split']!=split or rec['pair']!=state['pair_id']:
                raise ValueError("Parent/manifest identity mismatch")
            key=(split,rec['scene_index'])
            if key not in scenes:scenes[key]=env.scene(config,rec['task'],split,rec['scene_index'])
            s=scenes[key]
            if any(s[k]!=rec[k] for k in ('query','gap','seed','length') if k in rec):
                raise ValueError("Reconstructed scene disagrees with manifest")
            if s['length']!=rec['frames']:raise ValueError("Frame count mismatch")
            expected_pair=env.digest([config['data']['seed'],'motion',split,rec['scene_index']])[:24]
            if expected_pair!=state['pair_id']:raise ValueError("Pair reconstruction mismatch")
            if len(state['pool'])!=3 or max(state['pool'])>state['clock']-4:
                raise ValueError("Parent contains a frame not yet departed from recent")
            kinds=[frame_type(s,t) for t in state['pool']]
            type_rows[split].append(kinds)
            keys[split].append(candidate_keys(kinds,state['pool']))
            annotated.append({'analysis_classification':NOTICE,'split':split,'parent_id':state['id'],
                              'episode_id':state['episode_id'],'pair_id':state['pair_id'],
                              'scene_index':rec['scene_index'],'clock':state['clock'],'pool':state['pool'],
                              'types':kinds,'composition_O_S_P_E':list(composition(kinds))})
    probabilities, scales=targets(raw['train'])
    validation_probabilities,_=targets(raw['validation'],scales)
    saved_scales=read(gpu/'students/label_scales.json')['scales']
    if any(scales[name]!=saved_scales[name] for name in LABELS):
        raise ValueError("Train-RMS does not exactly reproduce the stored training scales")
    result={'analysis_classification':NOTICE,'analysis_date':'2026-11-09','data_scope':'existing ordinary train/validation parents only',
        'definitions':{'types':{'O':'occluded','S':'visible before t_c','P':'visible t_c <= t < end','E':'visible at end'},
                       'new_segment':'P or E','key':'candidate type + temporal rank (oldest=0) + unordered O/S/P/E counts',
                       'reference_prediction':'train-only key mean target, normalized across each validation parent; unseen key=1/3 before normalization',
                       'nondeployable_hidden_reference':True,'target_probability':'FP32 softmax(-(cost-parent_mean)/train_RMS), identical arithmetic to frozen training',
                       'cost_gap':'min(cost deleting either P/E) - cost deleting O/S; positive favors keeping P/E pair'},
        'splits':{s:split_summary(records[s],type_rows[s],raw[s]) for s in ('train','validation')},
        'train_rms':scales,
        'hidden_type_reference':{name:type_reference(keys['train'],keys['validation'],probabilities[name],validation_probabilities[name]) for name in LABELS},
        'alignment_train_minus_validation':{},'training_curves':[], 'saved_posthoc_best':{},
        'limitations':['Deletion labels on train are produced by predictors trained on those same train episodes.',
                       'Hidden-type reference is privileged posthoc explanation, not a deployable retention policy.',
                       'Parent rows share episodes, siblings and behavior trajectories; row counts are not independent sample counts.',
                       'Label inconsistency and sparse informative parents support a noise-dominated-label explanation, not causal identification of a unique training failure.'],
        'execution':{'device':'CPU','new_a100_seconds':0,'cumulative_a100_seconds':5886,'unused_a100_seconds':8514,
                     'gpu_jobs_submitted':0,'scene_calls':len(scenes),'episode_or_render_calls':0,'model_calls':0,
                     'student_training_calls':0,'test_or_motion_confirmation_opened_or_generated':False}}
    for name in LABELS[:-1]:
        result['alignment_train_minus_validation'][name]=result['splits']['train']['two_new_frame_alignment'][name]['argmin_deletes_nonnew_rate']-result['splits']['validation']['two_new_frame_alignment'][name]['argmin_deletes_nonnew_rate']
    for architecture in ('set','independent'):
        for name in LABELS:
            for seed in (17,29,43):
                h=read(gpu/'students'/f'{architecture}_{name}_seed{seed}_history.json')
                train=h['training'];val=h['validation']
                if [v['step'] for v in val]!=list(range(0,1001,100)) or len(train)!=1000:
                    raise ValueError("Incomplete saved training curve")
                blocks=[float(np.mean([v['soft_cross_entropy'] for v in train[start:start+100]])) for start in range(0,1000,100)]
                values=[v['own_validation_label_cross_entropy'] for v in val]
                result['training_curves'].append({'architecture':architecture,'label':name,'seed':seed,
                    'selected_step':val[int(np.argmin(values))]['step'],'train_100_step_block_means':blocks,
                    'train_last_block_below_first':blocks[-1]<blocks[0],
                    'train_block_means_strictly_decreasing':bool(np.all(np.diff(blocks)<0)),
                    'validation_steps':[v['step'] for v in val],'validation_soft_ce':values,
                    'validation_step100_above_step0':values[1]>values[0],
                    'validation_all_post100_above_step0':all(v>values[0] for v in values[1:]),
                    'validation_strictly_increasing_after_step100':bool(np.all(np.diff(values[1:])>0)),
                    'training_note':'train curve uses saved minibatch losses, not full-train reevaluation; not necessarily monotone per update'})
    costs=[r for r in result['training_curves'] if r['label']!='future_attention']
    result['cost_curve_summary']={'arms':len(costs),**{key:sum(r[key] for r in costs) for key in (
        'train_last_block_below_first','train_block_means_strictly_decreasing','validation_step100_above_step0',
        'validation_all_post100_above_step0','validation_strictly_increasing_after_step100')},
        'selected_step0':sum(r['selected_step']==0 for r in costs)}
    for split in ('ordinary','long_gap'):
        summary=read(gpu/'online'/split/'summary.json')
        result['saved_posthoc_best'][split]=[dict(row,minimum_position_mse_among_saved_methods=row['position_mse']['mean']==min(x['position_mse']['mean'] for x in summary['table'] if x['horizon']==row['horizon'])) for row in summary['table'] if row['method']=='reference/posthoc_best']
    result['reference_comparison']=compare_reference(result)
    result['environment']={'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'Pillow':PIL.__version__}
    result['provenance']={'input_sha256':input_hashes,'script_sha256':sha(__file__),
                          'frozen_training_source_sha256':sha(ROOT/'experiments/motion_v2/step34_students.py')}
    if any(sha(ROOT/path)!=digest for path,digest in input_hashes.items()):raise ValueError("Input modified during analysis")
    result['execution']['cpu_wall_seconds']=time.perf_counter()-began
    output.mkdir(parents=True,exist_ok=False)
    write_new(output/'diagnosis.json',result)
    write_new(output/'parent_types.json',{'analysis_classification':NOTICE,'analysis_only_never_model_input':True,'records':annotated})
    write_new(output/'status.json',{'analysis_classification':NOTICE,'status':'completed_cpu_posthoc_motion_v2_1_closed',
        'completed_utc':datetime.now(timezone.utc).isoformat(),'reference_values_match_rounding':result['reference_comparison']['all_numeric_values_match_rounding'],
        'motion_retention_comparison_closed':True,'confirmation_opened':False,'gpu_jobs_submitted':0,
        'a100_seconds_added':0,'cumulative_a100_seconds':5886,'unused_a100_seconds':8514,
        'original_execution_records_modified':False,'inputs_sha256_unchanged':True})
    sums=''.join(sha(p)+'  '+p.name+'\n' for p in sorted(output.iterdir()) if p.is_file())
    with (output/'SHA256SUMS').open('x') as f:f.write(sums)
    print(json.dumps({'analysis_classification':NOTICE,'output':str(output),'reference_check':result['reference_comparison'],
                      'cost_curve_summary':result['cost_curve_summary'],'cpu_seconds':result['execution']['cpu_wall_seconds']},ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
