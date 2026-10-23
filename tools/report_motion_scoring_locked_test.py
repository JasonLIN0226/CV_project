"""Report one prelocked score regression; keep original feature and task criteria."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def main():
    repo=Path(__file__).resolve().parents[1];root=repo/'results/motion_scoring_locked_test';native=root/'native_completed';job=json.loads((root/'status.json').read_text())['job'];out=native/f'output/job_{job}'
    proof=json.loads((native/'transfer_integrity.json').read_text());assert proof['native_before_local_native_after_all_match']
    for name,v in proof['files'].items():assert hashlib.sha256((native/name).read_bytes()).hexdigest()==v['sha256']
    s=json.loads((out/'summary.json').read_text());a=json.loads((out/'independent_audit.json').read_text());assert a['status']=='PASS_INDEPENDENT_LOCKED_SCORING_AUDIT';stats=s['statistics']
    with (out/'paired_scoring.csv').open() as f:rows=list(csv.DictReader(f))
    indices=np.random.default_rng(0).integers(0,128,(10000,128));diffs=[]
    def values(spec,col):
        rr=[r for r in rows if r['split']==spec['split'] and int(r['horizon'])==spec['horizon'] and (spec['seed']=='mean' or int(r['seed'])==int(spec['seed']))];ids=sorted({r['pair'] for r in rr});assert len(ids)==128
        return np.array([np.mean([float(r[col]) for r in rr if r['pair']==p]) for p in ids])
    def check(v,w):
        lo,hi=np.quantile(v[indices].mean(1),[.025,.975]);d=max(abs(v.mean()-w['mean']),abs(lo-w['low']),abs(hi-w['high']));assert d<2e-12;diffs.append(d)
    for x in stats['comparisons']:
        for m,w in x['metrics'].items():check(values(x,x['baseline']+'_'+m)-values(x,x['arm']+'_'+m),w['paired_gain'])
    for x in stats['ranking']:
        check(values(x,x['arm']+'_task_optimal'),x['agreement']);check(values(x,x['arm']+'_task_regret'),x['regret'])
    for x in stats['direct']:
        for m,w in x['full_minus_decoded'].items():check(values(x,'full_'+x['selection']+'_'+m)-values(x,'decoded_'+x['selection']+'_'+m),w)
    assert len(diffs)==864
    (root/'local_review.json').write_text(json.dumps(dict(status='PASS',statistics=864,max_difference=max(diffs),GPU=0),indent=2)+'\n')
    names={'test_id':'Ordinary','test_long_gap':'Long gap','test_distractor_shift':'Distractors'};order=[(sp,h) for sp in names for h in (3,6)]
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained');x=np.arange(6)
    for i,(arm,label,color) in enumerate([('full_same','Full feature score','.7'),('decoded_same','Decoded consistency','.1')]):
        rr=[next(z for z in stats['ranking'] if z['seed']=='mean' and z['arm']==arm and (z['split'],z['horizon'])==(sp,h)) for sp,h in order];v=[z['agreement'] for z in rr]
        axes[0].bar(x+(i-.5)*.36,[z['mean']*100 for z in v],.36,label=label,color=color,edgecolor='black')
        axes[0].errorbar(x+(i-.5)*.36,[z['mean']*100 for z in v],yerr=np.array([[z['mean']-z['low'],z['high']-z['mean']] for z in v]).T*100,fmt='none',color='black',capsize=2)
    axes[0].set(xticks=x,xticklabels=[f'{names[sp]}\nh{h}' for sp,h in order],ylabel='Task-optimal deletion agreement (%)',ylim=(0,105),title='A  Position alignment on revealed test splits');axes[0].legend(fontsize=8);axes[0].tick_params(axis='x',labelsize=8)
    counts=[stats['transfer_both_positive'][k] for k in ('full_transfer','decoded_transfer')];bars=axes[1].bar(['Full feature score','Decoded consistency'],counts,color=['.7','.1'],edgecolor='black');axes[1].set(ylim=(0,80),ylabel='Original both-metric comparisons met (out of 72)',title='B  Preserve BOTH original outcome metrics')
    for bar,n in zip(bars,counts):axes[1].text(bar.get_x()+bar.get_width()/2,n+1,f'{n}/72',ha='center')
    axes[1].axhline(72,color='black',ls='--',lw=1);fig.savefig(root/'scoring_regression.png',dpi=180);plt.close(fig)
    full,decoded=counts
    lines=['# Motion: one locked scoring-formula regression test','','## Objective and boundary','',
           'Test whether the validation-supported, task-specific decoded-consistency score repairs deletion ranking without hiding errors in the original complete-feature outcome. The predictors, RGB guard, normalization, decoder, capacity and original thresholds are unchanged. The only change is the offline future-informed score used to select a deletion. Formula: `mean(((prediction - normalized_future_feature) @ frozen_position_decoder_A)**2)`.', '',
           'In plain language, the new score compares predicted and future-feature-decoded ball positions, instead of allowing every feature coordinate to vote equally on what to delete. The fixed decoder was fitted earlier; it was not refitted here. Ordinary uses the original delay range, long gap extends the interval with the ball hidden, and distractor shift changes irrelevant visual content. h3/h6 mean predicting 3/6 simulator steps after the current frame.', '',
           'This is a prelocked regression check on previously revealed 6301 development test data, NOT fresh independent confirmation. No coefficient, crop, model or scoring switch was tuned after seeing this test. Future targets are scoring labels only; neither scorer is a deployable past-only policy.', '',
           '## Complete experiment','',
           'All 128 matched pairs in each ordinary, long-gap and distractor-shift split; both siblings; seeds 17/29/43; six original clock-only triples; all three deletions; horizons 3 and 6: **82,944 outcomes**, reduced to 2,304 paired rows. Every fork retains 2 past + 4 recent frames. Existing encoder caches and unchanged original outcome values are reused. A frozen CPU predictor pass supplies missing decoded forecasts; no training, encoder run or GPU is used.', '',
           'The two scoring rules select at the same horizon or at the OTHER horizon. Both are evaluated with the original complete-feature MSE and true-position MSE against uniform deletion, FIFO and reject-newest. The original cross-horizon rule remains a positive paired 95% lower confidence bound in BOTH outcome metrics. Individual seeds and seed means remain separate; their comparison counts share scenes.', '',
           '![Task alignment and both-metric coverage](../results/motion_scoring_locked_test/scoring_regression.png)', '',
           '**Figure.** A shows agreement with the task-optimal future-informed deletion; ties receive fractional credit. B checks whether selecting at the other horizon still improves BOTH original errors over each baseline. Better position alignment alone cannot replace B or the original prediction requirements.', '',
           '## 1. Original cross-horizon comparisons','','| Score | Both original metrics positive / total |','|---|---:|',f'|Full feature score|{full}/72|',f'|Decoded consistency|{decoded}/72|','',
           'The count is 3 splits × 2 evaluated horizons × 3 baselines × 4 seed estimates. The seed-mean estimate reuses the same three trained models and scenes; it is not an additional independent experiment.', '',
           '## 2. Same-horizon deletion alignment, seed means','','| Condition | Horizon | Original score agreement | Decoded score agreement | Original position regret | Decoded position regret |','|---|---:|---:|---:|---:|---:|']
    for sp,h in order:
        rr={z['arm']:z for z in stats['ranking'] if z['split']==sp and z['horizon']==h and z['seed']=='mean'}
        lines.append(f"|{names[sp]}|{h}|{100*rr['full_same']['agreement']['mean']:.2f}%|{100*rr['decoded_same']['agreement']['mean']:.2f}%|{rr['full_same']['regret']['mean']:.8f}|{rr['decoded_same']['regret']['mean']:.8f}|")
    lines += ['', 'Position regret is excess position MSE over the best of the same three deletions, using future truth for this diagnostic lower bound only. No pool, seed or tie is dropped.', '',
              '## 3. Cross-horizon decoded score: all seed-mean outcome comparisons','','| Condition | Horizon | Baseline | Complete-feature gain, 95% CI | Position gain, 95% CI | Both positive |','|---|---:|---|---:|---:|---|']
    for z in stats['comparisons']:
        if z['seed']!='mean' or z['arm']!='decoded_transfer':continue
        f=z['metrics']['feature_mse']['paired_gain'];t=z['metrics']['task_error']['paired_gain'];lines.append(f"|{names[z['split']]}|{z['horizon']}|{z['baseline']}|{f['mean']:.6f} [{f['low']:.6f}, {f['high']:.6f}]|{t['mean']:.6f} [{t['low']:.6f}, {t['high']:.6f}]|{z['positive_both_metrics']}|")
    lines += ['', 'Positive gains mean lower original outcome error than the baseline. All failures, including individual seeds, follow below.', '',
              '### Every decoded cross-horizon comparison missing the original both-metric rule','','| Split | Horizon | Seed estimate | Baseline | Feature interval | Position interval |','|---|---:|---|---|---:|---:|']
    failed=[z for z in stats['comparisons'] if z['arm']=='decoded_transfer' and not z['positive_both_metrics']]
    for z in failed:
        f=z['metrics']['feature_mse']['paired_gain'];t=z['metrics']['task_error']['paired_gain'];lines.append(f"|{names[z['split']]}|{z['horizon']}|{z['seed']}|{z['baseline']}|[{f['low']:.6f}, {f['high']:.6f}]|[{t['low']:.6f}, {t['high']:.6f}]|")
    if not failed:lines.append('|None|—|—|—|All positive|All positive|')
    lines += ['', '## 4. Direct tradeoff against the original scoring rule','','| Condition | Horizon | Same/other-horizon selection | Full-minus-decoded feature error | Full-minus-decoded position error |','|---|---:|---|---:|---:|']
    for z in stats['direct']:
        if z['seed']!='mean':continue
        f=z['full_minus_decoded']['feature_mse'];t=z['full_minus_decoded']['task_error'];lines.append(f"|{names[z['split']]}|{z['horizon']}|{z['selection']}|{f['mean']:.6f} [{f['low']:.6f}, {f['high']:.6f}]|{t['mean']:.6f} [{t['low']:.6f}, {t['high']:.6f}]|")
    lines += ['', 'Positive numbers favor decoded consistency; negative numbers favor the original full-feature score. In all 12 seed-mean same/other-horizon cells, decoded scoring lowers position error but raises full-feature error relative to the original feature-score choice; every respective interval excludes zero. Both scores still improve both errors over the simple baselines in the comparisons reported above. This is a measured tradeoff, not universal dominance.', '',
              '## Independent verification','',f"All 82,944 original outcome rows remain exact copies of the audited parent; fresh frozen inference replay differs by at most {s['max_original_scalar_replay_difference']:.3g}. The decoded-score vector/coordinate identity differs by at most {a['score_identity_max_difference']:.3g}. All 324 fixed complete-vector witnesses match the parent (maximum {a['witness_parent_max_difference']:.3g}). Independent raw-choice reconstruction, all 2,304 paired rows and 864 scalar intervals passed; maximum paired difference {a['max_paired_difference']:.3g}, statistic difference {a['max_statistic_difference']:.3g}. Original full-score statistics reproduce within {a['original_full_score_statistic_max_difference']:.3g}. Local 864-interval replay passed (maximum {max(diffs):.3g}). Native-before/local/native-after compact hashes match. Large arrays and raw rows remain on HPC.", '',
              'All intervals use the unchanged 10,000-draw paired bootstrap, seed 0. Average pools and siblings within each pair, then average model seeds for the seed-mean estimate; resample 128 pairs within each split. Ranking and direct-tradeoff intervals are descriptive; no new acceptance gate or multiplicity claim is introduced.', '',
              '## Conclusion','']
    if decoded==72:lines += ['Decoded consistency meets all 72 original cross-horizon both-metric comparisons in this locked revealed-development regression check. This supports the task-specific offline scoring hypothesis in this scope; it does not prove a deployable retention policy or general task-agnostic utility.']
    else:lines += [f'Decoded consistency meets {decoded}/72 original cross-horizon both-metric comparisons. Keep the remaining failures and tradeoffs. Do not retune the score on these test results or claim complete scoring acceptance from favorable position alignment alone.']
    lines += ['', '**Overall Phase1_GO remains unestablished.** Independent recomputation of the unchanged original prediction requirements still gives Motion 40/51 prediction checks, 13/32 fixed-reference checks and 0/8 critical long-gap checks. Gate retains its separately supported candidate. The new score does not change prediction outputs or satisfy those missing requirements. No retention-policy training, fresh9137 confirmation, model retraining or GitHub update.', '',
              'The score relies on an existing supervised position decoder and future feature targets for offline labels; it is task-specific and does not establish past-only learnability. The renderer-specific RGB guard and unproven feature-only byte-budget parity also remain. GPU use is unchanged at 12,723/28,800 seconds, leaving 16,077 seconds.']
    (repo/'docs/MOTION_SCORING_LOCKED_TEST.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(status='LOCAL_REPORT_COMPLETE',intervals=864,full=full,decoded=decoded)))
if __name__=='__main__':main()
