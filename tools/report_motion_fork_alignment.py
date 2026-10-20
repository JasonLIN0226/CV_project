"""Local reconstruction and report of a read-only Motion fork ranking audit."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    repo=Path(__file__).resolve().parents[1];root=repo/'results/motion_fork_alignment';native=root/'native_completed'
    proof=json.loads((native/'transfer_integrity.json').read_text());assert proof['native_before_local_native_after_all_match']
    for name,v in proof['files'].items():assert hashlib.sha256((native/name).read_bytes()).hexdigest()==v['sha256']
    out=native/'output/job_8359';s=json.loads((out/'summary.json').read_text());a=json.loads((out/'independent_audit.json').read_text());assert a['status']=='PASS_INDEPENDENT_SCALAR_FORK_AUDIT'
    with (out/'paired_alignment.csv').open() as f:rows=list(csv.DictReader(f))
    ix=np.random.default_rng(0).integers(0,128,(10000,128));differences=[]
    for cell in s['cells']:
        rr=[r for r in rows if r['split']==cell['split'] and int(r['horizon'])==cell['horizon']];pairs=sorted({r['pair'] for r in rr});assert len(pairs)==128 and len(rr)==384
        for metric,w in cell['stats'].items():
            v=np.array([np.mean([float(r[metric]) for r in rr if r['pair']==p]) for p in pairs]);lo,hi=np.quantile(v[ix].mean(1),[.025,.975]);d=max(abs(v.mean()-w['mean']),abs(lo-w['low']),abs(hi-w['high']));assert d<1e-12;differences.append(d)
    assert len(differences)==102
    (root/'local_review.json').write_text(json.dumps(dict(status='PASS',statistics=102,max_difference=max(differences),GPU=0),indent=2)+'\n')
    names={'test_id':'Ordinary','test_long_gap':'Long gap','test_distractor_shift':'Distractors'}
    labels=[f"{names[x['split']]}\nh{x['horizon']}" for x in s['cells']]
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');x=np.arange(6)
    for i,(metric,label,color) in enumerate([('fraction_task_optimal','All pools','.7'),('nonflat_conditional_task_optimal','Both errors nonflat','.15')]):
        values=[c[metric] if metric=='nonflat_conditional_task_optimal' else c['stats'][metric]['mean'] for c in s['cells']]
        axes[0].bar(x+(i-.5)*.36,np.array(values)*100,.36,label=label,color=color,edgecolor='black')
    axes[0].set(ylabel='Feature-selected choices also task-optimal (%)',ylim=(0,105),title='A  Do the best deletions agree?');axes[0].legend(fontsize=8)
    for i,(metric,label,color) in enumerate([('uniform_task_regret','Uniform deletion','.7'),('feature_task_regret','Feature-selected','.15')]):
        v=[c['stats'][metric] for c in s['cells']];axes[1].bar(x+(i-.5)*.36,[z['mean'] for z in v],.36,label=label,color=color,edgecolor='black')
        axes[1].errorbar(x+(i-.5)*.36,[z['mean'] for z in v],yerr=np.array([[z['mean']-z['low'],z['high']-z['mean']] for z in v]).T,fmt='none',color='black',capsize=2)
    axes[1].set(ylabel='Position MSE above the task-optimal deletion',title='B  How costly is the disagreement?');axes[1].legend(fontsize=8)
    for ax in axes:ax.set_xticks(x,labels);ax.tick_params(axis='x',labelsize=8)
    fig.savefig(root/'fork_alignment.png',dpi=180);plt.close(fig)
    lines=['# Motion deletion-score alignment: existing forks only','','## Objective and unchanged scope','',
           'Test whether minimizing future complete-feature error chooses the same deletion as minimizing future position error. This analyzes existing scalar outcomes from the locked Motion candidate; it changes no prediction and trains no retention strategy. All original thresholds remain unchanged.', '',
           'The analysis includes all 82,944 existing rows: 3 seeds, 3 test conditions, 128 matched pairs per condition, 2 sibling episodes, 6 clock-only triples, all 3 deletions and both horizons. That is 27,648 horizon-specific decisions. Each fork retains exactly 2 past + 4 recent frames. All flat and tied pools remain.', '',
           '## What the measurements mean','',
           '- **Feature-selected:** delete the item giving the lowest complete-feature MSE at that horizon. Choices within 1e-8 of minimum are averaged, exactly as in the parent study.',
           '- **Task-optimal:** the deletion giving the lowest position MSE among those same three forks. This is an additional future-informed diagnostic lower bound, not a deployable policy or replacement acceptance criterion.',
           '- **Agreement:** the fraction of feature-tied choices that are also task-optimal (task tie tolerance 1e-12). A flat feature score receives fractional credit, not automatic perfect agreement.',
           '- **Regret:** position error of the feature-selected deletion minus the task-optimal position error. Zero is ideal; compare its size with uniform deletion regret.',
           '- **Nonflat subset:** both errors have nonzero spread beyond their fixed numerical tie tolerances. It is shown separately to expose ties; no such cases are removed from the full result.', '',
           '![Ranking agreement and position regret](../results/motion_fork_alignment/fork_alignment.png)','',
           '**Figure.** A: agreement of the best deletion, with ties handled fractionally, over all pools and the separately displayed nonflat subset. B: excess position error over the task-optimal deletion. Whiskers are paired 95% bootstrap intervals. These are descriptive diagnostics, not new pass gates.', '',
           '## Complete results','','| Condition | Horizon | Task-optimal agreement, all pools | Nonflat agreement | Feature-flat pools | Task-flat pools | Conflicting pair orderings |', '|---|---:|---:|---:|---:|---:|---:|']
    for c in s['cells']:
        st=c['stats'];lines.append(f"|{names[c['split']]}|{c['horizon']}|{100*st['fraction_task_optimal']['mean']:.2f}%|{100*c['nonflat_conditional_task_optimal']:.2f}%|{100*st['feature_flat']['mean']:.2f}%|{100*st['task_flat']['mean']:.2f}%|{100*st['pair_order_conflict']['mean']:.2f}%|")
    lines += ['', 'The conflicting-order percentage uses all three unordered comparisons inside every triple, including ties in the denominator. It is not a correlation coefficient or the fraction of completely reversed triples.', '',
              '| Condition | Horizon | Feature-selected position regret, 95% CI | Transferred-choice position regret, 95% CI | Uniform position regret | Uniform task headroom lost by feature choice |','|---|---:|---:|---:|---:|---:|']
    for c in s['cells']:
        f=c['stats']['feature_task_regret'];t=c['stats']['transfer_task_regret'];u=c['stats']['uniform_task_regret'];ratio=c['uniform_task_headroom_lost_by_feature_choice']
        lines.append(f"|{names[c['split']]}|{c['horizon']}|{f['mean']:.6f} [{f['low']:.6f}, {f['high']:.6f}]|{t['mean']:.6f} [{t['low']:.6f}, {t['high']:.6f}]|{u['mean']:.6f}|{ratio*100:.2f}%|")
    lines += ['', 'For every interval, average 6 pools and 2 siblings within each pair, then average the 3 seeds; resample 128 pairs within each split using 10,000 draws, seed 0. Intervals are descriptive and unadjusted for multiplicity. The oracle choices depend on revealed future outcomes. They are not available to a causal deployed retention policy.', '',
              '## Interpretation and limits','',
              'Long-gap is the clearest weakness. Both errors are flat in 35.42% of pools, so counting all pools gives a more favorable agreement rate. Among pools where both errors vary, feature-selected choices are task-optimal only 56.32% (h3) and 53.33% (h6) of the time, with fractional tie handling. Mean position regret is 0.001754/0.002222 and its paired interval is strictly positive. This loses 34.83%/38.58% of the task-error headroom between uniform deletion and the future-informed task-optimal choice. Equivalently, 65.17%/61.42% of that headroom is retained: the score has useful signal, but substantial misalignment remains. These ratios are descriptive, not learned-policy performance or new pass thresholds.', '',
              'Within-pool disagreement establishes that this predictor\'s complete-feature loss and this position readout are different objectives. It does not make either metric a universal importance label. Regret is needed to tell whether differing choices matter: different argmins can have nearly identical task error, while a small disagreement count can hide large losses.', '',
              'These results describe the already revealed synthetic development set, frozen predictor and simulator-specific RGB guard. They cannot identify background versus object-feature residuals, guarantee learnability from legal past inputs, or demonstrate feature-only byte-budget parity. No candidate was selected or retuned on these outcomes.', '',
              '## Verification','',
              f"The current raw SHA matches the independently audited parent before and after analysis. All original compact oracle and baseline means were reconstructed (maximum difference {s['original_compact_replay_max_difference']:.3g}). A separate implementation checked all 27,648 decision groups, all 2,304 paired rows and 102 bootstrap intervals; maximum paired difference {a['max_paired_difference']:.3g}, interval difference {a['max_interval_difference']:.3g}. Local interval replay also passed (maximum {max(differences):.3g}). Native-before/local/native-after compact checksums match. Four synthetic fixtures establish software behavior only; they are not scientific results.", '',
              '## Conclusion','',
              '**Original Phase1_GO remains unestablished.** Motion still meets 40/51 original prediction checks and 13/32 fixed-reference checks, with critical long-gap 0/8. This additional ranking diagnosis neither replaces those requirements nor authorizes a learned retention policy or fresh9137 confirmation. Retain the failures; any later method change needs a separately fixed validation-only design.', '',
              'The next minimal investigation should use original validation caches to decompose the complete-feature residual into components visible and invisible to the frozen position readout. That can test whether deletion rankings are driven by feature residuals unrelated to position. It should be read-only, with the decomposition and comparisons frozen before inspecting outputs; no retraining, metric substitution in original gates, test-driven switch or policy pilot is implied. This fork audit alone does not establish that residual explanation.', '',
              'No new training, fitting, encoder/predictor call, GPU use or GitHub update. GPU use remains 12,723/28,800 seconds, with 16,077 seconds remaining.']
    (repo/'docs/MOTION_FORK_ALIGNMENT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='LOCAL_REPORT_COMPLETE',statistics=102,max_difference=max(differences))))
if __name__=='__main__':main()
