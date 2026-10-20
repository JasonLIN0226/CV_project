"""Explain the predeclared validation-only residual diagnosis; no candidate edits."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCORES=('full','full64','visible','null','basis_sensitive','basis_null','readout_consistency','uniform','task_optimal')
NAMES={'full':'Original full features','full64':'Full features (float64 control)','visible':'Position-visible feature energy','null':'Position-null feature energy','basis_sensitive':'Position-sensitive within basis','basis_null':'Position-null within basis','readout_consistency':'Decoded future-feature consistency','uniform':'Uniform deletion','task_optimal':'True-position oracle (lower bound)'}

def main():
    repo=Path(__file__).resolve().parents[1];root=repo/'results/motion_residual_decomposition';native=root/'native_completed';job=json.loads((root/'status.json').read_text())['job'];out=native/f'output/job_{job}'
    transfer=json.loads((native/'transfer_integrity.json').read_text());assert transfer['native_before_local_native_after_all_match']
    for name,v in transfer['files'].items():assert hashlib.sha256((native/name).read_bytes()).hexdigest()==v['sha256']
    s=json.loads((out/'summary.json').read_text());a=json.loads((out/'independent_audit.json').read_text());assert a['status']=='PASS_INDEPENDENT_SVD_RESIDUAL_AND_RANK_AUDIT'
    with (out/'paired_components.csv').open() as f:rows=list(csv.DictReader(f))
    ix=np.random.default_rng(0).integers(0,96,(10000,96));deltas=[]
    for cell in s['cells']:
        rr=[r for r in rows if int(r['horizon'])==cell['horizon']];pairs=sorted({r['pair'] for r in rr});assert len(pairs)==96 and len(rr)==288
        for key,w in cell['stats'].items():
            v=np.array([np.mean([float(r[key]) for r in rr if r['pair']==p]) for p in pairs]);lo,hi=np.quantile(v[ix].mean(1),[.025,.975]);delta=max(abs(v.mean()-w['mean']),abs(lo-w['low']),abs(hi-w['high']));assert delta<2e-12;deltas.append(delta)
    assert len(deltas)==128
    (root/'local_review.json').write_text(json.dumps(dict(status='PASS',statistics=128,max_difference=max(deltas),GPU=0),indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(13,4.1),layout='constrained');x=np.arange(2)
    base=np.zeros(2)
    for key,label,color in [('basis_sensitive','Sensitive inside basis','.15'),('basis_null','Null inside basis','.55'),('outside_basis','Outside predictor basis','.9')]:
        values=np.array([c['stats'][key+'_mean']['mean'] for c in s['cells']]);axes[0].bar(x,values,bottom=base,color=color,edgecolor='black',label=label);base+=values
    axes[0].set(title='A  Where is the feature error?',ylabel='Mean normalized feature MSE');axes[0].legend(fontsize=7)
    for i,(key,label,color) in enumerate([('basis_sensitive','Sensitive inside basis','.15'),('basis_null','Null inside basis','.55'),('outside_basis','Outside predictor basis','.9')]):
        values=[c['stats'][key+'_range']['mean'] for c in s['cells']];axes[1].bar(x+(i-1)*.25,values,.25,color=color,edgecolor='black',label=label)
    axes[1].set(title='B  Which parts vary across deletions?',ylabel='Mean range within a three-deletion pool')
    for i,(key,label,color) in enumerate([('full','Full features','.75'),('visible','Visible energy','.45'),('readout_consistency','Decoded consistency','.1')]):
        values=[c['stats'][key+'_task_regret'] for c in s['cells']];axes[2].bar(x+(i-1)*.25,[v['mean'] for v in values],.25,color=color,edgecolor='black',label=label)
        axes[2].errorbar(x+(i-1)*.25,[v['mean'] for v in values],yerr=np.array([[v['mean']-v['low'],v['high']-v['mean']] for v in values]).T,fmt='none',color='black',capsize=2)
    axes[2].set(title='C  Does a component change selection?',ylabel='Position MSE above task-optimal deletion');axes[2].legend(fontsize=7)
    for ax in axes:ax.set_xticks(x,['h3','h6'])
    fig.savefig(root/'residual_decomposition.png',dpi=180);plt.close(fig)
    lines=['# Motion validation: which feature residuals change deletion rankings?','','## Objective and fixed data','',
           'Separate large but deletion-invariant feature error from error components that actually change deletion choices. This is a cache-only diagnosis using all 96 original Motion validation pairs, both siblings, three frozen seeds, six clock-only triple pools, every deletion and both horizons: 20,736 cached predictions and 6,912 horizon-specific choice problems. No test arrays, fresh9137, model/encoder call, fitting, training or policy execution is used.', '',
           'The observed-input guard and frozen predictors are unchanged. Every projection and diagnostic score below was specified before inspecting these outputs. All use future targets for OFFLINE scoring; they are not causal policies. No original pass criterion is changed.', '',
           '## Decomposition in plain language','',
           'The frozen position readout maps a 5,120-dimensional normalized feature vector z to two coordinates: zA+b. For the residual e=prediction−target, project onto the two directions spanned by A. The **visible** residual can affect this decoder; the orthogonal **null** residual cannot. Their squared lengths add exactly to complete-feature error. Null means invisible to this particular linear decoder, not useless for every visual task.', '',
           'A second, orthogonal partition uses the frozen 128-dimensional predictor output basis: position-sensitive directions inside that basis, position-null directions inside the basis, and everything outside it. Large outside-basis target error can be constant across all deletion choices. If so, it raises absolute prediction MSE but does not drive deletion ranking.', '',
           'Finally, decoded feature consistency is ||eA||²/2. It keeps the decoder\'s coordinate scaling, whereas visible feature energy uses the Euclidean norm in feature space. The actual position error also includes target-decoder calibration error: if c=targetA+b−true_position, then position MSE=||eA||²/2+||c||²/2+(eA)·c. This checks whether the decoder\'s own target error explains remaining disagreement.', '',
           '![Residual components and ranking consequences](../results/motion_residual_decomposition/residual_decomposition.png)', '',
           '**Figure.** A shows average error energy. B shows how much each component changes among the three deletion choices within the same pool. A large bar in A alone cannot explain ranking errors; the component must vary in B. C asks how much position error remains after choosing the lowest score. Lower is better; the reference is the future-informed true-position-optimal deletion. Whiskers are paired descriptive 95% intervals.', '',
           '## 1. Error energy and deletion variation','','| Horizon | Readout-visible energy | Readout-null energy | Outside-basis energy | Null fraction of total | Outside-basis fraction |', '|---|---:|---:|---:|---:|---:|']
    for c in s['cells']:
        st=c['stats'];lines.append(f"|{c['horizon']}|{st['visible_mean']['mean']:.8f}|{st['null_mean']['mean']:.8f}|{st['outside_basis_mean']['mean']:.8f}|{100*c['null_energy_fraction']:.4f}%|{100*c['outside_basis_energy_fraction']:.4f}%|")
    lines += ['', '| Horizon | Sensitive-within-basis range | Null-within-basis range | Outside-basis range | Original/float64 choice disagreement |','|---|---:|---:|---:|---:|']
    for c in s['cells']:
        st=c['stats'];lines.append(f"|{c['horizon']}|{st['basis_sensitive_range']['mean']:.8g}|{st['basis_null_range']['mean']:.8g}|{st['outside_basis_range']['mean']:.8g}|{100*st['full_float64_choice_differs']['mean']:.4f}%|")
    lines += ['', 'Ranges do not add: maxima/minima can occur at different deletion choices. This table is not a percentage decomposition of ranking variance.', '',
              '## 2. Which component overrides useful position information?','','| Horizon | Feature/task conflicting orderings | Null-overrides-visible among conflicts | Basis-null-overrides-sensitive among conflicts | Visible energy itself conflicts, all comparisons |','|---|---:|---:|---:|---:|']
    for c in s['cells']:
        st=c['stats'];u=c['conflict_explained_by_null_override'];v=c['conflict_explained_by_basis_null_override'];lines.append(f"|{c['horizon']}|{100*st['order_conflict']['mean']:.2f}%|{100*u:.2f}%|{100*v:.2f}%|{100*st['visible_itself_conflicts']['mean']:.2f}%|")
    lines += ['', 'A null override here means that, among two deletion choices with opposite original-feature/task preferences, the position-visible component favors the lower-position-error choice while the null component favors the other one. It is a signed decomposition of the observed disagreement. The counterfactual component scores below quantify the choice consequence; no predictor output is altered. Stored float32 scores remain authoritative for original ranking; the float64 minimum-choice control changes no choices in this validation set.', '',
              '## 3. All predeclared diagnostic scoring rules','','| Scoring rule | Horizon | Task-optimal choice agreement | Position regret, 95% CI |','|---|---:|---:|---:|']
    for name in SCORES:
        for c in s['cells']:
            st=c['stats'];v=st[name+'_task_regret'];lines.append(f"|{NAMES[name]}|{c['horizon']}|{100*st[name+'_task_optimal']['mean']:.2f}%|{v['mean']:.8f} [{v['low']:.8f}, {v['high']:.8f}]|")
    lines += ['', 'Agreement is fractional when several scores tie. All flat/tied pools remain. Feature-energy tie tolerance is 1e-8; coordinate-error tie tolerance is 1e-12. These are numerical tie conventions, not new acceptance thresholds.', '',
              '### Seed coverage for the repair hypothesis', '',
              '| Seed | Horizon | Original full-score agreement | Decoded-consistency agreement | Original position regret | Decoded-consistency position regret |', '|---|---:|---:|---:|---:|---:|']
    for seed in (17,29,43):
        for h in (3,6):
            rr=[r for r in rows if int(r['seed'])==seed and int(r['horizon'])==h]
            val=lambda k:float(np.mean([float(r[k]) for r in rr]))
            lines.append(f"|{seed}|{h}|{100*val('full_task_optimal'):.2f}%|{100*val('readout_consistency_task_optimal'):.2f}%|{val('full_task_regret'):.8f}|{val('readout_consistency_task_regret'):.8f}|")
    lines += ['', 'These are per-seed descriptive means on the same 96 validation pairs, not independent datasets or additional pass checks.', '',
              '## 4. Decoder scaling and target calibration','',f"Position-decoder singular values: {s['decoder_singular_values']}; condition number {s['decoder_condition_number']:.4f}. Coordinate consistency accounts for this scaling; orthogonal visible energy alone does not.", '',
              '| Horizon | Decoded feature residual MSE | Target-decoder calibration MSE | Cross term | Reconstructed position MSE |','|---|---:|---:|---:|---:|']
    for c in s['cells']:
        st=c['stats'];lines.append(f"|{c['horizon']}|{st['readout_consistency_mean']['mean']:.8f}|{st['target_readout_error_mean']['mean']:.8f}|{st['task_cross_mean']['mean']:.8f}|{st['task_reconstructed_mean']['mean']:.8f}|")
    lines += ['', 'These terms average all forks, not only the feature-selected fork. Positive and negative cross terms are retained; calibration can matter even if its standalone MSE is small.', '',
              '## Verification and statistical unit','',
              f"Every stored forecast was reconstructed from the same cached baseline and unchanged geometry/guard rule. Maximum original-scalar replay difference {s['original_scalar_replay_max_difference']:.3g}; maximum additive-energy identity discrepancy {s['energy_identity_max_difference']:.3g}. An independent SVD-based implementation reconstructs every vector component and guard correction (maximum difference {a['component_max_difference']:.3g}), checks null-space leakage ({a['null_readout_leak_max']:.3g}), and independently rebuilds all rankings/576 paired rows/128 intervals. Local 128-interval replay also passed (maximum {max(deltas):.3g}). Compact native-before/local/native-after hashes match; all large vectors and raw component rows remain on HPC.", '',
              f"The first analysis stopped at a source-replay guard: a float64 decomposition energy was compared directly with a legacy float32 fallback scalar (first difference 1.09e-7). That failure is retained. A separately declared, hashed repair reproduces the original float32 arithmetic for source replay and records the float64 difference separately (maximum {s.get('float64_vs_source_max_difference',0):.3g}). The original tolerance, source ranking scores, cached predictions and decomposition calculations are unchanged. No acceptance rule was relaxed or additional allocation created.", '',
              'The bootstrap averages six pools and two siblings inside each pair, then averages three seeds. It resamples the 96 original validation pairs 10,000 times with seed 0. All 95% percentile intervals are descriptive, without multiplicity correction or a new pass rule. Five adversarial software fixtures verify numerical identities and known ranking failure cases only.', '',
              '## Repair boundary','',
              'The measured finding is specific: 99.86–99.88% of full residual energy is invisible to the fixed position readout. More importantly for ranking, 86.49–87.35% of conflicting deletion orderings have null and visible components voting in opposite directions as described above. Within the predictor basis, null-energy ranges are 0.0644/0.0706, whereas outside-basis ranges are only about 4e-10. Therefore large outside-basis error is effectively deletion-invariant here; the varying within-basis null residual is the stronger scoring interference candidate. These components cannot be labeled background, color or texture without further evidence.', '',
              'Simply projecting to visible feature energy raises task-optimal agreement to 88.70%/89.16%. Accounting for the frozen decoder\'s coordinate scaling with decoded consistency raises it to 96.12%/96.27%; position regret falls by 99.79%/99.73% relative to original full-score selection on validation. The gain appears in all three seeds. Residual target-readout calibration and ties remain, so this is not perfect agreement.', '',
              '**Recommended minimal repair hypothesis:** freeze the existing predictor and decoder, and use the fixed offline scoring formula `mean(((prediction - future_feature) @ A)**2)` for the Motion deletion-utility diagnostic. Compare it with the unchanged full-feature score on a separately locked test. No mixing coefficient, new training or test-selected switch is needed. Future features are scoring labels only; they must never enter a deployed past-only scorer.', '',
              'This is a task-specific readout-assisted diagnostic. It is not yet a general task-agnostic JEPA utility, a learned retention strategy, or proof of past-only learnability. Keep the original full-feature prediction/acceptance metrics alongside it. Do not replace those gates with the favorable position score. This scoring hypothesis does not repair Motion\'s original absolute full-feature prediction failure, and it needs its own locked test before adoption.', '',
              '**Original Phase1_GO remains unestablished:** Motion 40/51 original prediction checks, 13/32 fixed-reference checks and 0/8 critical long-gap checks. No gate, model or retained sample changes; no fresh confirmation or GitHub update. GPU use remains 12,723/28,800 seconds, leaving 16,077 seconds.']
    (repo/'docs/MOTION_RESIDUAL_DECOMPOSITION.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(status='LOCAL_REPORT_COMPLETE',statistics=128,max_difference=max(deltas))))
if __name__=='__main__':main()
