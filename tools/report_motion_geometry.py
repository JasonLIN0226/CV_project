"""Render retained independent Motion validation diagnostics; no model execution."""
import json,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    repo=Path(__file__).resolve().parents[1]
    status=json.loads((repo/'results/motion_geometry/status.json').read_text())
    native=repo/'results/motion_geometry/native_completed'
    job=status['job'];out=native/f'output/job_{job}'
    s=json.loads((out/'summary.json').read_text());a=json.loads((out/'independent_audit.json').read_text())
    receipt=json.loads((native/'execution_receipt.json').read_text())
    assert a['status']=='PASS' and receipt['status']=='COMPLETE_WITH_INDEPENDENT_AUDIT'
    transfer=json.loads((native/'transfer_integrity.json').read_text())
    assert transfer['native_before_local_native_after_all_match']
    assert all(hashlib.sha256((native/k).read_bytes()).hexdigest()==v['sha256'] for k,v in transfer['files'].items())
    cells={(x['horizon'],x['method']):x for x in a['statistics']}
    methods=['hold_last_visible','recent_only','privileged_small_budget','full_history','kinematic_actions','oracle_position_only','oracle_velocity_only','oracle_both','true_target_readout']
    names=['Hold visible','Recent predictor','Two-frame predictor','Full-history predictor','Decoded geometry','Oracle position','Oracle velocity','Oracle both','True-feature readout']
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    ax=axes[0]
    for offset,m,color,label in [(-.18,'privileged_small_budget','.75','Frozen two-frame predictor'),(.18,'kinematic_actions','.15','Decoded geometry')]:
        vals=[cells[h,m] for h in (3,6)];ys=[1000*v['mean'] for v in vals]
        bars=ax.bar([i+offset for i in range(2)],ys,.34,color=color,edgecolor='black',label=label,yerr=[[1000*(v['mean']-v['low']) for v in vals],[1000*(v['high']-v['mean']) for v in vals]],capsize=3)
        ax.bar_label(bars,fmt='%.3f',padding=18,fontsize=9)
    ax.set(xticks=[0,1],xticklabels=['3-step future','6-step future'],ylabel='Position MSE × 1,000 (lower is better)',ylim=(0,9.5),title='A  Is usable motion information present?')
    ax.legend(fontsize=8,loc='upper left')
    ax=axes[1];keys=['kinematic_actions','oracle_position_only','oracle_velocity_only','true_target_readout'];labels=['Decoded geometry','Replace initial position','Replace velocity','Decode true future feature']
    vals=[cells[6,m] for m in keys];ys=[1000*v['mean'] for v in vals]
    bars=ax.barh(range(4),ys,color=['.15','white','white','white'],edgecolor='black',xerr=[[1000*(v['mean']-v['low']) for v in vals],[1000*(v['high']-v['mean']) for v in vals]],capsize=3)
    ax.bar_label(bars,fmt='%.4f',padding=24,fontsize=9)
    ax.set(yticks=range(4),yticklabels=labels,xlabel='Position MSE × 1,000 (lower is better)',xlim=(0,.28),title='B  What limits geometry? (6-step future)');ax.invert_yaxis()
    fig.suptitle('Frozen Motion validation: same two observed historical frames\n96 base pairs; white bars use scoring truth and are not model inputs',fontsize=12)
    fig.savefig(repo/'results/motion_geometry/position_diagnosis.png',dpi=180);plt.close(fig)
    lines=['# Frozen Motion validation geometry diagnosis','',f'One CPU job {job}; no training, fitting, new encoding or GPU. All 96 original validation pairs (192 episodes), three frozen models and two horizons were evaluated. The run produced 1,152 rows; baseline replay max difference {s["baseline_replay_max_difference"]:.3g}. Independent vector/axis/geometric replay and compact transfer hashes passed.','', '## What was tested','', 'The black bar uses decoded positions in the two prescribed visible historical frames, timestamps and known actions to propagate motion through the unchanged simulator dynamics. It has a known-physics prior and is an explanatory diagnostic, not a new learned predictor. Gray bars are frozen-predictor/hold-position comparisons. Oracle position/velocity controls use simulator truth only in a separate scoring path; they are not valid normal model inputs. True-feature readout measures a diagnostic decoding limitation.','', '![Position diagnosis](../results/motion_geometry/position_diagnosis.png)','', 'Error bars are 95% percentile intervals from 2,000 bootstrap samples of 96 independent base pairs. Seeds are averaged, then siblings within each pair; repeated geometry across seeds is not extra evidence. Lower is better. These are position errors, not full 5,120-dimensional feature losses or new acceptance gates.','', '| Method | Horizon 3 position MSE [95%CI] | Horizon 6 position MSE [95%CI] |','|---|---:|---:|']
    for m,n in zip(methods,names):
        vals=[cells[h,m] for h in [3,6]]
        lines.append('|'+n+'|'+'|'.join(f'{v["mean"]:.6g} [{v["low"]:.6g}, {v["high"]:.6g}]' for v in vals)+'|')
    d=a['visible_readout_diagnostics'];lines+=['','## Visible evidence and controls','',f'Visible-start position MSE: {d["visible_start_position_MSE"]:.6g}; visible-end MSE: {d["visible_end_position_MSE"]:.6g}; decoded visible-velocity RMSE: {d["visible_velocity_RMSE"]:.6g}. No new decoder was fitted. Oracle-both maximum transition reconstruction discrepancy: {s["oracle_transition_max_difference"]:.3g}.','', 'Paired differences below are comparator MSE minus decoded-geometry MSE; positive favors decoded geometry. Confidence intervals compare identical basepairs, not separate unpaired means.','', '| Comparator minus decoded geometry | Horizon 3 difference [95%CI] | Horizon 6 difference [95%CI] |','|---|---:|---:|']
    for m in ['hold_last_visible','recent_only','privileged_small_budget','full_history','oracle_position_only','oracle_velocity_only','oracle_both']:
        vals=[cells[h,m+'_minus_kinematic_actions'] for h in [3,6]]
        lines.append('|'+m+'|'+'|'.join(f'{v["mean"]:.6g} [{v["low"]:.6g}, {v["high"]:.6g}]' for v in vals)+'|')
    gains=[100*(1-cells[h,'kinematic_actions']['mean']/cells[h,'privileged_small_budget']['mean']) for h in (3,6)]
    velgain=100*(1-cells[6,'oracle_velocity_only']['mean']/cells[6,'kinematic_actions']['mean'])
    lines+=['','## Interpretation and the next repair boundary','',f'Using positions decoded from the same two observed frames and a known transition law reduces position MSE by {gains[0]:.2f}% at horizon 3 and {gains[1]:.2f}% at horizon 6 relative to the frozen two-frame predictor. Paired improvement intervals are positive. On this validation set, the observations contain useful motion information; missing evidence alone cannot explain the predictor deficit. This diagnostic has extra structural knowledge of the simulator and does not identify one unique faulty predictor branch.','',f'Within the geometry diagnostic, replacing decoded velocity with scoring-truth velocity reduces horizon 6 MSE by {velgain:.2f}%; replacing only initial position produces a smaller improvement. Thus accumulated velocity-estimation error is the larger remaining limitation of this particular geometry control. These oracle interventions are separate, non-additive sensitivities, not a causal percentage decomposition of the original predictor error. There were no decoded positions outside the arena and no geometric reflections in this validation run, so reflection handling cannot be validated empirically from these scientific examples; only software fixtures exercised it.','', 'Known-action versus no-action geometry differs only modestly on this validation distribution; this is not permission to drop actions in other conditions. The near-zero oracle-both error checks time indexing and simulator replay, not research success.','', 'The next possible repair must produce legitimate complete visual features from allowed observed inputs, while preserving Gate and all original Motion tests. A coordinate-only output or projection optimized merely to satisfy the position readout would not establish reliable deletion utility or the full-feature criteria. First design a validation-only, frozen feature-consistency check; do not repeat a broad predictor training search or switch to a revealed-test-selected branch. No further repair was executed in this task.','', '## Scope and conclusion boundary','', 'This completes a validation-only diagnosis. It does not establish a repaired full-feature predictor: the geometry diagnostic outputs two coordinates and cannot replace the required 5,120-feature forecast. Original Motion acceptance counts remain 35/51 and 8/32 fixed-reference; Gate remains its separately locked supported development candidate. No test set was forwarded or new switch selected. Fresh 9137 remains closed, no policy was trained, and overall Phase1_GO is not established.','', 'All weights, cached features and raw feature-vector witnesses remain on HPC. Compact per-episode position scalars, source/model/asset proofs, paired scalars and the independent audit are retained in results/motion_geometry/native_completed/. Reproduction uses the frozen experiments/motion_geometry source on a verified course CPU allocation. No GitHub publication occurred.']
    (repo/'docs/MOTION_GEOMETRY_RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('Wrote docs/MOTION_GEOMETRY_RESULTS.md and results/motion_geometry/position_diagnosis.png')

if __name__=='__main__':main()
