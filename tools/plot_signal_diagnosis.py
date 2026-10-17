#!/usr/bin/env python3
"""Plot deterministic diagnostic output; no model or research data generation."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT/'results/diagnostics/signal_diagnosis.json').read_text())
splits = ('test_id', 'test_long_gap', 'test_distractor_shift')
labels = ('Ordinary', 'Long gap', 'Distractors')
roles = (('gate', 'cue'), ('motion', 'start'), ('motion', 'end'))
plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
fig, (a, b) = plt.subplots(1, 2, figsize=(12.2, 4.6), gridspec_kw={'width_ratios':[1, 1.4]})
values = np.array([[next(x['positive_both_metrics_all_seeds'] for x in data['cross_seed_consistency']
                        if x['task']==task and x['role']==role and x['split']==split and x['horizon']==6)
                    for split in splits] for task, role in roles])
a.imshow(values/128, cmap='Greys', vmin=0, vmax=1, aspect='auto')
a.set_xticks(range(3), labels)
a.set_yticks(range(3), ('Gate cue', 'Motion start', 'Motion end'))
for i in range(3):
    for j in range(3):
        a.text(j, i, f'{values[i,j]}/128', ha='center', va='center', color='white' if values[i,j]>77 else 'black')
a.set_title('A  Stable positive deletion effects\nBoth errors positive in all three seeds', loc='left', pad=14)
colors = {'gate':'#aaaaaa', 'motion':'#222222'}
for j, task in enumerate(('gate', 'motion')):
    for i, split in enumerate(splits):
        rows = [x for x in data['baseline_headroom'] if x['task']==task and x['split']==split
                and x['horizon']==6 and x['metric']=='feature_mse']
        reject = np.mean([x['reject_newest'] for x in rows]);oracle = np.mean([x['future_feature_oracle'] for x in rows])
        gap = 100*(reject-oracle)/reject
        y = i*3+j
        b.barh(y, gap, color=colors[task], height=.75)
        b.text(69, y, f'{gap:.2f}%', va='center', ha='right', fontsize=9)
        b.plot([100*x['relative_gap'] for x in rows], [y]*3, 'o', color='black', markerfacecolor='white', markersize=4)
b.set_yticks([i*3+j for i in range(3) for j in range(2)], [f'{label}  {task.title()}' for label in labels for task in ('gate','motion')])
b.invert_yaxis();b.set_xlim(-1, 71);b.set_xticks((0, 10, 20, 30, 40, 50, 60))
b.set_xlabel('Feature error reduction vs reject-newest (%)')
b.set_title('B  Future-informed oracle headroom\nSmaller means less room above the simple rule', loc='left', pad=14)
fig.text(.5, .015, '6-step development diagnostics. Left: independent pairs, not accuracy. Right: bars average losses over seeds; dots show each seed, not confidence intervals.', ha='center', fontsize=8)
fig.tight_layout(rect=(0,.05,1,1))
out = ROOT/'reports/figures';out.mkdir(exist_ok=True)
for ext in ('png', 'pdf'):
    fig.savefig(out/f'13_signal_diagnosis.{ext}', dpi=200, bbox_inches='tight')
print('Updated reports/figures/13_signal_diagnosis.png and .pdf')
