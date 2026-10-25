"""Read-only paired summaries of a completed reader-scale pilot."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def interval(rows, key):
    pairs = defaultdict(list)
    for row in rows:
        pairs[row['pair']].append(row[key])
    values = np.array([np.mean(v) for v in pairs.values()])
    rng = np.random.default_rng(20261024)
    draws = values[rng.integers(len(values), size=(5000, len(values)))].mean(1)
    return {'mean': float(values.mean()), 'low': float(np.quantile(draws, .025)),
            'high': float(np.quantile(draws, .975)), 'scene_pairs': len(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows_path = args.run / 'evaluation_rows.json'
    rows = json.loads(rows_path.read_text())
    if len(rows) != 1920:
        raise ValueError('Expected 4 readers x 5 controls x 96 episodes')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use a new empty output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    readers = ('old_16_reference', 'scale_16', 'scale_64', 'scale_512')
    result = {'scope': 'single-model-seed diagnostic, pair bootstrap is descriptive scenario uncertainty',
              'input_sha256': hashlib.sha256(rows_path.read_bytes()).hexdigest(), 'readers': {}}
    for reader in readers:
        lookup = defaultdict(dict)
        for row in rows:
            if row['reader'] == reader:
                if row['control'] in lookup[row['episode_id']]:
                    raise ValueError('Duplicate reader/control/episode')
                lookup[row['episode_id']][row['control']] = row
        if len(lookup) != 96 or any(len(v) != 5 for v in lookup.values()):
            raise ValueError('Incomplete control pairing')
        derived = []
        for episode_id, v in lookup.items():
            cue = v['cue_K2']
            derived.append({'episode_id': episode_id, 'pair': cue['pair'], 'condition': cue['condition'],
                'cue_mse': cue['mse'], 'no_cue_mse': v['no_cue_K2']['mse'],
                'wrong_cue_mse': v['wrong_cue_K2']['mse'], 'recent_mse': v['recent']['mse'],
                'cue_minus_no_cue': cue['mse']-v['no_cue_K2']['mse'],
                'cue_minus_wrong_cue': cue['mse']-v['wrong_cue_K2']['mse'],
                'both_horizons_correct_direction': all(a > 0 for a in cue['contrast_amplitude']),
                'cue_feature_direction_fraction': cue['feature_branch_correct']})
        result['readers'][reader] = {}
        for condition in ('all', 'id', 'long_gap', 'distractor_shift'):
            selected = [r for r in derived if condition == 'all' or r['condition'] == condition]
            result['readers'][reader][condition] = {
                'episodes': len(selected),
                **{k: interval(selected, k) for k in ('cue_mse','no_cue_mse','wrong_cue_mse','recent_mse',
                    'cue_minus_no_cue','cue_minus_wrong_cue','cue_feature_direction_fraction')},
                'cue_worse_than_no_cue': sum(r['cue_minus_no_cue'] > 1e-10 for r in selected),
                'correct_cue_worse_than_wrong': sum(r['cue_minus_wrong_cue'] > 1e-10 for r in selected),
                'both_horizons_correct_direction': sum(r['both_horizons_correct_direction'] for r in selected)}
    (args.output/'paired_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    x = np.arange(4)
    for offset, key, label, color in ((-.22,'cue_mse','Correct cue','#2b7a78'),
          (0,'no_cue_mse','No cue, same K','#888888'),(.22,'wrong_cue_mse','Wrong cue, matched age/actions','#c15b56')):
        values = [result['readers'][r]['all'][key] for r in readers]
        means = np.array([v['mean'] for v in values])
        err = np.array([[m-v['low'] for m,v in zip(means, values)],
                        [v['high']-m for m,v in zip(means, values)]])
        axes[0].bar(x+offset, means, .21, label=label, color=color, yerr=err, capsize=2)
    axes[0].set_ylabel('Mean h3/h6 feature MSE (lower is better)')
    axes[0].legend(fontsize=8)
    axes[0].set_title('One common input interface; 48 new scene pairs')
    counts = [result['readers'][r]['all']['correct_cue_worse_than_wrong'] for r in readers]
    axes[1].bar(x, counts, .55, color='#6f639c')
    for i, count in enumerate(counts):
        axes[1].text(i, count+.7, f'{count}/96', ha='center')
    axes[1].set_ylim(0, max(counts)+8)
    axes[1].set_ylabel('Episodes where correct cue has higher MSE')
    axes[1].set_title('Matched wrong-cue content intervention')
    for ax in axes:
        ax.set_xticks(x, ['Old 16\nreference','Retrain 16','Retrain 64','Retrain 512'])
    fig.text(.5, .01, 'Fixed old transforms; retraining uses seed 17 and 512 updates. Bars are diagnostic controls, not online retention policies.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(args.output/'reader_scale.png', dpi=180)
    fig.savefig(args.output/'reader_scale.pdf')
    print(json.dumps({r: result['readers'][r]['all'] for r in readers}, indent=2))


if __name__ == '__main__':
    main()
