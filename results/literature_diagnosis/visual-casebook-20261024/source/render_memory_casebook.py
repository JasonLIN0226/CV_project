"""Reorganize saved reader and WR evidence for inspection; no model calls.

The 20261022 casebook and scientific results remain unchanged. This renderer
uses their published rows/native PNGs plus the separately audited WR case.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'results/literature_diagnosis/visual-casebook-20261022'
FACTORS = ROOT / 'results/literature_diagnosis/cue-factors-20261022'
ARMS = [(0, 0, 0), (1, 0, 0), (1, 1, 1), (2, 1, 1)]


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, obj): Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
def uri(path): return 'data:image/png;base64,'+base64.b64encode(Path(path).read_bytes()).decode()


def matched_age_rows(card):
    rows = {(r['content'], r['age'], r['past_action']): r for r in card['arms']}
    selected = [rows[(content, age, 0)] for content in (0, 1) for age in (0, 1)]
    for a, b in zip(selected[::2], selected[1::2]):
        assert a['content'] == b['content'] and a['past_action'] == b['past_action'] == 0
        assert a['input']['slot_past_action'] == b['input']['slot_past_action']
        assert a['input']['shared_filler_time'] == b['input']['shared_filler_time']
        assert a['reader']['slot_attention'] == b['reader']['slot_attention']
    return selected


def figures(output, cards, age_rows):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), gridspec_kw={'height_ratios': [1, 1.25]})
    for col, key in enumerate(('content_harm', 'correct_cue_success')):
        c = cards[key]; a, b = c['display_rows'][:2]
        color = '#ba4e41' if col == 0 else '#247d65'
        ax = axes[0, col]
        ax.plot([0, 1], [a['mse'], b['mse']], '-o', color=color, lw=2.5, markersize=9)
        ax.set_xticks([0, 1], ['Noncue content', 'Correct cue content'])
        ax.set_ylim(0, .38); ax.set_xlim(-.18, 1.18); ax.set_ylabel('Feature MSE (lower is better)')
        for x, r in enumerate((a, b)): ax.annotate(f"{r['mse']:.6f}", (x, r['mse']), xytext=(0, 11), textcoords='offset points', ha='center')
        ax.set_title(('FAILURE: adding correct content hurts' if col == 0 else 'SUCCESS: adding correct content helps'), color=color, weight='bold', pad=20)
        ax = axes[1, col]
        for j, (label, r) in enumerate(zip(['Noncue', 'Correct cue'], (a, b))):
            ax.plot(r['contrast_amplitude'], [j-.08, j+.08], '-', color=color if j else '#718096')
            ax.scatter(r['contrast_amplitude'][0], j-.08, marker='o', s=55, color=color if j else '#718096')
            ax.scatter(r['contrast_amplitude'][1], j+.08, marker='s', s=55, color=color if j else '#718096')
        ax.axvline(0, color='#8c98a4', lw=1); ax.axvline(-1, color='#ba4e41', ls='--', alpha=.5); ax.axvline(1, color='#247d65', ls='--', alpha=.5)
        ax.set_xlim(-1.2, 1.4); ax.set_ylim(1.5, -.5); ax.set_yticks([0, 1], ['Noncue', 'Correct cue'])
        ax.set_xlabel('Feature direction: -1 sibling / 0 midpoint / +1 truth\nCircle = h3; square = h6. Not decoded position.')
        ax.text(.5, -.38, f"cue t={c['cue_time']}; input timestamp t={c['noncue_time']}; fixed filler t={c['filler_time']}", transform=ax.transAxes, ha='center', fontsize=10)
    fig.suptitle('One controlled question, two outcomes: does correct cue content help?', fontsize=15, weight='bold', y=.99)
    fig.subplots_adjust(top=.84, bottom=.19, hspace=.48, wspace=.38)
    fig.text(.04, .025, 'Age, past-action summary, recent context, future controls and weights stay fixed within each comparison.\nThe two scenes are selected examples, not a matched cross-scene experiment. Hybrids are diagnostic inputs and may be out of distribution.', fontsize=10, color='#53606c')
    for ext in ('png', 'pdf'): fig.savefig(output/'figures'/f'reader_pair.{ext}', dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for label, selected, color in [('Noncue content', age_rows[:2], '#718096'), ('Correct cue content', age_rows[2:], '#b56c36')]:
        ax.plot([0, 1], [r['mse'] for r in selected], '-o', label=label, color=color, lw=2)
        for x, r in enumerate(selected): ax.annotate(f"{r['mse']:.6f}", (x, r['mse']), xytext=(0, 9), textcoords='offset points', ha='center')
    ax.set_xticks([0, 1], ['Timestamp from noncue (t=11)', 'Timestamp from cue (t=7)'])
    ax.set_xlim(-.2, 1.2); ax.set_ylim(.07, .30); ax.set_ylabel('Feature MSE (lower is better)'); ax.legend(loc='upper left')
    ax.set_title('Auxiliary ablation: change timestamp only', weight='bold')
    fig.subplots_adjust(bottom=.24, top=.86)
    fig.text(.09, .06, 'Same content within each line, same past-action summary (from t=11), filler, reader and targets.\nThis selected case shows content-by-time interaction, not a universal penalty for old memories.', fontsize=10, color='#53606c')
    for ext in ('png', 'pdf'): fig.savefig(output/'figures'/f'timestamp_auxiliary.{ext}', dpi=170)
    plt.close(fig)


def run(output):
    output = output.resolve(); (output/'figures').mkdir(parents=True, exist_ok=True)
    old_manifest = read(OLD/'manifest.json')['files']
    for relative in ('cases.json', 'rgb_frames.json'):
        assert sha(OLD/relative) == old_manifest[relative]['sha256']
    assert sha(FACTORS/'case_cards.json') == read(FACTORS/'manifest.json')['files']['case_cards.json']['sha256']
    cards = read(OLD/'cases.json'); factor_cards = read(FACTORS/'case_cards.json')
    frame_info = read(OLD/'rgb_frames.json'); frames = {}
    for index, info in frame_info.items():
        path = OLD/info['png']
        assert sha(path) == old_manifest[info['png']]['sha256']
        assert hashlib.sha256(np.asarray(Image.open(path).convert('RGB')).tobytes()).hexdigest() == info['rgb_sha256']
        frames[index] = uri(path)
    for key, c in cards.items():
        keyed = {(r['content'], r['age'], r['past_action']): r for r in factor_cards[key]['arms']}
        assert c['display_rows'] == [keyed[a] for a in ARMS]
    age_rows = matched_age_rows(factor_cards['age_contribution'])
    reader = {'cards': cards, 'age_rows': age_rows,
              'age_delta_noncue': age_rows[1]['mse']-age_rows[0]['mse'],
              'age_delta_cue': age_rows[3]['mse']-age_rows[2]['mse'],
              'selection': 'Reuse the three published factorial representatives; no new selection on confirmation data.'}
    save(output/'reader_cases.json', reader)
    figures(output, cards, age_rows)
    wr_provenance = read(output/'wr/provenance.json')
    for relative, digest in wr_provenance['outputs'].items():
        assert sha(output/'wr'/relative) == digest
    wr = read(output/'wr/case.json')
    wr_frame_map = {str(x['original_time']): uri(output/'wr'/x['path']) for x in wr['frames']}
    payload = {'reader': reader, 'frames': frames, 'wr': wr, 'wr_frames': wr_frame_map,
               'reader_plot': uri(output/'figures/reader_pair.png'),
               'age_plot': uri(output/'figures/timestamp_auxiliary.png')}
    template = Path(__file__).with_name('memory_casebook_template.html')
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace('</', '<\\/')
    (output/'index.html').write_text(template.read_text().replace('__CASE_PAYLOAD__', encoded))
    source = output/'source'; source.mkdir(exist_ok=True)
    for p in (Path(__file__), template): shutil.copy2(p, source/p.name)
    save(output/'audit.json', {'new_model_calls': 0, 'new_encoder_calls': 0, 'simulator_calls': 0,
        'source_casebook_sha256': sha(OLD/'cases.json'), 'factor_cards_sha256': sha(FACTORS/'case_cards.json'),
        'wr_case_sha256': sha(output/'wr/case.json'), 'native_reader_pngs_verified': len(frames),
        'timestamp_only_rows_verified': 4, 'age_delta_noncue': reader['age_delta_noncue'],
        'age_delta_correct_cue': reader['age_delta_cue'],
        'reader_comparison': 'Matched within each case, not a controlled comparison between scenes.',
        'prediction_images_generated': False, 'WR_control_is_oracle': False})
    save(output/'manifest.json', {'files': {str(p.relative_to(output)): {'sha256': sha(p), 'bytes': p.stat().st_size}
        for p in sorted(output.rglob('*')) if p.is_file() and p != output/'manifest.json'}})
    print(json.dumps({'output': str(output), 'native_reader_pngs_verified': len(frames), 'new_model_calls': 0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/literature_diagnosis/visual-casebook-20261024')
    run(parser.parse_args().output)
