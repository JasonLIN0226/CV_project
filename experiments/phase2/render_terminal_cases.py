"""Render saved R/D/S cases without model calls or fabricated forecast pixels.

Results mode reads the frozen run's records, development decisions and reader
qualification JSON. Images are regenerated exact simulator observations/actual
future truth. Selection is deterministic, with all-task denominators alongside
the selected examples. --illustration-only can render generator-profile data
but never supplies or invents model results.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import html
import json
from pathlib import Path
import sys
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.phase2 import terminal_structures as env

TASK_NAMES = {'R': 'Redundancy: two copies + complementary evidence',
              'D': 'Decoy: two useful cues + a marked distractor',
              'S': 'Synergy: matching tags + XOR payloads'}
TASK_RULES = {'R': 'A and Aprime repeat one direction. Keep one copy and the other direction.',
              'D': 'A and B determine two directions. The slashed-circle cue is irrelevant.',
              'S': 'Find the two matching tags. XOR their payload bits; discard the singleton tag.'}
METHODS = [('medoid', 'Forecast medoid'), ('full_sensitivity', 'Full-set sensitivity'),
           ('reader_attention', 'Reader attention'), ('visual_diversity', 'Visual diversity'),
           ('fifo', 'FIFO'), ('terminal_uniform', 'Uniform deletion')]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def font(size, bold=False):
    names = ['/System/Library/Fonts/Supplemental/Arial Bold.ttf' if bold else '/System/Library/Fonts/Supplemental/Arial.ttf',
             '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
    for name in names:
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def qualification_problem(record_id, rows):
    rr = rows.get(record_id, [])
    native = [r for r in rr if r['arm'].startswith('native_') and r['semantic_complete']]
    for row in sorted(native, key=lambda r: r['arm']):
        if not row['strict_correct']:
            return {'kind': 'complete_native_prediction_wrong', 'arm': row['arm'],
                    'class': row['class'], 'target_class': row['target_class'],
                    'description': 'A semantically complete retained pair was not decoded correctly.'}
    for row in sorted(rr, key=lambda r: r['arm']):
        if row['arm'].startswith('flip_') and not row['strict_correct']:
            return {'kind': 'counterfactual_content_failure', 'arm': row['arm'],
                    'class': row['class'], 'target_class': row['target_class'],
                    'description': 'The fixed-time content-flip diagnostic did not follow the changed cue.'}
    complete = [r for r in rr if r['arm'].startswith('fixed_') and r['semantic_complete']]
    incomplete = [r for r in rr if r['arm'].startswith('fixed_') and not r['semantic_complete']]
    for good in sorted(complete, key=lambda r: r['arm']):
        for bad in sorted(incomplete, key=lambda r: r['arm']):
            if good['mse'] >= bad['mse']:
                return {'kind': 'complete_content_not_lower_mse', 'arm': good['arm'],
                        'comparison_arm': bad['arm'], 'complete_mse': good['mse'], 'incomplete_mse': bad['mse'],
                        'description': 'At fixed ages, a complete pair has no lower MSE than an incomplete pair.'}
    return None


def locally_decoded(record_id, rows):
    rr = [r for r in rows.get(record_id, []) if r['arm'].startswith('native_') and r['semantic_complete']]
    return bool(rr) and all(r['strict_correct'] for r in rr)


def choose_cases(task, records, decisions, qualification, qrows, illustration):
    rr = sorted([r for r in records if r['task_family'] == task], key=lambda r: r['id'])
    if not rr:
        return {'task': task, 'cases': [], 'status': 'no_saved_records'}
    if illustration:
        return {'task': task, 'status': 'illustration_only', 'reader_qualified': None,
                'cases': [{'id': rr[0]['id'], 'kind': 'illustration',
                           'label': 'GENERATOR ILLUSTRATION ONLY - NO MODEL RESULT',
                           'reason': 'First record by lexicographic episode ID.'}]}
    qualified = bool(qualification['gates'][task+'/K2_content'])
    if not qualified:
        problematic = [(r, qualification_problem(r['id'], qrows)) for r in rr]
        found = [(r, p) for r, p in problematic if p is not None]
        chosen, problem = found[0] if found else (rr[0], None)
        cases = [{'id': chosen['id'], 'kind': 'reader_qualification_case',
                  'label': 'READER QUALIFICATION CASE - NOT A RETENTION CLAIM',
                  'reason': 'First episode ID with a saved content/decoding failure.' if found else 'Task aggregate qualification failed; no single-row failure located by the fixed search.',
                  'diagnostic': problem}]
        controls = [r for r in rr if r['id'] != chosen['id']
                    and locally_decoded(r['id'], qrows)
                    and qualification_problem(r['id'], qrows) is None]
        controls.sort(key=lambda r: (r['family'] != chosen['family'], r['id']))
        if controls:
            cases.append({'id': controls[0]['id'], 'kind': 'local_reader_success_control',
                          'label': 'LOCAL READER SUCCESS - TASK GATE STILL FAILED',
                          'reason': 'First record passing the saved native-pair, content-flip and fixed-time MSE checks; same family preferred.'})
        return {'task': task, 'status': 'reader_gate_failed', 'reader_qualified': False, 'cases': cases}
    failures = [r for r in rr if not decisions[r['id']]['policies']['medoid']['semantic_correct']]
    if not failures:
        return {'task': task, 'status': 'no_medoid_semantic_failure_observed', 'reader_qualified': True,
                'cases': [{'id': rr[0]['id'], 'kind': 'medoid_success_control',
                           'label': 'SUCCESS CONTROL - NO MEDOID EVIDENCE LOSS IN THIS TASK',
                           'reason': 'First episode ID; the complete task denominator is reported.'}]}
    qualified_cases = [r for r in failures if locally_decoded(r['id'], qrows)]
    chosen = qualified_cases[0] if qualified_cases else failures[0]
    cases = [{'id': chosen['id'], 'kind': 'medoid_retention_failure' if qualified_cases else 'mixed_reader_retention_case',
              'label': 'MEDOID RETENTION FAILURE - COMPLETE PAIRS DECODE CORRECTLY' if qualified_cases else 'SEMANTIC LOSS WITH CASE-LEVEL READER UNCERTAINTY',
              'reason': 'First medoid semantic failure with every native complete pair decoded correctly, by episode ID; not the largest loss.' if qualified_cases else 'Task gate passed but no semantic failure had all native complete pairs decoded correctly; do not call this isolated retention failure.'}]
    controls = [r for r in rr if decisions[r['id']]['policies']['medoid']['semantic_correct']
                and locally_decoded(r['id'], qrows)]
    controls.sort(key=lambda r: (r['family'] != chosen['family'], r['id']))
    if controls:
        cases.append({'id': controls[0]['id'], 'kind': 'medoid_success_control',
                      'label': 'MEDOID RETENTION + READER SUCCESS CONTROL',
                      'reason': 'First medoid semantic success with all native complete pairs decoded correctly; same family preferred.'})
    else:
        full = decisions[chosen['id']]['policies'].get('full_sensitivity', {})
        if full.get('semantic_correct') and qualification['gates'].get(task+'/K3_content', False):
            cases[0]['same_parent_control'] = {'method': 'full_sensitivity',
                'description': 'No medoid success exists in this task. Full-set sensitivity retains the necessary pair on this SAME parent; compare the two table rows. This is a policy control, not a second independent case.'}
    return {'task': task, 'status': 'medoid_failure_found', 'reader_qualified': True, 'cases': cases}


def task_summary(task, records, decisions, qualification, illustration):
    rr = [r for r in records if r['task_family'] == task]
    result = {'episodes': len(rr), 'families': len({r['family'] for r in rr}), 'policies': {}}
    if illustration:
        return result
    result['K2_qualified'] = bool(qualification['gates'][task+'/K2_content'])
    result['K3_qualified'] = bool(qualification['gates'][task+'/K3_content'])
    result['failed_reader_gates'] = [key for key, passed in qualification['gates'].items() if key.startswith(task+'/') and not passed]
    for method, _ in METHODS:
        pp = [decisions[r['id']]['policies'][method] for r in rr if method in decisions[r['id']]['policies']]
        if pp:
            result['policies'][method] = {'episodes': len(pp), 'semantic_loss_count': sum(not p['semantic_correct'] for p in pp),
                                          'mean_regret': float(np.mean([p['regret'] for p in pp]))}
    return result


def wrap(draw, text, xy, width=110, size=17, color=(35, 45, 57)):
    x, y = xy
    for line in textwrap.wrap(text, width=width):
        draw.text((x, y), line, font=font(size), fill=color)
        y += size+6
    return y


def verified_episode(record, index_to_hash):
    """Verify actual displayed raw frames, including Pillow/font differences."""
    ep = env.episode(record)
    receipts = []
    for t in record['evidence']+[record['query']]+record['target_times']:
        index = record['rgb_indices'][t]
        actual = hashlib.sha256(ep['rgb'][t].tobytes()).hexdigest()
        if index not in index_to_hash or actual != index_to_hash[index]:
            raise ValueError(f'Regenerated RGB does not match archived frame hash: episode={record["id"]},t={t},index={index}; do not call these exact source frames')
        receipts.append({'timestamp': t, 'unique_rgb_index': index, 'raw_rgb_sha256': actual, 'matches_archived_rgb_index': True})
    return ep, receipts


def draw_card(record, case, decision, qrows, path, illustration, verified_rgb=None):
    task = record['task_family']
    ep = env.episode(record) if verified_rgb is None else verified_rgb
    width, height = 1240, 900
    image = Image.new('RGB', (width, height), (247, 249, 252))
    draw = ImageDraw.Draw(image)
    draw.text((25, 18), TASK_NAMES[task], font=font(25, True), fill=(18, 35, 52))
    label_color = (155, 55, 35) if 'FAIL' in case['label'] or 'NOT A RETENTION' in case['label'] else (22, 100, 72)
    draw.text((25, 55), case['label'], font=font(17, True), fill=label_color)
    draw.text((25, 82), 'ID '+record['id']+' | family '+record['family']+' | '+record['condition'], font=font(14), fill=(65, 75, 90))
    wrap(draw, TASK_RULES[task], (25, 107), size=17)
    times = record['evidence']+[record['query']]+record['target_times']
    titles = ['Cue 1', 'Cue 2', 'Cue 3', 'Shared current', 'TRUE future h3', 'TRUE future h6']
    for j, (t, label) in enumerate(zip(times, titles)):
        x, y = 25+j*202, 167
        draw.text((x, y-25), label+' (t='+str(t)+')', font=font(15, True), fill=(35, 45, 57))
        image.paste(Image.fromarray(ep['rgb'][t]).resize((192, 192), Image.Resampling.NEAREST), (x, y))
        if j < 3:
            description = 'payload='+str(record['slot_payload'][j])
            if task == 'S':
                description += '; tag='+str(record['tags'][j])
            else:
                description += '; axis='+record['semantic_axes'][j]
            draw.text((x, y+198), description, font=font(13), fill=(65, 75, 90))
            draw.text((x, y+218), 'Eval role: '+record['roles'][j], font=font(13), fill=(65, 75, 90))
    y = 420
    correct = ' OR '.join('cue '+str(j+1) for j in record['semantic_correct_delete_set'])
    draw.text((25, y), 'Evaluator-only rule: delete ONE item: '+correct+'.  Roles are not scorer inputs.', font=font(16, True), fill=(35, 45, 57))
    y += 33
    if illustration:
        y = wrap(draw, 'No saved model result is loaded. This card demonstrates the generator and task rule only.', (25, y), size=19)
    else:
        columns = [(25, 'Method'), (300, 'Delete'), (395, 'Keep'), (540, 'Evidence'), (695, 'Mean-plan MSE'), (880, 'Regret'), (1045, 'Reader gate')]
        for x, title in columns:
            draw.text((x, y), title, font=font(15, True), fill=(35, 45, 57))
        y += 27
        for method, name in METHODS:
            if method not in decision['policies']:
                continue
            p = decision['policies'][method]
            d = p['drop']
            foreground = (155, 45, 35) if not p['semantic_correct'] else (22, 100, 72)
            draw.rectangle((20, y-3, width-22, y+24), fill=(237, 242, 247) if method == 'medoid' else (247, 249, 252))
            vals = [name, str(d+1), ', '.join(str(j+1) for j in range(3) if j != d),
                    'LOST' if not p['semantic_correct'] else 'retained', f"{p['mse']:.6f}", f"{p['regret']:.6f}",
                    'pass' if p.get('reader_qualified_for_claim') is True else ('fail' if p.get('reader_qualified_for_claim') is False else 'not checked')]
            for (x, _), val in zip(columns, vals):
                draw.text((x, y), val, font=font(15), fill=foreground if x in (300, 395, 540) else (35, 45, 57))
            y += 29
        diagnostic = case.get('diagnostic')
        if diagnostic:
            y += 8
            y = wrap(draw, 'Reader diagnostic: '+diagnostic['description']+' Saved arm: '+diagnostic['arm']+'.', (25, y), size=16)
            if 'class' in diagnostic:
                y = wrap(draw, f"Nearest true-future FEATURE prototype class: {diagnostic['class']}; expected class: {diagnostic['target_class']}. This is not a decoded prediction image.", (25, y), size=15)
            if diagnostic['kind'] == 'counterfactual_content_failure':
                y = wrap(draw, 'The diagnostic used a fixed-age content intervention; images above show the original native episode.', (25, y), size=15)
        if case.get('same_parent_control'):
            y = wrap(draw, 'Control: '+case['same_parent_control']['description'], (25, y+8), size=15)
    y += 14
    y = wrap(draw, 'Selection: '+case['reason'], (25, y), size=15)
    end = wrap(draw, 'All displayed images are simulator observations or recorded-plan ground-truth futures, never model-generated forecasts. Policy losses average the saved evaluation action bank; the shown truth uses the recorded actual plan. The text below images is evaluator annotation.', (25, y+8), size=14, color=(75, 85, 98))
    image.crop((0, 0, width, min(height, end+25))).save(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--illustration-only', action='store_true')
    args = p.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new empty casebook directory')
    data = run/'data' if (run/'data/records.json').exists() else run
    source_records = data/'records.json'
    all_records = read(source_records)
    records = [r for r in all_records if r['split'] == 'development']
    if not records:
        raise ValueError('No saved development records')
    input_files = [source_records]
    rgb_index_path = data/'rgb_index.json'
    if not rgb_index_path.is_file():
        raise ValueError('Casebook requires saved rgb_index.json to verify every displayed frame')
    rgb_index = read(rgb_index_path)
    index_to_hash = {int(index): digest for digest, index in rgb_index.items()}
    if len(index_to_hash) != len(rgb_index):
        raise ValueError('RGB index contains duplicate numeric indices')
    input_files.append(rgb_index_path)
    generator = ROOT/'experiments/phase2/terminal_structures.py'
    if (run/'protocol.json').exists():
        protocol = read(run/'protocol.json')
        expected = protocol['source_sha256']['experiments/phase2/terminal_structures.py']
        if sha(generator) != expected:
            raise ValueError('Current generator differs from frozen source; use the matching source archive')
        input_files.append(run/'protocol.json')
    elif not args.illustration_only:
        raise ValueError('Results mode requires the frozen protocol source binding')
    decisions, qualification, qrows = {}, {}, defaultdict(list)
    if not args.illustration_only:
        result_dir = run/'development'
        paths = [result_dir/'baseline_decisions.json', result_dir/'reader_qualification.json', result_dir/'reader_qualification_rows.json']
        decision_rows = read(paths[0]); qualification = read(paths[1])
        decisions = {r['id']: r for r in decision_rows}
        for row in read(paths[2]):
            qrows[row['id']].append(row)
        if {r['id'] for r in records} != set(decisions):
            raise ValueError('Development decision/record identities differ')
        input_files += paths
    output.mkdir(parents=True, exist_ok=True)
    sections, selections = [], []
    lookup = {r['id']: r for r in records}
    for task in env.TASK_FAMILIES:
        selection = choose_cases(task, records, decisions, qualification, qrows, args.illustration_only)
        selection['population'] = task_summary(task, records, decisions, qualification, args.illustration_only)
        cards = []
        for number, case in enumerate(selection['cases']):
            filename = f'{task.lower()}-{number+1}.png'
            ep, receipts = verified_episode(lookup[case['id']], index_to_hash)
            case['raw_frame_hash_checks'] = receipts
            draw_card(lookup[case['id']], case, decisions.get(case['id']), qrows, output/filename, args.illustration_only, ep)
            case['image'] = filename
            case['image_sha256'] = sha(output/filename)
            cards.append('<figure><figcaption>'+html.escape(case['label'])+'</figcaption><img src="'+filename+'" alt="'+html.escape(case['label'])+'"></figure>')
        pop = selection['population']
        table = ''
        if not args.illustration_only:
            table = '<table><tr><th>Method</th><th>Lost necessary evidence</th><th>Mean regret</th></tr>'
            for method, name in METHODS:
                if method in pop['policies']:
                    s = pop['policies'][method]
                    table += f'<tr><td>{name}</td><td>{s["semantic_loss_count"]}/{s["episodes"]}</td><td>{s["mean_regret"]:.6f}</td></tr>'
            table += '</table>'
        warning = ''
        if not args.illustration_only and not pop['K2_qualified']:
            warning = '<p class="failure"><strong>READER GATE FAILED.</strong> These are reader-qualification diagnostics. The policy losses below do not establish a retention-method advantage.</p>'
        elif not args.illustration_only and not pop['K3_qualified']:
            warning = '<p class="failure"><strong>FULL-REFERENCE READER GATE FAILED.</strong> Full-set sensitivity is an unqualified diagnostic comparator for this task.</p>'
        sections.append('<section><h2>'+html.escape(TASK_NAMES[task])+'</h2><p>'+html.escape(TASK_RULES[task])+'</p><p>'+html.escape(selection['status'])+f' | {pop["episodes"]} episodes, {pop["families"]} independent base families</p>'+warning+table+''.join(cards)+'</section>')
        selections.append(selection)
    result = {'mode': 'illustration_only_no_model_results' if args.illustration_only else 'saved_development_results',
              'selection_rule': 'First eligible episode ID; same-family success preferred; all task denominators retained.',
              'no_model_or_encoder_calls': True, 'image_semantics': 'Actual simulator RGB and actual-plan future truth; no decoded forecasts.',
              'all_displayed_frames_match_saved_rgb_index': True,
              'recorded_plan_truth_and_bank_average_losses_are_distinct': True,
              'source_sha256': {'renderer': sha(__file__), 'generator': sha(generator)},
              'input_sha256': {str(f): sha(f) for f in input_files}, 'tasks': selections}
    (output/'selection.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    title = 'Terminal R/D/S: saved cases' if not args.illustration_only else 'Terminal R/D/S: generator illustrations ONLY'
    intro = ('Cases are selected deterministically from saved development results. A failed reader gate is shown as a reader diagnosis, not a retention-method claim. '
             'All images show observations and ground-truth futures, not model-generated predictions. Regret is averaged over the saved evaluation action bank. '
             'These are development results, not independent confirmation.') if not args.illustration_only else 'Raw generator illustrations only. No model result, baseline comparison or research success is implied.'
    document = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>'+title+'</title><style>body{font:17px/1.5 system-ui,sans-serif;background:#f7f9fc;color:#172536;margin:0 auto;max-width:1280px;padding:28px}h1{font-size:30px}h2{font-size:24px}section{margin:35px 0;border-top:1px solid #cbd4df;padding-top:18px}figure{margin:22px 0}figcaption{font-weight:650;margin-bottom:8px}img{display:block;max-width:100%;height:auto;border:1px solid #ccd5df}table{border-collapse:collapse;background:white}th,td{text-align:left;padding:7px 17px;border:1px solid #d6dfe8}.notice{padding:16px;background:#eaf0f7}.failure{padding:16px;background:#fff0e9;border-left:5px solid #af492e}</style><h1>'+title+'</h1><p class="notice">'+intro+'</p>'+''.join(sections)+'<p>Selection, all denominators and source hashes: <a href="selection.json">selection.json</a>.</p></html>'
    (output/'index.html').write_text(document)
    print(json.dumps({'output': str(output), 'mode': result['mode'], 'cards': sum(len(t['cases']) for t in selections), 'model_calls': 0}))


if __name__ == '__main__':
    main()
