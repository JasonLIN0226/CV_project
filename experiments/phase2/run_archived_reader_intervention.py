"""CPU-only, no-fit weighting intervention on the released terminal reader.

Only the already observed development split is read. No training, generation,
encoding, fresh diagnostic, student, or confirmation entry point exists here.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'experiments/predictor/control/src')]

from experiments.phase2 import run_reader_aggregation as aggregation
from experiments.phase2 import run_terminal_structures as legacy
from experiments.phase2.reader_aggregation import build_reader

ARMS = ('original', 'uniform_intervention')
read, write, sha = legacy.read, legacy.write, legacy.sha


class ArchivedPredictor(legacy.Predictor):
    def __init__(self, source, config, arm):
        import torch
        if arm not in ARMS:
            raise ValueError('Only original and the no-fit uniform intervention are allowed')
        self.torch, self.device, self.c = torch, 'cpu', config
        self.basis = dict(np.load(source/'basis.npz', allow_pickle=False))
        state = torch.load(source/'reader_selected.pt', map_location='cpu', weights_only=False)
        if state['basis_sha256'] != sha(source/'basis.npz'):
            raise ValueError('Archived checkpoint/basis binding failed')
        self.model = build_reader(len(self.basis['input_mean']), config['reader'], legacy.H,
                                  self.basis, arm, heads=4)
        self.model.load_state_dict(state['model'], strict=True)
        self.model.eval().requires_grad_(False)
        self.calls, self.examples, self.seconds, self.reconstruction_max, self.attention_queries = 0, 0, 0., 0., 0


def paired_summary(original_rows, intervention_rows):
    """Descriptive matched-row differences; families remain the reporting unit."""
    original = {(r['id'], r['arm']): r for r in original_rows}
    intervention = {(r['id'], r['arm']): r for r in intervention_rows}
    if len(original) != len(original_rows) or len(intervention) != len(intervention_rows):
        raise ValueError('Duplicate qualification key')
    if original.keys() != intervention.keys():
        raise ValueError('Intervention qualification rows are not paired')
    families = defaultdict(list)
    for key, before in original.items():
        after = intervention[key]
        for field in ('family', 'task_family', 'condition', 'target_class', 'semantic_complete'):
            if before[field] != after[field]:
                raise ValueError('Paired row metadata changed: '+field)
        cell = (before['task_family'], before['condition'], before['arm'], before['family'])
        families[cell].append([float(after['strict_correct'])-float(before['strict_correct']),
                               after['mse']-before['mse']])
    cells = defaultdict(list); family_rows = []
    for key, values in sorted(families.items()):
        difference = np.asarray(values, dtype=np.float64).mean(0)
        cells[key[:3]].append(difference)
        family_rows.append({'task_family': key[0], 'condition': key[1], 'arm': key[2], 'family': key[3],
                            'accuracy_difference': float(difference[0]), 'mse_difference': float(difference[1])})
    summary = {}
    for key, values in sorted(cells.items()):
        a = np.asarray(values)
        summary['/'.join(key)] = {'families': len(a), 'accuracy_difference': float(a[:, 0].mean()),
                                  'mse_difference': float(a[:, 1].mean())}
    return {'direction': 'uniform_intervention minus original', 'scope': 'descriptive old-DEV diagnosis; no new confirmation',
            'family_cells': family_rows, 'equal_family_summary': summary}


def run(config_path, source, output):
    import torch
    config_path, source, output = map(lambda p: Path(p).resolve(), (config_path, source, output))
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output must be empty; do not overwrite an intervention run')
    c = read(config_path)
    if c['status'] != 'ready-for-source-freeze-not-yet-executed' or c['must_resolve_before_freeze']:
        raise ValueError('Protocol is not ready for source freeze')
    if c['students_enabled'] or c['original_confirm_open_enabled']:
        raise ValueError('No students or confirmation are permitted in this diagnostic')
    parent = aggregation.verify_parent(c, source)
    base_c = read(source/'protocol.json')['config']
    output.mkdir(parents=True, exist_ok=True)
    source_hashes = aggregation.source_hashes()
    source_hashes['experiments/phase2/run_archived_reader_intervention.py'] = sha(Path(__file__))
    write(output/'protocol.json', {'scope': 'archived checkpoint, old development only, no fitting',
          'config_file_sha256': sha(config_path), 'source_sha256': source_hashes, 'parent_binding': parent,
          'checkpoint_sha256': sha(source/'reader_selected.pt'), 'device': 'cpu', 'arms': list(ARMS),
          'training': False, 'data_generated': False, 'fresh_diagnostic_opened': False, 'confirmation_opened': False})
    shutil.copyfile(source/'basis.npz', output/'basis.npz')
    data = legacy.Data(source/'data', dict(np.load(source/'basis.npz', allow_pickle=False)))
    records = [r for r in data.records if r['split'] == 'development']
    if not records or {r['task_family'] for r in records} != set(legacy.TASKS):
        raise ValueError('Archived development must contain all three task families')
    # Sufficient native inputs for coefficient reconstruction, with the bound basis.
    np.savez_compressed(output/'reconstruction_inputs.npz', ids=np.array([r['id'] for r in records]),
                        current=np.stack([data.capsule(r)['current'] for r in records]))
    torch.set_num_threads(4)
    began = time.perf_counter(); summaries = {}; rows = {}; executions = {}
    for arm in ARMS:
        predictor = ArchivedPredictor(source, base_c, arm)
        dest = output/arm; dest.mkdir()
        started = time.perf_counter()
        summaries[arm] = legacy.qualify(base_c, data, predictor, records, dest)
        rows[arm] = read(dest/'reader_qualification_rows.json')
        executions[arm] = {'seconds': time.perf_counter()-started, 'examples': predictor.examples,
                           'batches': predictor.calls, 'inference_seconds': predictor.seconds,
                           'coefficient_reconstruction_max_abs_difference': predictor.reconstruction_max,
                           'same_checkpoint_sha256': sha(source/'reader_selected.pt')}
        write(dest/'execution.json', executions[arm])
    if aggregation.verify_parent(c, source) != parent:
        raise ValueError('Parent artifacts changed during archived intervention')
    if sha(config_path) != read(output/'protocol.json')['config_file_sha256']:
        raise ValueError('Config changed during archived intervention')
    now = aggregation.source_hashes()
    now['experiments/phase2/run_archived_reader_intervention.py'] = sha(Path(__file__))
    if now != source_hashes:
        raise ValueError('Scientific source changed during archived intervention')
    write(output/'paired_summary.json', paired_summary(rows['original'], rows['uniform_intervention']))
    summary = {'status': 'completed_archived_checkpoint_weight_intervention_development_only',
               'scope': 'same published checkpoint; observed DEV only; no fitting or new data',
               'seconds': time.perf_counter()-began, 'families': len({r['family'] for r in records}),
               'records': len(records), 'qualification': summaries, 'execution': executions,
               'new_reader_trained': False, 'data_generated': False,
               'fresh_diagnostic_opened': False, 'original_confirmation_opened': False}
    write(output/'summary.json', summary)
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    run(args.config, args.source, args.output)


if __name__ == '__main__':
    main()
