"""Boundaries for the no-fit, old-DEV-only checkpoint intervention."""
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.phase2 import run_archived_reader_intervention as intervention


def test_script_help_runs_by_absolute_path_outside_repo(tmp_path):
    result = subprocess.run([sys.executable, str(Path(intervention.__file__).resolve()), '--help'],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0
    assert '--source' in result.stdout and '--output' in result.stdout


@pytest.mark.parametrize('extra', [['--split', 'confirmation'], ['--device', 'cuda'], ['--train']])
def test_cli_has_no_training_gpu_or_split_override(extra):
    with pytest.raises(SystemExit) as exc:
        intervention.parser().parse_args(['--config', 'config.json', '--source', 'source', '--output', 'out']+extra)
    assert exc.value.code == 2


def test_existing_result_rejected_before_source_or_config_access(tmp_path, monkeypatch):
    output = tmp_path/'out'; output.mkdir(); (output/'result.json').write_text('{}')
    def forbidden(*args):
        raise AssertionError('Existing output reached artifact access')
    monkeypatch.setattr(intervention, 'read', forbidden)
    with pytest.raises(ValueError, match='Output must be empty'):
        intervention.run(tmp_path/'absent.json', tmp_path/'absent_source', output)


def test_paired_summary_rejects_different_evaluation_rows():
    row = {'id': 'one', 'arm': 'full', 'family': 'f', 'task_family': 'S', 'condition': 'neutral',
           'target_class': 0, 'semantic_complete': True, 'strict_correct': True, 'mse': .1}
    with pytest.raises(ValueError, match='not paired'):
        intervention.paired_summary([row], [{**row, 'id': 'two'}])
