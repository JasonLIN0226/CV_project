"""Software checks for the fixed repair; does not read scientific data."""
import ast
import math
from pathlib import Path
import torch
import pytest
from phase1.repair_trial import learning_rate_factor, trainable_parameters, actual_allocation_start


SOURCE = Path(__file__).parents[1] / 'provisional_control/src/phase1'


def test_schedule_has_no_validation_or_seed_dependency():
    rates = [learning_rate_factor(i) for i in range(120)]
    assert rates[:5] == [.2, .4, .6, .8, 1.]
    assert all(a >= b for a, b in zip(rates[4:], rates[5:]))
    assert math.isclose(rates[-1], .1)
    assert min(rates) >= .1 and max(rates) <= 1.


def test_reference_parameters_stay_outside_optimizer():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 1))
    model[0].requires_grad_(False)
    actual = {id(p) for p in trainable_parameters(model)}
    assert actual == {id(p) for p in model[1].parameters()}


def test_scientific_runtime_rejects_cpu_before_any_work(monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', '12345')
    monkeypatch.setenv('SLURM_JOB_ACCOUNT', 'csci_ga_2271-2026fa')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(ValueError, match='no CPU training fallback'):
        actual_allocation_start()


def test_actual_requested_gpu_wall_cannot_exceed_remaining_human_cap():
    text = (SOURCE.parents[1] / 'repair_train.sbatch').read_text()
    setting = next(line.split('=', 1)[1] for line in text.splitlines() if line.startswith('#SBATCH --time='))
    hours, minutes, seconds = map(int, setting.split(':'))
    assert 6247 + 3600 * hours + 60 * minutes + seconds <= 18000


def definition(path, name):
    return next(node for node in ast.parse(path.read_text()).body if getattr(node, 'name', None) == name)


def test_original_training_sampling_losses_and_counts_unchanged():
    old = definition(SOURCE / 'proposal_trial.py', 'train_epoch')
    new = definition(SOURCE / 'repair_trial.py', 'train_epoch')
    class NormalizeGradientParameterBoundary(ast.NodeTransformer):
        def visit_Expr(self, node):
            if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr == 'clip_grad_norm_'):
                return ast.parse("CLIP_ONLY_EXPLICIT_TRAINABLE_PARAMETERS()").body[0]
            return self.generic_visit(node)
    transform = NormalizeGradientParameterBoundary()
    assert ast.dump(transform.visit(old), include_attributes=False) == ast.dump(transform.visit(new), include_attributes=False)


def test_full_evaluation_loop_and_original_passing_rules_unchanged():
    old = ast.parse((SOURCE / 'proposal_evaluation.py').read_text())
    new = ast.parse((SOURCE / 'repair_evaluation.py').read_text())
    class NormalizeLoader(ast.NodeTransformer):
        def visit_Constant(self, node):
            if node.value in {'AnchoredMemoryPredictor', 'RelationalMemoryPredictor'}:
                node.value = 'FixedReader'
            return node
        def visit_Call(self, node):
            if (isinstance(node.func, ast.Name) and node.func.id == 'code_hash' and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value in {'proposal_evaluation.py', 'repair_evaluation.py'}):
                # Only provenance file identities differ; the executable metric
                # and threshold statements outside these strings must agree.
                node.args = [ast.Constant(value='CURRENT_EXECUTABLE_SOURCES')]
            return self.generic_visit(node)
        def visit_ImportFrom(self, node):
            if node.module in {'proposal_trial', 'repair_trial'}:
                node.module = 'trial_loader'
            if node.module in {'anchored_memory', 'relational_memory'}:
                node.module = 'reader_loader'
            for alias in node.names:
                if alias.name in {'AnchoredMemoryPredictor', 'RelationalMemoryPredictor'}:
                    alias.name = 'FixedReader'
            return node
        def visit_Name(self, node):
            if node.id in {'AnchoredMemoryPredictor', 'RelationalMemoryPredictor'}:
                node.id = 'FixedReader'
            return node
    # The module docstring is editorial; all executable statements must agree.
    old.body.pop(0); new.body.pop(0)
    transform = NormalizeLoader()
    assert ast.dump(transform.visit(old), include_attributes=False) == ast.dump(transform.visit(new), include_attributes=False)
