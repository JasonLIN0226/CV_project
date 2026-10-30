"""Reader-only aggregation ablations; the input/target adapter stays unchanged.

``original`` returns the unmodified DynamicsPredictor, including its checkpoint
keys and forward implementation. Other arms change only aggregation/readout
capacity. All consume the same causal observation, age, executed-action and
current-context tensors, plus the proposed action plan used by the predictor.
No role, target, outcome, task ID or future observation is read here.

Independent item attention logits do not imply that a pooled nonlinear reader
cannot represent relations. These are empirical architecture comparisons.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

import torch
from torch import nn

CONTROL_SRC = Path(__file__).resolve().parents[1] / 'predictor' / 'control' / 'src'
if str(CONTROL_SRC) not in sys.path:
    sys.path.insert(0, str(CONTROL_SRC))
from phase1.model import DynamicsPredictor


VARIANTS = ('original', 'attention_mean', 'multihead', 'item_attention', 'wide_control')
INTERVENTIONS = ('uniform_intervention',)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _output(input_dim, hidden, output_dim, dropout):
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Dropout(dropout),
                         nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, output_dim))


def _masked_softmax(logits, history):
    """[B,H,N] weights; no-history rows are exactly zero, including backward."""
    if logits.shape[-1] == 0:
        return torch.zeros_like(logits)
    mask = history[:, None, :]
    safe = logits.masked_fill(~mask, -torch.inf)
    safe = torch.where(mask.any(-1, keepdim=True), safe, torch.zeros_like(safe))
    return torch.softmax(safe, -1).masked_fill(~mask, 0)


class AggregationPredictor(DynamicsPredictor):
    """Same basis/action modules, with explicit history aggregation variants."""
    def __init__(self, dim, config, horizons, basis, variant, heads=4):
        if variant not in ('attention_mean', 'multihead', 'item_attention'):
            raise ValueError('Use build_reader for original and wide_control')
        super().__init__(dim, config, horizons, basis)
        self.aggregation_variant = variant
        self.read_heads = heads if variant == 'multihead' else (2 if variant == 'attention_mean' else 1)
        d, hidden = self.projection.shape[1], config['hidden']
        if heads < 1:
            raise ValueError('heads must be positive')
        if variant == 'multihead':
            self.score = nn.Sequential(nn.Linear(2*d, hidden), nn.GELU(), nn.Linear(hidden, heads))
        if variant == 'item_attention':
            if d % heads:
                raise ValueError('Projected feature width must divide self-attention heads')
            self.item_age = nn.Linear(1, d, bias=False)
            self.item_norm1 = nn.LayerNorm(d)
            self.item_attention = nn.MultiheadAttention(d, heads, dropout=config['dropout'], batch_first=True)
            self.item_norm2 = nn.LayerNorm(d)
            self.item_ff = nn.Sequential(nn.Linear(d, 2*d), nn.GELU(),
                                         nn.Dropout(config['dropout']), nn.Linear(2*d, d))
        # Shared current/action/horizon, and six d-vectors plus two statistics
        # per aggregation path. Single-path feature order remains original.
        input_dim = 8*d+3 if self.read_heads == 1 else 2*d+1+self.read_heads*(6*d+2)
        if self.read_heads != 1:
            self.output = _output(input_dim, hidden, self.output_basis.shape[1], config['dropout'])

    def _observed(self, batch):
        valid = batch['mask']
        z = batch['z'].masked_fill(~valid[..., None], 0)
        p = batch['projected_z'] if 'projected_z' in batch else (z-self.input_mean)@self.projection
        p = p.masked_fill(~valid[..., None], 0)
        current = batch['projected_current'] if 'projected_current' in batch else (batch['current']-self.input_mean)@self.projection
        history = valid & (batch['type'] == 1)
        past = self.past_action(batch['past_actions'].masked_fill(~valid[..., None], 0))
        age = batch['age'].masked_fill(~valid, 0)/8
        aligned = p+past
        score_input = p
        if self.aggregation_variant == 'item_attention' and p.shape[1]:
            x = (aligned+self.item_age(age[..., None])).masked_fill(~history[..., None], 0)
            # MHA cannot receive an all-masked key row. Expose a zero dummy key
            # only for such rows, then erase their complete output and gradient.
            safe_history = history.clone()
            empty = ~history.any(1)
            safe_history[empty, 0] = True
            norm = self.item_norm1(x)
            delta, _ = self.item_attention(norm, norm, norm,
                                           key_padding_mask=~safe_history, need_weights=False)
            x = x+delta
            x = x+self.item_ff(self.item_norm2(x))
            aligned = x.masked_fill(~history[..., None], 0)
            score_input = aligned
        if self.aggregation_variant == 'uniform_intervention':
            weights = (history.float()/history.sum(1, keepdim=True).clamp_min(1))[:, None]
        else:
            score = self.score(torch.cat([score_input, score_input-current[:, None]], -1))
            salience = torch.log1p((score_input-current[:, None]).square().mean(-1))
            weights = _masked_softmax(score.transpose(1, 2)+salience[:, None], history)
        if self.aggregation_variant == 'attention_mean':
            uniform = history.float()/history.sum(1, keepdim=True).clamp_min(1)
            weights = torch.cat([weights, uniform[:, None]], 1)
        return current, history, past, age, aligned, weights

    def attention_weights(self, batch):
        """Actual pooling weights [B,paths,N], with zeros on every nonhistory item."""
        return self._observed(batch)[-1]

    def forward(self, batch):
        current, history, past, age, aligned, weights = self._observed(batch)
        summaries = []
        for w in weights.unbind(1):
            mean_age = (w*age).sum(1)
            center = age-mean_age[:, None]
            memory = (w[..., None]*aligned).sum(1)
            past_mean = (w[..., None]*past).sum(1)
            variance = (w*center.square()).sum(1)
            velocity = -(w[..., None]*center[..., None]*(aligned-memory[:, None])).sum(1)/variance.clamp_min(.01)[:, None]
            summaries.append((memory, velocity, past_mean, mean_age))
        outputs = []
        for h in self.horizons:
            actions = batch['actions'][:, :h]
            action = self.action(torch.cat([actions[:, 0], actions.sum(1), actions[:, -1]], -1))
            hv = torch.full_like(age.sum(1), h/8)
            if self.read_heads == 1:
                memory, velocity, past_mean, mean_age = summaries[0]
                transported = memory+velocity*(mean_age+h/8)[:, None]
                stats = torch.stack([mean_age, history.any(1).float(), hv], -1)
                features = torch.cat([current, memory, velocity, transported, action,
                                      memory*action, past_mean, velocity*action, stats], -1)
            else:
                parts = [current, action, hv[:, None]]
                for memory, velocity, past_mean, mean_age in summaries:
                    transported = memory+velocity*(mean_age+h/8)[:, None]
                    stats = torch.stack([mean_age, history.any(1).float()], -1)
                    parts.extend([memory, velocity, transported, memory*action,
                                  past_mean, velocity*action, stats])
                features = torch.cat(parts, -1)
            coeff = self.output(features)
            outputs.append(batch['current']+self.target_mean+coeff@self.output_basis.T)
        return torch.stack(outputs, 1)


class UniformInterventionPredictor(AggregationPredictor):
    """No-training intervention: load an original checkpoint with strict=True.

    All original parameters, buffers and state keys are preserved. Only weights
    are forced uniform; memory, mean age, velocity and transported summaries
    are then recomputed. Do not train this as a sixth architecture arm.
    """
    def __init__(self, dim, config, horizons, basis):
        DynamicsPredictor.__init__(self, dim, config, horizons, basis)
        self.aggregation_variant = 'uniform_intervention'
        self.read_heads = 1


def _wide_width(dim, config, horizons, basis, heads):
    """Nearest integer output width to the largest noncontrol arm's parameter count.

    fork_rng ensures this sizing query does not alter the eventual model's
    initialization or the caller's subsequent random training sequence.
    """
    with torch.random.fork_rng(devices=[]):
        base = DynamicsPredictor(dim, config, horizons, basis)
        target = max(parameter_count(AggregationPredictor(dim, config, horizons, basis, v, heads))
                     for v in VARIANTS[1:4])
        fixed = parameter_count(base)-parameter_count(base.output)
        n, r = base.output[0].in_features, base.output[-1].out_features
        # Output params = w^2 + (n+r+2)w + r.
        estimate = (-(n+r+2)+math.sqrt((n+r+2)**2+4*(target-fixed-r)))/2
        candidates = {max(1, int(math.floor(estimate))), max(1, int(math.ceil(estimate)))}
        width = min(candidates, key=lambda w: (abs(fixed+w*w+(n+r+2)*w+r-target), w))
    return width, target


def build_reader(dim, config, horizons, basis=None, variant='original', heads=4):
    """Factory used in both train and checkpoint reload; no data/label access.

    ``original`` state_dict and predictions are exactly legacy-compatible.
    ``wide_control`` keeps original score/action/aggregation and changes only
    both hidden widths in its output MLP. Save variant and heads with config.
    """
    if variant not in VARIANTS+INTERVENTIONS:
        raise ValueError(f'Unknown reader aggregation variant: {variant}')
    if variant == 'uniform_intervention':
        return UniformInterventionPredictor(dim, config, horizons, basis)
    if variant in ('original', 'wide_control'):
        width, target = _wide_width(dim, config, horizons, basis, heads) if variant == 'wide_control' else (None, None)
        model = DynamicsPredictor(dim, config, horizons, basis)
        model.aggregation_variant = variant
        model.read_heads = 1
        if width is not None:
            model.output = _output(model.output[0].in_features, width,
                                   model.output[-1].out_features, config['dropout'])
            model.capacity_target_parameters = target
        return model
    return AggregationPredictor(dim, config, horizons, basis, variant, heads)


def describe_reader(model):
    result = {'variant': model.aggregation_variant, 'trainable_parameters': parameter_count(model),
              'pooling_paths': model.read_heads, 'output_input_width': model.output[0].in_features,
              'output_hidden_width': model.output[0].out_features,
              'item_self_attention_heads': model.item_attention.num_heads if hasattr(model, 'item_attention') else 0,
              'horizons': list(model.horizons)}
    if model.aggregation_variant in INTERVENTIONS:
        result.update(no_training_intervention=True, requires_original_checkpoint=True)
    if hasattr(model, 'capacity_target_parameters'):
        target = model.capacity_target_parameters
        result.update(capacity_target_parameters=target,
                      capacity_relative_difference=(parameter_count(model)-target)/target)
    return result


def reader_attention(model, projected_candidates, projected_current, ages=None, past_actions=None):
    """Pooling weights averaged across paths for a supplied *observed* candidate set.

    Self-attention needs true ages/executed actions, because they condition its
    tokens. This is not a substitute for the future-query teacher protocol.
    The caller must identify the observed query/context and any action timing.
    """
    p, q = projected_candidates, projected_current
    if p.ndim != 3 or q.shape != (p.shape[0], p.shape[2]):
        raise ValueError('Expected projected candidates [B,N,d] and current [B,d]')
    if isinstance(model, AggregationPredictor):
        if model.aggregation_variant == 'item_attention' and (ages is None or past_actions is None):
            raise ValueError('Self-attention weights require observed ages and past_actions')
        b, n, _ = p.shape
        batch = {'z': p.new_zeros((b,n,len(model.input_mean))), 'projected_z': p,
                 'projected_current': q, 'current': p.new_zeros((b,len(model.input_mean))),
                 'mask': torch.ones((b,n), dtype=torch.bool, device=p.device),
                 'type': torch.ones((b,n), dtype=torch.long, device=p.device),
                 'age': p.new_zeros((b,n)) if ages is None else ages,
                 'past_actions': p.new_zeros((b,n,2)) if past_actions is None else past_actions}
        return model.attention_weights(batch).mean(1)
    score = model.score(torch.cat([p,p-q[:, None]], -1)).squeeze(-1)
    score = score+torch.log1p((p-q[:, None]).square().mean(-1))
    return torch.softmax(score, -1)
