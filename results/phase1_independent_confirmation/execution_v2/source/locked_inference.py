"""Exact accepted development inference, parameterized by fresh train-only assets.

No hidden simulator state, targets, pair/sibling IDs, or evidence-role annotations
are accepted by these inference functions. Wrong-content interventions assemble
the exposed frames upstream, exactly as the original sampler does.
"""
import numpy as np
import torch

from core import maps, constrain
from geometry import observed_forecast
from guard import allow_update
from kernel import predict as observation_predict
from gate_intervention import traced_forward


def gate_transport_off(model, batch):
    """The entire original transport projection is disabled, for every input."""
    handle = model.transport_values.register_forward_hook(
        lambda _module, _args, output: torch.zeros_like(output))
    try:
        with torch.inference_mode():
            return model(batch)
    finally:
        handle.remove()


def gate_pair_with_transport(model, batch):
    """Exact accepted intervention plus a primitive vector for independent replay."""
    before, trace = traced_forward(model, batch)
    off, other = traced_forward(model, batch, True)
    if any(not torch.equal(a,b) for name in trace for a,b in zip(trace[name],other[name])):
        raise ValueError('Transport ablation changed an upstream output')
    history = (batch['mask'] & (batch['type']==1)).any(1)
    transported = (torch.stack(trace['transport'],1) @ model.output_basis.T)*history[:,None,None]
    if float((before-off-transported).abs().max()) >= 3e-5:
        raise ValueError('Full transport subtraction does not match the accepted branch intervention')
    if (~history).any() and not torch.equal(before[~history],off[~history]):
        raise ValueError('Empty-history reference identity changed')
    return before.numpy(),off.numpy(),transported.numpy()


class MotionInference:
    def __init__(self, basis, decoder, heads):
        self.decoder = decoder
        self.B, self.A, self.b, self.right, _ = maps(
            basis['output_basis'], decoder.weight, decoder.std, decoder.mean, decoder.bias)
        self.heads = heads
        if set(heads) != {3, 6}:
            raise ValueError('Exactly the two fixed fresh training-only maps are required')
        for h, head in heads.items():
            n = head['N']
            if (n.shape != (5120, 126) or np.max(np.abs(n.T @ self.A)) >= 1e-10
                    or float(head['effective_df']) > 32.000001):
                raise ValueError('Fixed position-null/df32 observation-map contract changed')

    def context(self, sample, times, retained_rgb, current_rgb, own_actions, query):
        """Only actually supplied retained observations and known actions enter."""
        times = tuple(map(int, times))
        if len(times) != 2 or not times[0] < times[1] < int(query)-3:
            raise ValueError('Exactly two distinct already-observed past frames required')
        if sample['z'].shape[0] != 6:
            raise ValueError('Context must be recent4 plus the actual two retained frames')
        enabled = allow_update(retained_rgb, current_rgb)
        xy = self.decoder.predict(sample['z'][4:])[:, 1:]
        geometry = np.stack([observed_forecast(xy, times,
            own_actions[times[1]:query+h], query+h)[0] for h in (3, 6)])
        if not np.isfinite(geometry).all():
            raise ValueError('Nonfinite legal observed-geometry forecast')
        return enabled, geometry

    def correct(self, base, reference_history, sample, enabled, geometry):
        """Fixed same-history visual output, guarded position, equal innovation mix."""
        base = np.asarray(base)
        history = np.asarray(reference_history)
        if base.shape != history.shape or base.shape != (2, 5120):
            raise ValueError('Full original two-horizon feature vectors required')
        current = constrain(base, geometry, self.B, self.A, self.b, self.right)[0] if enabled else base
        result = constrain(history, current @ self.A + self.b,
                           self.B, self.A, self.b, self.right)[0]
        if enabled:
            for i, horizon in enumerate((3, 6)):
                head = self.heads[horizon]
                coords = ((sample['current'] - result[i]) @ head['N']
                          + observation_predict(geometry[i][None], head)[0])
                delta = .5 * (coords @ head['N'].T)
                if np.max(np.abs(delta @ self.A)) >= 1e-10:
                    raise ValueError('Observation innovation altered decoded position')
                result[i] += delta
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite fixed repaired prediction')
        return result


def motion_context_features(sample, times, decoder, own_actions, query):
    """Training map input only; neither targets nor hidden trajectory are inputs."""
    times = tuple(map(int, times))
    if len(times) != 2 or not times[0] < times[1] < int(query)-3:
        raise ValueError('Original two observed training evidence frames required')
    xy = decoder.predict(sample['z'][4:])[:, 1:]
    return np.stack([observed_forecast(xy, times,
        own_actions[times[1]:query+h], query+h)[0] for h in (3, 6)])
