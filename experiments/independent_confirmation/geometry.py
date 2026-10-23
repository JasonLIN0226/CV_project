"""Observed-input diagnostic, not a trained predictor or acceptance replacement."""
import numpy as np


def propagate(position, velocity, actions, steps, use_actions=True):
    p = np.asarray(position, dtype=np.float64).copy()
    v = float(velocity)
    reflections = 0
    for a in np.asarray(actions)[:steps]:
        x = float(p[0]) + v
        if use_actions:
            x += .0015 * float(a[0]) + .001 * float(a[1])
        while x < .08 or x > .92:
            if x < .08:
                x, v = .16-x, -v
                reflections += 1
            if x > .92:
                x, v = 1.84-x, -v
                reflections += 1
        p[0] = x
    if len(actions) < steps:
        raise ValueError('Incomplete known action sequence')
    return p, reflections


def observed_forecast(observed_positions, times, actions_after_end, target_time, use_actions=True):
    positions = np.asarray(observed_positions, dtype=np.float64)
    times = tuple(map(int, times))
    assert positions.shape == (2, 2) and times[1] > times[0] and target_time > times[1]
    velocity = float((positions[1, 0]-positions[0, 0])/(times[1]-times[0]))
    result, reflections = propagate(positions[1], velocity, actions_after_end, target_time-times[1], use_actions)
    return result, velocity, reflections
