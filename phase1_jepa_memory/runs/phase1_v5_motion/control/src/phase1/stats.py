import math
import numpy as np


def clustered_interval(values, groups, repeats=1000, seed=0):
    """Average within base pair first, bootstrap pairs (never frames or siblings)."""
    grouped = {}
    for value, group in zip(values, groups):
        grouped.setdefault(group, []).append(float(value))
    means = np.array([np.mean(v) for v in grouped.values()])
    rng = np.random.default_rng(seed)
    boot = means[rng.integers(0, len(means), (repeats, len(means)))].mean(1)
    return {"mean": float(means.mean()), "low": float(np.quantile(boot, 0.025)),
            "high": float(np.quantile(boot, 0.975)), "pairs": len(means)}


def wilson_pair_bound(mean, n):
    # Conservative effective-n companion to clustered bootstrap for bounded pair accuracy.
    z = 1.959963984540054
    center = (mean+z*z/(2*n))/(1+z*z/n)
    radius = z*math.sqrt(mean*(1-mean)/n+z*z/(4*n*n))/(1+z*z/n)
    return [max(0., center-radius), min(1., center+radius)]


def paired_improvement(base, alternative, groups, repeats, floor, absolute_gain, seed=0):
    """Positive = better. Ratio of aggregate errors, paired bootstrap of numerator AND denominator."""
    grouped = {}
    for a, b, g in zip(base, alternative, groups):
        grouped.setdefault(g, []).append([a, b])
    pair = np.array([np.mean(v, axis=0) for v in grouped.values()])
    use_absolute = pair[:, 0].mean() < floor
    def effect(x):
        a, b = x[..., 0].mean(-1), x[..., 1].mean(-1)
        return a-b if use_absolute else (a-b)/np.maximum(a, floor)
    rng = np.random.default_rng(seed)
    boot = effect(pair[rng.integers(0, len(pair), (repeats, len(pair)))])
    return {"mean": float(effect(pair)), "low": float(np.quantile(boot, .025)),
            "high": float(np.quantile(boot, .975)), "pairs": len(pair),
            "scale": "absolute_error_reduction" if use_absolute else "relative_error_reduction",
            "absolute_threshold": absolute_gain if use_absolute else None,
            "baseline_error": float(pair[:, 0].mean())}
