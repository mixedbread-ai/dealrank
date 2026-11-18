"""The paired hierarchical bootstrap behind every interval in the paper.

A paired difference is resampled in two stages: queries with replacement
within each benchmark (in sorted benchmark order), then benchmarks with
replacement. The statistic is the macro mean, the unweighted mean over
benchmarks of each benchmark's query mean. 10,000 resamples with
``numpy.random.default_rng(42)``; 95% percentile intervals; the two-sided
p-value sums both tails of the bootstrap distribution centred at its mean,
floored at ``1 / 10,000``.

Each replicate draws one query resample per benchmark, and a benchmark that
the second stage selects several times contributes that same resampled mean
each time. ``tests/test_stats_bootstrap.py`` pins the construction.

The interval and the p-value are two summaries of the same replicates and can
disagree at the margin: the interval is the percentile interval, and the
p-value recentres the replicates at their mean, so an interval that just
excludes zero can come with a p-value just above 0.05.

The order in which random numbers are drawn is part of the estimator: pass
each benchmark's differences in sorted qid order to reproduce the paper's
intervals.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

SEED = 42
N_RESAMPLES = 10_000


def bootstrap_two_sided_p(
    replicates: np.ndarray, observed: float, *, center: str | float = "observed", minimum: float = 0.0
) -> float:
    """Two-sided p-value from a centred bootstrap distribution.

    ``P(T* - c <= -|T|) + P(T* - c >= |T|)``, where ``c`` is ``observed``,
    the replicates' mean (``center="mean"``) or a given number, clipped to
    ``[minimum, 1]``.
    """
    reps = np.asarray(replicates, dtype=float)
    if reps.ndim != 1 or reps.size == 0:
        raise ValueError("replicates must be a non-empty 1-D array")
    if center == "observed":
        c = float(observed)
    elif center == "mean":
        c = float(reps.mean())
    else:
        c = float(center)
    t = abs(float(observed))
    centred = reps - c
    p = float(np.mean(centred <= -t) + np.mean(centred >= t))
    return max(min(p, 1.0), minimum)


def hier_boot(
    diffs: Mapping[str, Sequence[float] | np.ndarray], n_resamples: int = N_RESAMPLES, seed: int = SEED
) -> dict[str, Any]:
    """Hierarchical (benchmark, then query) bootstrap of the macro mean of ``diffs``.

    Returns ``diff`` (the observed macro mean), ``ci_lo``, ``ci_hi``,
    ``p_two_sided``, ``n_datasets``, ``n_queries`` and the replicates as ``boot``.
    """
    arrays = {k: np.asarray(v, dtype=float) for k, v in diffs.items()}
    if not arrays or any(a.size == 0 for a in arrays.values()):
        raise ValueError("every benchmark needs at least one difference")
    rng = np.random.default_rng(seed)
    names = sorted(arrays)
    within = np.empty((len(names), n_resamples))
    for i, name in enumerate(names):
        d = arrays[name]
        idx = rng.integers(0, len(d), size=(n_resamples, len(d)))
        within[i] = d[idx].mean(axis=1)
    pick = rng.integers(0, len(names), size=(n_resamples, len(names)))
    boot = within[pick, np.arange(n_resamples)[:, None]].mean(axis=1)
    observed = float(np.mean([a.mean() for a in arrays.values()]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "diff": observed,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_two_sided": bootstrap_two_sided_p(boot, observed, center="mean", minimum=1.0 / n_resamples),
        "n_datasets": len(names),
        "n_queries": int(sum(a.size for a in arrays.values())),
        "boot": boot,
    }


def paired(
    a: Mapping[str, Mapping[str, float]], b: Mapping[str, Mapping[str, float]], scale: float = 1.0
) -> dict[str, Any] | None:
    """Macro mean of ``a - b`` over the queries both have, with its hierarchical bootstrap.

    ``a`` and ``b`` map benchmark to ``{qid: value}``. Benchmarks without a
    shared query are left out. ``scale`` multiplies the reported difference
    and interval (e.g. 100 for nDCG@10 points). Returns ``None`` when nothing
    is shared.
    """
    diffs = {}
    for name in a:
        if name not in b:
            continue
        common = sorted(set(a[name]) & set(b[name]))
        if common:
            diffs[name] = np.array([a[name][q] - b[name][q] for q in common])
    if not diffs:
        return None
    return summary(hier_boot(diffs), scale)


def summary(boot: Mapping[str, Any], scale: float = 1.0) -> dict[str, Any]:
    """The JSON-ready part of :func:`hier_boot`'s result, scaled."""
    return {
        "diff": scale * boot["diff"],
        "ci": [scale * boot["ci_lo"], scale * boot["ci_hi"]],
        "p": boot["p_two_sided"],
        "benchmarks": boot["n_datasets"],
        "queries": boot["n_queries"],
    }


def bootstrap_mean(values: Sequence[float], n_resamples: int = N_RESAMPLES, seed: int = SEED) -> dict[str, Any]:
    """Mean of ``values`` with a 95% percentile interval from a plain query bootstrap."""
    vals = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = vals[rng.integers(0, len(vals), size=(n_resamples, len(vals)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean": float(vals.mean()), "ci": [float(lo), float(hi)], "n": int(vals.size)}
