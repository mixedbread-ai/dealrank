"""A Mallows retriever model with query-varying dispersion (Figure 2's dashed lines).

Under a Mallows model with dispersion ``phi`` (``phi -> 0``: a perfect
retriever, ``phi = 1``: a random list), the retriever ranks ``T`` of a
consistent ranker's top ``k`` among ``n`` candidates have
``P(T) ∝ phi^sum(T)``. The weight factorises over groups, so the expected cut
of a split follows from elementary symmetric polynomials of ``phi^t`` over
each group's ranks.

Dispersion varies by query: ``theta = -log(phi)`` is exponential with mean
``theta_mean``, represented by 200 equal-probability nodes. Figure 2 traces
the model over ``theta_mean`` (:func:`curve`), against the mean retriever
rank it implies. :func:`fit` gives the maximum-likelihood ``theta_mean`` for
the observed ``T`` of each query (the retriever ranks of the sliding window's
top ten); nothing about the cuts enters either.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

N, H, W, Q, K = 100, 5, 20, 4, 10
NODES = 200


def nodes(theta_mean: float, g: int = NODES) -> np.ndarray:
    """``phi`` at the ``g`` mid-quantiles of ``theta ~ Exp(theta_mean)``."""
    u = (np.arange(g) + 0.5) / g
    return np.exp(theta_mean * np.log1p(-u))


def _esp(values: np.ndarray, k: int) -> np.ndarray:
    """Elementary symmetric polynomials ``e_0..e_k`` of each row of ``values`` (shape ``(m, len)``)."""
    e = np.zeros((values.shape[0], k + 1))
    e[:, 0] = 1.0
    for j in range(values.shape[1]):
        v = values[:, j : j + 1]
        e[:, 1:] = e[:, 1:] + e[:, :-1] * v
    return e


def log_z(phis: np.ndarray, n: int = N, k: int = K) -> np.ndarray:
    """``log sum_T phi^sum(T)`` over the ``k``-subsets of ``1..n``, for each ``phi``."""
    t = np.arange(n)
    e = _esp(phis[:, None] ** t[None, :], k)  # weights phi^(t - 1), rescaled by phi^-k
    return k * np.log(phis) + np.log(e[:, k])


def neg_log_likelihood(theta_mean: float, rank_sums: np.ndarray, n: int = N, k: int = K) -> float:
    """Negative log-likelihood of the exponential mixture given each query's ``sum(T)``."""
    phis = nodes(theta_mean)
    lp = rank_sums[:, None] * np.log(phis)[None, :] - log_z(phis, n, k)[None, :]
    top = lp.max(axis=1, keepdims=True)
    return float(-np.sum(top[:, 0] + np.log(np.mean(np.exp(lp - top), axis=1))))


def fit(ranks: Sequence[Sequence[int]], n: int = N, lo: float = 0.005, hi: float = 0.5, tol: float = 1e-8) -> float:
    """Maximum-likelihood ``theta_mean`` for queries whose top ``k`` sit at retriever ranks ``ranks`` (1-based).

    Golden-section search on ``[lo, hi]``; the likelihood is unimodal in ``theta_mean``.
    """
    sums = np.array([sum(r) for r in ranks], dtype=float)
    k = len(ranks[0])
    ratio = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc, fd = neg_log_likelihood(c, sums, n, k), neg_log_likelihood(d, sums, n, k)
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = neg_log_likelihood(c, sums, n, k)
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = neg_log_likelihood(d, sums, n, k)
    return (a + b) / 2


def groups(split: str, n: int = N, hands: int = H) -> list[list[int]]:
    """1-based retriever ranks of each group: ``"deal"`` or ``"chunk"``."""
    if split == "deal":
        return [list(range(h + 1, n + 1, hands)) for h in range(hands)]
    size = n // hands
    return [list(range(h * size + 1, (h + 1) * size + 1)) for h in range(hands)]


def expected_cut(phis: np.ndarray, split: str, n: int = N, hands: int = H, k: int = K, q: int = Q) -> np.ndarray:
    """Expected cut of the true top ``k`` by ``split`` under ``P(T) ∝ phi^sum(T)``, for each ``phi``."""
    total = np.zeros((len(phis), k + 1))
    total[:, 0] = 1.0
    lost = np.zeros_like(total)
    excess = np.maximum(np.arange(k + 1) - q, 0)
    for ranks in groups(split, n, hands):
        w = _esp(phis[:, None] ** (np.array(ranks)[None, :] - 1), k)
        new_total, new_lost = np.zeros_like(total), np.zeros_like(lost)
        for s in range(k + 1):
            for c in range(k + 1 - s):
                new_total[:, s + c] += total[:, s] * w[:, c]
                new_lost[:, s + c] += lost[:, s] * w[:, c] + total[:, s] * w[:, c] * excess[c]
        total, lost = new_total, new_lost
    return lost[:, k] / total[:, k]


def mean_rank(phis: np.ndarray, n: int = N, k: int = K) -> np.ndarray:
    """Mean retriever rank of the true top ``k``, ``E sum(T) / k = phi d/dphi log Z / k``, for each ``phi``."""
    e = np.zeros((len(phis), k + 1))
    de = np.zeros_like(e)
    e[:, 0] = 1.0
    for t in range(1, n + 1):
        v = (phis**t)[:, None]
        dv = (t * phis ** (t - 1))[:, None]
        de[:, 1:] = de[:, 1:] + de[:, :-1] * v + e[:, :-1] * dv
        e[:, 1:] = e[:, 1:] + e[:, :-1] * v
    return phis * de[:, k] / e[:, k] / k


def prediction(theta_mean: float) -> dict[str, float]:
    """Expected cuts of chunks and dealing and the implied mean rank, averaged over the mixture."""
    phis = nodes(theta_mean)
    return {
        "theta_mean": float(theta_mean),
        "chunk": float(np.mean(expected_cut(phis, "chunk"))),
        "deal": float(np.mean(expected_cut(phis, "deal"))),
        "mean_rank_top10": float(np.mean(mean_rank(phis))),
    }


def curve(lo: float = 0.004, hi: float = 0.6, points: int = 36) -> list[dict[str, float]]:
    """:func:`prediction` over a log-spaced grid of ``theta_mean`` (Figure 2's dashed curves)."""
    return [prediction(float(t)) for t in np.exp(np.linspace(np.log(lo), np.log(hi), points))]
