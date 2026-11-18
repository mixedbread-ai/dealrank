"""Proposition 1's numbers, as the paper prints them, Proposition 2's call counts and the Mallows model's limits."""

from __future__ import annotations

from fractions import Fraction

import pytest

from repro.analysis import expected_cuts as ec


@pytest.mark.parametrize(("R", "chunk", "deal"), [(20, "6.0", "0.00"), (30, "2.9", "0.05"), (50, "1.1", "0.10")])
def test_chunks_and_dealing_by_depth(R: int, chunk: str, deal: str) -> None:
    assert f"{float(ec.l_chunk(R)):.1f}" == chunk
    assert f"{float(ec.l_deal(R)):.2f}" == deal


def test_extremes_are_exact() -> None:
    assert ec.l_chunk(20) == 6  # k - q, every top-ten passage in one chunk
    assert all(ec.l_deal(r) == 0 for r in range(10, 21))  # R <= H * q
    assert ec.l_deal(100) == ec.l_chunk(100) == ec.l_rand()  # groups of w agree at R = N


def test_dealing_below_random_below_chunks_at_every_depth() -> None:
    rand = ec.l_rand()
    for r in range(10, 101):
        assert ec.l_deal(r) <= rand <= ec.l_chunk(r)


def test_random_partition() -> None:
    assert f"{float(ec.l_rand()):.3f}" == "0.149"
    assert f"{100 * float(ec.p_cut_random()):.1f}" == "12.7"
    assert f"{float(ec.l_rand(10, 10, 10, 2)):.2f}" == "0.69"
    assert f"{float(ec.l_rand(20, 5, 10, 1)):.2f}" == "1.68"


def test_hypergeometric_excess_matches_enumeration() -> None:
    from itertools import combinations

    R, c, k, q = 8, 5, 4, 2
    subsets = list(combinations(range(R), k))
    brute = Fraction(sum(max(sum(1 for i in s if i < c) - q, 0) for s in subsets), len(subsets))
    assert ec.g(R, c, k, q) == brute


def test_proposition_2() -> None:
    assert ec.calls_one_round() == 30
    assert ec.calls_two_rounds() == 6


def test_mallows_limits() -> None:
    np = pytest.importorskip("numpy")
    from repro.analysis import mallows

    near_random = np.array([1 - 1e-9])
    for split in ("deal", "chunk"):
        assert mallows.expected_cut(near_random, split)[0] == pytest.approx(float(ec.l_rand()), abs=1e-6)
    assert mallows.mean_rank(near_random)[0] == pytest.approx(50.5, abs=1e-5)
    near_perfect = np.array([1e-4])
    assert mallows.expected_cut(near_perfect, "chunk")[0] == pytest.approx(6.0, abs=1e-3)
    assert mallows.expected_cut(near_perfect, "deal")[0] == pytest.approx(0.0, abs=1e-6)


def test_mallows_fit_recovers_a_point_mass() -> None:
    pytest.importorskip("numpy")
    from repro.analysis import mallows

    # Every query's top ten at the retriever's top ten: the fitted dispersion is as small as allowed.
    theta = mallows.fit([list(range(1, 11))] * 50, hi=2.0)
    assert theta > 1.9


def test_a_ranker_that_keeps_up_to_its_quota_gives_proposition_1() -> None:
    from repro.analysis import retention

    # a(x) = min(x, 4): a consistent ranker, which loses exactly what is over quota.
    a = [Fraction(min(x, 4)) for x in range(11)]
    by_depth = retention.losses(a, range(10, 101, 10))
    for depth, v in by_depth.items():
        assert v["deal"] == pytest.approx(float(ec.l_deal(depth)))
        assert v["chunk"] == pytest.approx(float(ec.l_chunk(depth)))
        assert v["deal_is_optimal"]
    assert retention.second_differences(a)[4] == -1
