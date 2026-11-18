"""The paired hierarchical bootstrap: determinism and cases with a known answer."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from repro.stats import N_RESAMPLES, bootstrap_mean, bootstrap_two_sided_p, hier_boot, paired  # noqa: E402


def test_deterministic_and_order_free() -> None:
    rng = np.random.default_rng(0)
    diffs = {b: rng.normal(0.3, 1.0, size=n) for b, n in (("x", 40), ("a", 25), ("m", 60))}
    one = hier_boot(diffs)
    two = hier_boot(dict(reversed(list(diffs.items()))))
    assert np.array_equal(one["boot"], two["boot"])
    assert (one["ci_lo"], one["ci_hi"], one["p_two_sided"]) == (two["ci_lo"], two["ci_hi"], two["p_two_sided"])


def test_macro_is_unweighted_over_benchmarks() -> None:
    r = hier_boot({"a": [1.0] * 100, "b": [0.0, 0.0]})
    assert r["diff"] == pytest.approx(0.5)
    assert r["n_datasets"] == 2 and r["n_queries"] == 102


def test_constant_difference() -> None:
    r = hier_boot({"a": [2.0] * 5, "b": [2.0] * 7})
    assert r["ci_lo"] == pytest.approx(2.0) and r["ci_hi"] == pytest.approx(2.0)
    assert r["p_two_sided"] == 1.0 / N_RESAMPLES  # every replicate sits at the mean: floored


def test_one_benchmark_matches_a_plain_bootstrap() -> None:
    d = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
    r = hier_boot({"only": d}, n_resamples=2000, seed=7)
    rng = np.random.default_rng(7)
    within = d[rng.integers(0, 5, size=(2000, 5))].mean(axis=1)  # one benchmark: picking it changes nothing
    assert np.allclose(r["boot"], within)
    assert r["diff"] == pytest.approx(0.6)


def test_a_benchmark_drawn_twice_reuses_its_query_resample() -> None:
    # One query resample per benchmark per replicate, reused when the benchmark is drawn more than once.
    d = {"a": np.array([-1.0, 1.0]), "b": np.array([-1.0, 1.0])}
    r = hier_boot(d, n_resamples=20_000, seed=3)
    rng = np.random.default_rng(3)
    within = np.stack([d[name][rng.integers(0, 2, size=(20_000, 2))].mean(axis=1) for name in ("a", "b")])
    pick = rng.integers(0, 2, size=(20_000, 2))
    assert np.allclose(r["boot"], within[pick, np.arange(20_000)[:, None]].mean(axis=1))
    # Each benchmark mean has variance 1/2. Reusing the resample gives the macro mean a variance of
    # 3/8; redrawing the queries for every occurrence of a benchmark would give 1/4.
    assert r["boot"].var() == pytest.approx(0.375, abs=0.01)


def test_two_sided_p() -> None:
    reps = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert bootstrap_two_sided_p(reps, 1.0, center="mean") == pytest.approx(0.8)
    assert bootstrap_two_sided_p(reps, 3.0, center="mean", minimum=0.01) == 0.01


def test_paired_uses_shared_queries_in_sorted_order() -> None:
    a = {"b1": {"q2": 1.0, "q1": 0.5, "q3": 0.0}, "b2": {"q9": 0.2}}
    b = {"b1": {"q1": 0.5, "q2": 0.0}, "b3": {"q1": 1.0}}
    r = paired(a, b, scale=100)
    assert r["benchmarks"] == 1 and r["queries"] == 2
    assert r["diff"] == pytest.approx(50.0)


def test_bootstrap_mean() -> None:
    r = bootstrap_mean([1.0, 1.0, 1.0])
    assert r == {"mean": 1.0, "ci": [1.0, 1.0], "n": 3}
