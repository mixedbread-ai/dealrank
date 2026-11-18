from __future__ import annotations

import pytest

from dealrank.partition import (
    chunks,
    count_cuts,
    deal,
    dealt_order,
    is_prefix_balanced,
    max_load,
    num_hands,
    random_partition,
    split,
)


def test_deal_gives_hand_h_every_hth_rank() -> None:
    hands = deal(100, 5)
    assert len(hands) == 5
    for h, hand in enumerate(hands):
        assert hand == list(range(h, 100, 5))


def test_deal_puts_four_of_the_top_twenty_in_each_of_five_hands() -> None:
    hands = deal(100, 5)
    assert [sum(i < 20 for i in hand) for hand in hands] == [4] * 5
    assert max_load(hands, 20) == 4


def test_chunks_are_contiguous_and_put_the_top_twenty_in_one_group() -> None:
    groups = chunks(100, 5)
    assert groups == [list(range(20 * h, 20 * (h + 1))) for h in range(5)]
    assert max_load(groups, 20) == 20


def test_chunks_drop_empty_trailing_groups() -> None:
    assert chunks(7, 4) == [[0, 1], [2, 3], [4, 5], [6]]
    assert chunks(9, 4) == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]


@pytest.mark.parametrize("n,hands", [(100, 5), (99, 5), (123, 7), (1000, 50), (3, 5)])
def test_every_policy_partitions_all_candidates(n: int, hands: int) -> None:
    for policy in ("deal", "chunks", "random"):
        groups = split(policy, n, hands, seed=3)
        assert sorted(i for g in groups for i in g) == list(range(n))
        assert all(g == sorted(g) for g in groups)
        assert all(groups)


def test_deal_is_prefix_balanced_but_chunks_and_random_are_not() -> None:
    assert is_prefix_balanced(deal(100, 5), 100)
    assert not is_prefix_balanced(chunks(100, 5), 100)
    assert not is_prefix_balanced(random_partition(100, 5, seed=0), 100)


def test_random_partition_is_reproducible_and_pinned() -> None:
    # Pinned values: changing the generator would change the paper's random-partition runs.
    assert random_partition(10, 3, seed=0) == [[2, 5, 6, 7], [0, 3, 8], [1, 4, 9]]
    assert random_partition(20, 4, seed=7) == [
        [1, 7, 14, 16, 17],
        [0, 6, 8, 12, 15],
        [4, 9, 11, 13, 19],
        [2, 3, 5, 10, 18],
    ]
    assert random_partition(100, 5, seed=1) == random_partition(100, 5, seed=1)
    assert random_partition(100, 5, seed=1) != random_partition(100, 5, seed=2)


def test_random_partition_has_deal_sized_groups() -> None:
    assert [len(g) for g in random_partition(103, 5, seed=0)] == [len(g) for g in deal(103, 5)]


def test_dealt_order_lays_hands_out_one_after_another() -> None:
    order = dealt_order(100, 20)
    assert order[:20] == list(range(0, 100, 5))
    assert order[20:40] == list(range(1, 100, 5))
    assert sorted(order) == list(range(100))
    assert dealt_order(0, 20) == []


def test_num_hands() -> None:
    assert num_hands(100, 20) == 5
    assert num_hands(101, 20) == 6
    assert num_hands(1000, 20) == 50
    with pytest.raises(ValueError):
        num_hands(10, 0)


def test_count_cuts_matches_the_quota_formula() -> None:
    top10 = range(10)
    assert count_cuts(deal(100, 5), top10, quota=4) == 0
    assert count_cuts(chunks(100, 5), top10, quota=4) == 6
    # A target set deeper than the deal's guarantee (top H*q = 20) can be cut.
    assert count_cuts(deal(100, 5), [0, 5, 10, 15, 20], quota=4) == 1


def test_split_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError, match="unknown split"):
        split("stride", 10, 2)
