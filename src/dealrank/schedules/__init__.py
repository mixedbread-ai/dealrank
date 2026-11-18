"""Reranking schedules.

DealRank and BlitzRank on dealt input are the schedules the paper recommends.
The others are the baselines it compares against, all sharing the same
prompt, parser and LLM interface.
"""

from __future__ import annotations

import inspect
from typing import Any

from dealrank.schedules.base import Reordered, Schedule, input_permutation, run_sync
from dealrank.schedules.blitzrank import BlitzRank
from dealrank.schedules.dealrank import DealRank, HierarchicalDealRank
from dealrank.schedules.jointrank import JointRank
from dealrank.schedules.listk import ListK
from dealrank.schedules.simple import RetrieverOrder, TopPrefix
from dealrank.schedules.sliding_window import SlidingWindow
from dealrank.schedules.tdpart import TDPart
from dealrank.schedules.tourrank import TourRank

SCHEDULES: dict[str, type[Schedule]] = {
    "dealrank": DealRank,
    "dealrank_hierarchical": HierarchicalDealRank,
    "blitzrank": BlitzRank,
    "sliding_window": SlidingWindow,
    "tourrank": TourRank,
    "tdpart": TDPart,
    "jointrank": JointRank,
    "listk": ListK,
    "top_prefix": TopPrefix,
    "retriever": RetrieverOrder,
}


def build_schedule(spec: dict[str, Any]) -> Schedule:
    """Build a schedule from a mapping such as ``{"type": "dealrank", "quota": 4}``.

    An ``input_order`` key (``"retriever"`` or ``"dealt"``) is passed to
    schedules that take one (BlitzRank) and otherwise wraps the schedule in
    :class:`Reordered`, with ``input_window`` as its ``window``.
    """
    spec = dict(spec)
    kind = spec.pop("type")
    try:
        cls = SCHEDULES[kind]
    except KeyError:
        raise ValueError(f"unknown schedule {kind!r}; expected one of {sorted(SCHEDULES)}") from None
    if "input_order" in inspect.signature(cls).parameters:
        return cls(**spec)  # the schedule reorders its input itself (BlitzRank)
    order = spec.pop("input_order", None)
    window = spec.pop("input_window", 20)
    schedule = cls(**spec)
    if order is not None and order != "retriever":
        schedule = Reordered(schedule, order, window=window)
    return schedule


__all__ = [
    "SCHEDULES",
    "BlitzRank",
    "DealRank",
    "HierarchicalDealRank",
    "JointRank",
    "ListK",
    "Reordered",
    "RetrieverOrder",
    "Schedule",
    "SlidingWindow",
    "TDPart",
    "TopPrefix",
    "TourRank",
    "build_schedule",
    "input_permutation",
    "run_sync",
]
