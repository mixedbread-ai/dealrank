"""DealRank: parallel listwise reranking with dealt splits.

>>> from dealrank import DealRank, deal
>>> deal(10, 3)
[[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]]
"""

from __future__ import annotations

from dealrank.llm import FakeLLM, ListwiseLLM, ResponseCache
from dealrank.partition import chunks, count_cuts, deal, dealt_order, num_hands, random_partition, split
from dealrank.prompt import parse_permutation, rankgpt_messages
from dealrank.result import Call, RerankResult
from dealrank.schedules import (
    BlitzRank,
    DealRank,
    HierarchicalDealRank,
    JointRank,
    ListK,
    Reordered,
    RetrieverOrder,
    Schedule,
    SlidingWindow,
    TDPart,
    TopPrefix,
    TourRank,
    build_schedule,
)

__version__ = "0.1.0"

__all__ = [
    "BlitzRank",
    "Call",
    "DealRank",
    "FakeLLM",
    "HierarchicalDealRank",
    "JointRank",
    "ListK",
    "ListwiseLLM",
    "Reordered",
    "RerankResult",
    "ResponseCache",
    "RetrieverOrder",
    "Schedule",
    "SlidingWindow",
    "TDPart",
    "TopPrefix",
    "TourRank",
    "__version__",
    "build_schedule",
    "chunks",
    "count_cuts",
    "deal",
    "dealt_order",
    "num_hands",
    "parse_permutation",
    "random_partition",
    "rankgpt_messages",
    "split",
]
