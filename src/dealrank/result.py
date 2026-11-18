"""Records of LLM calls and reranking results."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Call:
    """One listwise LLM call.

    ``permutation`` holds 0-based positions into the passages the call was
    given, best first. ``latency_s`` is the duration of the successful request
    and ``started_at`` / ``ended_at`` are wall-clock (``time.time()``) stamps of
    the first attempt's start and the successful attempt's end.
    """

    permutation: list[int]
    text: str = ""
    repairs: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    started_at: float | None = None
    ended_at: float | None = None
    cached: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def repaired(self) -> bool:
        return bool(self.repairs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "permutation": list(self.permutation),
            "repairs": list(self.repairs),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_s": self.latency_s,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "cached": self.cached,
            "meta": dict(self.meta),
        }


@dataclass
class RerankResult:
    """The output of a schedule on one query.

    ``ranking`` is a permutation of the input positions, best first.
    ``rounds`` groups the calls by dependency: the calls of one round depend
    only on earlier rounds and are issued concurrently.
    """

    ranking: list[int]
    rounds: list[list[Call]]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def num_rounds(self) -> int:
        return len(self.rounds)

    @property
    def num_calls(self) -> int:
        return sum(len(r) for r in self.rounds)

    @property
    def round_widths(self) -> list[int]:
        return [len(r) for r in self.rounds]

    @property
    def max_width(self) -> int:
        return max(self.round_widths, default=0)

    @property
    def prompt_tokens(self) -> int:
        return sum(c.prompt_tokens for r in self.rounds for c in r)

    @property
    def completion_tokens(self) -> int:
        return sum(c.completion_tokens for r in self.rounds for c in r)

    @property
    def repair_rate(self) -> float:
        n = self.num_calls
        return sum(c.repaired for r in self.rounds for c in r) / n if n else 0.0

    def critical_path_s(self) -> float:
        """The sum over rounds of each round's slowest call.

        This is the latency of the schedule when the server has spare capacity.
        """
        return sum(max((c.latency_s for c in r), default=0.0) for r in self.rounds)

    def reorder(self, items: Sequence[Any]) -> list[Any]:
        """Return ``items`` (aligned with the input passages) in ranked order."""
        return [items[i] for i in self.ranking]


def complete_ranking(head: Sequence[int], n: int, rest_order: Sequence[int] | None = None) -> list[int]:
    """Extend a best-first ``head`` to a permutation of ``range(n)``.

    Positions missing from ``head`` follow in ``rest_order`` (input order by
    default).
    """
    seen = set(head)
    tail = rest_order if rest_order is not None else range(n)
    return list(head) + [i for i in tail if i not in seen]
