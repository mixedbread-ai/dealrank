"""BlitzRank as released by its authors, run in process as a dealrank schedule.

Table 1's BlitzRank rows run the released ``blitzrank`` package (version
0.1.0) end to end: its scheduling and stopping rule, prompt assembly, document
preparation, parser, retries and LiteLLM transport are the package's own. Two
things are held fixed so that every method sees the same model under the same
decoding regime:

* the candidate texts, cut to the ranker's ``max_passage_words`` before the
  package sees them, as every other schedule's prompt cuts them;
* the request parameters the package's public API does not expose (seed,
  output budget, reasoning flag and provider routing, endpoint), which are
  injected at the single place where the package calls ``litellm.acompletion``.
  The temperature goes through the package's own ``ComparerConfig``.

With ``concurrent_cover=True`` the package's first ``n // window`` calls, which
are the contiguous blocks of its input whatever the model answers, are issued
concurrently through the package's own comparer before its sort starts, and
the sort's first requests are answered from them. Every later call runs live,
in the order the package schedules it, so its decisions are unchanged; only
the opening calls overlap in time. ``input_order="dealt"`` presents the
candidates in DealRank's dealt order, so those opening calls are dealt hands
when the number of candidates is a multiple of ``window``, as for the pools of
100. On other pools (32 of the paper's 1,622 BM25 queries, none of the dense
ones) the package still cuts blocks of ``window``, and each holds one hand and
the start of the next.

The package is an optional dependency (``pip install blitzrank==0.1.0``) and
is imported on first use. Any :class:`~dealrank.ListwiseLLM` works as the
ranker: :class:`~dealrank.llm.OpenAICompatible` and
:class:`~dealrank.llm.LiteLLM` are reached through LiteLLM, as the package
does; any other ranker, such as :class:`~dealrank.FakeLLM` in a dry run,
answers the package's requests through :meth:`ListwiseLLM.rank`.
"""

from __future__ import annotations

import asyncio
import contextvars
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from dealrank.llm.base import ListwiseLLM
from dealrank.prompt import truncate_words
from dealrank.result import Call, RerankResult, complete_ranking
from dealrank.schedules.base import Schedule, input_permutation

PACKAGE_VERSION = "0.1.0"


@dataclass
class Route:
    """How the package's requests reach the model.

    ``model`` uses LiteLLM's ``provider/model`` naming. ``ranker`` is set
    instead of a LiteLLM endpoint when requests are answered by
    :meth:`ListwiseLLM.rank`.
    """

    model: str
    temperature: float | None = 0.0
    seed: int | None = 42
    max_tokens: int | None = 1024
    extra_body: dict[str, Any] = field(default_factory=dict)
    api_base: str | None = None
    api_key: str | None = None
    ranker: ListwiseLLM | None = None


def route_for(llm: ListwiseLLM) -> Route:
    """The LiteLLM route equivalent to ``llm``'s endpoint and decoding regime."""
    from dealrank.llm.litellm_backend import LiteLLM
    from dealrank.llm.openai_compatible import OPENROUTER_BASE_URL, OpenAICompatible

    if isinstance(llm, OpenAICompatible):
        common = {
            "temperature": llm.temperature,
            "seed": llm.seed,
            "max_tokens": llm.max_tokens,
            "extra_body": dict(llm.extra_body),
        }
        base = (llm.base_url or "").rstrip("/")
        if base == OPENROUTER_BASE_URL:
            # LiteLLM's openrouter provider reads OPENROUTER_API_KEY, as the client does.
            return Route(model=f"openrouter/{llm.model}", **common)
        # Any other OpenAI-compatible server (vLLM, ...) is LiteLLM's openai provider.
        api_key = getattr(llm, "api_key", None) or "EMPTY"
        return Route(model=f"openai/{llm.model}", api_base=base or None, api_key=api_key, **common)
    if isinstance(llm, LiteLLM):
        kwargs = dict(llm.completion_kwargs)
        return Route(
            model=llm.model,
            temperature=llm.temperature,
            seed=llm.seed,
            max_tokens=llm.max_tokens,
            extra_body=dict(kwargs.pop("extra_body", None) or {}),
            api_base=kwargs.pop("api_base", None),
            api_key=kwargs.pop("api_key", None),
        )
    # The openrouter/ prefix keeps the package from asking for a local server URL;
    # these requests never reach LiteLLM.
    return Route(model=f"openrouter/{llm.model}", ranker=llm)


@dataclass
class _Active:
    """The rerank in progress in the current task: its route, query and call trace."""

    route: Route
    query: str
    calls: list[dict[str, Any]]


_ACTIVE: contextvars.ContextVar[_Active | None] = contextvars.ContextVar("blitzrank_package_active", default=None)
# Per logical call: [start of the first attempt, failed attempts so far].
_ATTEMPTS: contextvars.ContextVar[list[Any] | None] = contextvars.ContextVar("blitzrank_package_attempts", default=None)
_INSTALLED = False


def _install() -> None:
    """Route the package's ``acompletion`` through :func:`_acompletion` (once per process)."""
    global _INSTALLED
    if _INSTALLED:
        return
    import blitzrank.engine.comparer.client as client

    client.acompletion = _acompletion
    _INSTALLED = True


async def _litellm_acompletion(**kwargs: Any) -> Any:
    import litellm

    return await litellm.acompletion(**kwargs)


async def _acompletion(
    *args: Any,
    api_key: str | None = None,
    api_base: str | None = None,
    extra_body: dict[str, Any] | None = None,
    timeout: float | None = None,
    **kwargs: Any,
) -> Any:
    """The package's LiteLLM call, with the decoding regime pinned and every call recorded."""
    active = _ACTIVE.get()
    if active is None:  # the package used outside a rerank of this module
        return await _litellm_acompletion(
            api_key=api_key, api_base=api_base, extra_body=extra_body, timeout=timeout, **kwargs
        )
    route = active.route
    body = dict(extra_body or {})
    body.update(route.extra_body)  # the pinned regime wins; the package's other keys stay
    if route.seed is not None:
        kwargs.setdefault("seed", route.seed)
    if route.max_tokens is not None:
        kwargs.setdefault("max_tokens", route.max_tokens)
    api_base = api_base or route.api_base
    api_key = api_key or route.api_key

    attempts = _ATTEMPTS.get()
    if attempts is None:
        attempts = [None, 0]
        _ATTEMPTS.set(attempts)
    started_at = time.time()
    if attempts[0] is None:
        attempts[0] = started_at
    t0 = time.monotonic()
    try:
        if route.ranker is not None:
            response = await _rank_with(route.ranker, active.query, kwargs["messages"])
        else:
            response = await _litellm_acompletion(
                *args, api_key=api_key, api_base=api_base, extra_body=body, timeout=timeout, **kwargs
            )
    except BaseException:
        attempts[1] += 1
        raise
    latency = time.monotonic() - t0
    first_start, failed = attempts
    attempts[0], attempts[1] = None, 0
    active.calls.append(_trace_row(response, kwargs, body, latency, first_start, time.time(), failed))
    return response


async def _rank_with(ranker: ListwiseLLM, query: str, messages: list[dict[str, str]]) -> Any:
    """Answer a package request with ``ranker.rank``, shaped like a LiteLLM response."""
    num = max(0, (len(messages) - 4) // 2)
    docs = []
    for i, turn in enumerate(messages[3 : 3 + 2 * num : 2], 1):
        content = turn["content"]
        prefix = f"[{i}] "
        docs.append(content[len(prefix) :] if content.startswith(prefix) else content)
    call = await ranker.rank(query, docs)
    usage = SimpleNamespace(
        prompt_tokens=call.prompt_tokens,
        completion_tokens=call.completion_tokens,
        completion_tokens_details=None,
        prompt_tokens_details=None,
    )
    choice = SimpleNamespace(message=SimpleNamespace(content=call.text), finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=usage, model=ranker.model, id=None)


def _trace_row(
    response: Any,
    kwargs: dict[str, Any],
    body: dict[str, Any],
    latency: float,
    started_at: float,
    ended_at: float,
    failed: int,
) -> dict[str, Any]:
    choices = getattr(response, "choices", None) or []
    choice = choices[0] if choices else None
    usage = getattr(response, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    hidden = getattr(response, "_hidden_params", None)
    return {
        "messages": kwargs.get("messages") or [],
        "text": (choice.message.content if choice is not None else None) or "",
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "latency_s": latency,
        "started_at": started_at,
        "ended_at": ended_at,
        "meta": {
            "requested_model": kwargs.get("model"),
            "response_model": getattr(response, "model", None),
            "provider": hidden.get("custom_llm_provider") if isinstance(hidden, dict) else None,
            "finish_reason": getattr(choice, "finish_reason", None) if choice is not None else None,
            "reasoning_tokens": getattr(details, "reasoning_tokens", None) if details is not None else None,
            "cached_tokens": getattr(prompt_details, "cached_tokens", None) if prompt_details is not None else None,
            "temperature": kwargs.get("temperature"),
            "seed": kwargs.get("seed"),
            "max_tokens": kwargs.get("max_tokens"),
            "extra_body": body,
            "failed_attempts": failed,
        },
    }


class BlitzRankPackage(Schedule):
    """The released BlitzRank package with ``window`` passages per call, certifying the top ``top_m``.

    Args:
        llm: The listwise ranker whose endpoint and decoding regime the package uses.
        window: Passages per call (BlitzRank's ``k``).
        top_m: The size of the certified top set (BlitzRank's ``m``).
        input_order: ``"retriever"`` (the package as released) or ``"dealt"``.
        concurrent_cover: Issue the opening ``n // window`` calls as one concurrent round.

    The ranking is the package's certified top ``top_m``, then the other
    candidates in the order the package was given them. Each call is its own
    round, except the opening calls under ``concurrent_cover``, which form the
    first round.
    """

    def __init__(
        self,
        llm: ListwiseLLM | None = None,
        *,
        window: int = 20,
        top_m: int = 10,
        input_order: str = "retriever",
        concurrent_cover: bool = False,
    ) -> None:
        super().__init__(llm)
        if window < 2:
            raise ValueError("window must be at least 2")
        if top_m < 1:
            raise ValueError("top_m must be positive")
        if input_order not in ("retriever", "dealt"):
            raise ValueError(f"input_order must be 'retriever' or 'dealt', not {input_order!r}")
        self.window = window
        self.top_m = top_m
        self.input_order = input_order
        self.concurrent_cover = concurrent_cover
        self.name = (
            f"blitzrank-package-w{window}-m{top_m}"
            + ("-dealt" if input_order == "dealt" else "")
            + ("-concurrent" if concurrent_cover else "")
        )

    async def _rerank(self, query: str, passages: list[str], llm: ListwiseLLM) -> RerankResult:
        n = len(passages)
        perm = input_permutation(self.input_order, n, window=self.window)
        diagnostics: dict[str, Any] = {"input_order": {"order": self.input_order, "permutation": perm}}
        if n == 0:
            return RerankResult(
                ranking=[], rounds=[], diagnostics={"stop": "certified", "certified": True, **diagnostics}
            )

        docs = [truncate_words(passages[i], llm.max_passage_words) for i in perm]
        active = _Active(route=route_for(llm), query=query, calls=[])
        token = _ACTIVE.set(active)
        attempts_token = _ATTEMPTS.set([None, 0])
        try:
            top, matches, cover, num_calls = await self._run(query, docs, active.route)
        finally:
            _ATTEMPTS.reset(attempts_token)
            _ACTIVE.reset(token)
        if num_calls != len(active.calls):
            raise RuntimeError(f"{self.name}: the package made {num_calls} calls but {len(active.calls)} were traced")

        calls = [_to_call(row) for row in active.calls]
        rounds = [[c] for c in calls]
        if cover >= 2:
            rounds = [calls[:cover], *rounds[cover:]]
        ranking = [perm[i] for i in complete_ranking(top, n)]
        diagnostics.update(
            {
                "stop": "certified",
                "certified": True,
                "package": f"blitzrank=={PACKAGE_VERSION}",
                "calls": [[perm[i] for i in match] for match in matches],
                "cover_calls": cover,
            }
        )
        return RerankResult(ranking=ranking, rounds=rounds, diagnostics=diagnostics)

    async def _run(self, query: str, docs: list[str], route: Route) -> tuple[list[int], list[list[int]], int, int]:
        """Run the package; returns (top positions, positions of each call, concurrent calls, calls)."""
        from blitzrank.engine.algorithms.tournament_graph.experimental_interface import (
            LLMCompareOracle,
            RerankTask,
            SingleContent,
            TournamentGraphRerankConfig,
            rerank_with_tournament_graph,
        )
        from blitzrank.engine.algorithms.tournament_graph.types import TournamentGraphSortConfig
        from blitzrank.engine.config import ComparerConfig, ComparerType

        _install()
        n = len(docs)
        # The construction of blitzrank.rankers.BlitzRank.__call__, plus the
        # temperature and a round log, which its public API does not take.
        oracle = LLMCompareOracle(
            min(self.window, n),
            ComparerConfig(type=ComparerType.LISTWISE_RANK_GPT, model=route.model, temperature=route.temperature),
        )
        contents = [
            SingleContent(content=d, qid="0", docid=str(i), rank=i + 1, score=n - i, orig_idx=i)
            for i, d in enumerate(docs)
        ]
        hits = [{"content": d, "qid": "0", "docid": str(i), "rank": i + 1, "score": n - i} for i, d in enumerate(docs)]
        task = RerankTask(query=query, contents=contents, hits=hits)
        matches: list[list[int]] = []

        def on_round(log: dict[str, Any]) -> None:
            matches.append([int(i) for i in log["nodes_for_match"]])

        cover = await _prefetch_cover(oracle, task) if self.concurrent_cover else 0
        result = await rerank_with_tournament_graph(
            task,
            TournamentGraphRerankConfig(top_m=min(self.top_m, n), oracle=oracle),
            TournamentGraphSortConfig(on_round_complete=on_round),
        )
        top = [item.content.orig_idx for item in result.results]
        return top, matches, cover, result.num_oracle_calls


async def _prefetch_cover(oracle: Any, task: Any) -> int:
    """Issue the package's opening calls concurrently; returns how many were issued.

    While some candidate is untouched, the package's next call is the next
    ``k`` untouched candidates in input order, so its first ``n // k`` calls
    are the contiguous blocks of its input. They are issued at once through
    the oracle's own comparer, and the oracle then answers the sort's first
    requests from them, refusing any request that is not the next block.
    """
    from blitzrank.engine.algorithms.tournament_graph.compare_oracle import OracleResult
    from blitzrank.engine.algorithms.tournament_graph.experimental_interface import ContentItem

    k = oracle.get_k()
    n = len(task.contents)
    cover = n // k if n > k else 0
    if cover < 2:
        return 0
    active = _ACTIVE.get()
    assert active is not None
    blocks = [[ContentItem(c, task.query) for c in task.contents[j * k : (j + 1) * k]] for j in range(cover)]
    live: Callable[[list[Any]], Awaitable[Any]] = oracle._compare_k_async
    logs_before = len(oracle.call_logs)

    async def one(block: list[Any]) -> tuple[Any, list[dict[str, Any]]]:
        # Each block runs in its own task: give it its own trace and attempt state.
        mine = _Active(route=active.route, query=active.query, calls=[])
        _ACTIVE.set(mine)
        _ATTEMPTS.set([None, 0])
        return await live(block), mine.calls

    outs = await asyncio.gather(*(one(b) for b in blocks))
    for _, rows in outs:  # block order, not completion order
        active.calls.extend(rows)
    oracle.call_logs[logs_before:] = [res.metadata for res, _ in outs]
    pending = deque((tuple(str(it.id) for it in b), res) for b, (res, _) in zip(blocks, outs))

    async def compare(items: list[Any]) -> Any:
        if not pending:
            return await live(items)
        ids, res = pending.popleft()
        if tuple(str(it.id) for it in items) != ids:
            raise RuntimeError("the package asked for a call other than its next opening block")
        # The package's graph keys nodes by object: restate the edges on the sort's own items.
        own = {str(it.id): it for it in items}
        return OracleResult(edges=[(own[str(u.id)], own[str(v.id)]) for u, v in res.edges], metadata=res.metadata)

    oracle._compare_k_async = compare
    return cover


def _to_call(row: dict[str, Any]) -> Call:
    """A traced package call as a dealrank :class:`~dealrank.result.Call`, parsed by the package's parser."""
    from blitzrank.engine.comparer.listwise_rank_gpt import parse_permutation

    num = max(0, (len(row["messages"]) - 4) // 2)
    permutation, missing, duplicates = parse_permutation(row["text"], num)
    repairs = []
    parsed_any = len(missing) < num
    if not parsed_any:
        repairs.append("empty")
    if duplicates:
        repairs.append("dedupe")
    if missing and parsed_any:
        repairs.append("fill_missing")
    return Call(
        permutation=list(permutation),
        text=row["text"],
        repairs=repairs,
        prompt_tokens=int(row["prompt_tokens"]),
        completion_tokens=int(row["completion_tokens"]),
        latency_s=float(row["latency_s"]),
        started_at=row["started_at"],
        ended_at=row["ended_at"],
        meta=row["meta"],
    )


__all__ = ["PACKAGE_VERSION", "BlitzRankPackage", "Route", "route_for"]
