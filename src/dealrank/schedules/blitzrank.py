"""BlitzRank (Agrawal et al., 2026): tournament-graph reranking with certification.

Every call on a set ``Q`` of at most ``window`` passages reveals the full
order of ``Q``, that is all ``|Q|(|Q|-1)/2`` pairwise outcomes. The outcomes
accumulate in a directed graph, and ``u`` is known to beat ``v`` when ``u``
reaches ``v``. A passage is *resolved* when its relation to every other
passage is known. The loop stops once the current top ``m`` (the passages with
the fewest known winners over them) are all resolved, which certifies them.

The next call takes one representative from each strongly connected component
that still holds an unresolved passage, ordered by (in-reach, out-reach,
smallest member), and the representative of a component is its member with the
least resolved relations, then the smallest position. These are the
tie-breaks of the released implementation.

While any passage is untouched, untouched passages sort first, so the first
``n // window`` calls are the contiguous blocks ``[0, window)``,
``[window, 2 * window)``, ... whatever the model answers. They depend on
nothing, and ``concurrent_cover=True`` issues them as one round. With
``input_order="dealt"`` and ``n`` a multiple of ``window``, as with 100
candidates and a window of 20, those blocks are DealRank's hands.

When ``n`` is not a multiple of ``window``, the two differ. The deal makes
``ceil(n / window)`` equal hands of fewer than ``window`` passages, and the
opening blocks are still cut every ``window`` positions of the dealt order, so
a block holds one hand and the start of the next. With 95 candidates the first
block is hand 0 (19 passages) and the first passage of hand 1; with 85, the
four full blocks hold 7, 5, 4 and 4 of the retriever's top twenty where each
hand holds 4. The released package cuts its opening blocks the same way. In the
paper's BM25 pools this concerns 32 of 1,622 queries (31 of NFCorpus, one of
SciFact): pools of 21 to 99 candidates other than 40, 60 or 80. Their nDCG@10
is part of the "dealt input" rows; the cost columns use full pools only.
"""

from __future__ import annotations

from dealrank.llm.base import ListwiseLLM
from dealrank.result import Call, RerankResult, complete_ranking
from dealrank.schedules.base import INPUT_ORDERS, InputOrder, Schedule, input_permutation


def _bits(mask: int) -> list[int]:
    """Indices of the set bits of ``mask``, ascending."""
    out = []
    while mask:
        low = mask & -mask
        out.append(low.bit_length() - 1)
        mask ^= low
    return out


def strongly_connected_components(adj: list[int], n: int) -> tuple[list[int], list[int]]:
    """Iterative Tarjan on a bitmask adjacency list.

    Returns ``(component_of, component_masks)``, with components in reverse
    topological order of the condensation (successors first).
    """
    succ = [_bits(a) for a in adj]
    index = [-1] * n
    low = [0] * n
    on_stack = [False] * n
    stack: list[int] = []
    comp_of = [-1] * n
    comp_masks: list[int] = []
    counter = 0

    for root in range(n):
        if index[root] != -1:
            continue
        frames = [(root, 0)]
        while frames:
            v, pi = frames.pop()
            if pi == 0:
                index[v] = low[v] = counter
                counter += 1
                stack.append(v)
                on_stack[v] = True
            else:  # back from the child succ[v][pi - 1]
                low[v] = min(low[v], low[succ[v][pi - 1]])
            descended = False
            while pi < len(succ[v]):
                w = succ[v][pi]
                pi += 1
                if index[w] == -1:
                    frames.append((v, pi))
                    frames.append((w, 0))
                    descended = True
                    break
                if on_stack[w]:
                    low[v] = min(low[v], index[w])
            if descended:
                continue
            if low[v] == index[v]:
                mask = 0
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    comp_of[w] = len(comp_masks)
                    mask |= 1 << w
                    if w == v:
                        break
                comp_masks.append(mask)
    return comp_of, comp_masks


def reachability(adj: list[int], n: int, comp_of: list[int], comp_masks: list[int]) -> tuple[list[int], list[int]]:
    """Per-vertex ``(in_reach, out_reach)`` bitmasks, via closure on the condensation."""
    comp_out = [0] * len(comp_masks)
    for c, members in enumerate(comp_masks):
        ext = 0
        for u in _bits(members):
            ext |= adj[u]
        ext &= ~members
        reach = members
        seen = 0
        for w in _bits(ext):
            d = comp_of[w]
            if not (seen >> d) & 1:
                seen |= 1 << d
                reach |= comp_out[d]
        comp_out[c] = reach

    out_reach = [comp_out[comp_of[v]] & ~(1 << v) for v in range(n)]
    in_reach = [0] * n
    for u in range(n):
        for v in _bits(out_reach[u]):
            in_reach[v] |= 1 << u
    return in_reach, out_reach


class BlitzRank(Schedule):
    """BlitzRank with ``window`` passages per call, certifying the top ``top_m``.

    Args:
        llm: The listwise ranker.
        window: Passages per call (BlitzRank's ``k``).
        top_m: The size of the certified top set (BlitzRank's ``m``).
        input_order: ``"dealt"`` (the default here) presents the candidates
            in DealRank's dealt order, so the opening calls are dealt hands
            when ``n`` is a multiple of ``window`` (see the module docstring
            for other sizes); ``"retriever"`` is BlitzRank as released, whose
            opening calls are contiguous chunks.
        concurrent_cover: Issue the opening ``n // window`` calls, which depend
            on nothing, as one concurrent round.
        max_calls: Raise if the top set is not certified within this many
            calls, rather than return an uncertified ranking.
    """

    def __init__(
        self,
        llm: ListwiseLLM | None = None,
        *,
        window: int = 20,
        top_m: int = 10,
        input_order: InputOrder | str = "dealt",
        concurrent_cover: bool = False,
        max_calls: int = 400,
    ) -> None:
        super().__init__(llm)
        if window < 2:
            raise ValueError("window must be at least 2")
        if top_m < 1:
            raise ValueError("top_m must be positive")
        if input_order not in INPUT_ORDERS:
            raise ValueError(f"unknown input order {input_order!r}; expected one of {INPUT_ORDERS}")
        self.window = window
        self.top_m = top_m
        self.input_order = input_order
        self.concurrent_cover = concurrent_cover
        self.max_calls = max_calls
        self.name = (
            f"blitzrank-w{window}-m{top_m}"
            + ("-dealt" if input_order == "dealt" else "" if input_order == "retriever" else f"-{input_order}")
            + ("-concurrent" if concurrent_cover else "")
        )

    async def _rerank(self, query: str, passages: list[str], llm: ListwiseLLM) -> RerankResult:
        perm = input_permutation(self.input_order, len(passages), window=self.window)
        result = await self._tournament(query, [passages[i] for i in perm], llm)
        result.diagnostics["input_order"] = {"order": self.input_order, "permutation": perm}
        result.ranking = [perm[i] for i in result.ranking]
        return result

    def _next_call(
        self, n: int, comp_masks: list[int], in_reach: list[int], out_reach: list[int], kappa: list[int]
    ) -> tuple[int, ...]:
        candidates = []
        for members in comp_masks:
            ids = _bits(members)
            if all(kappa[v] == n - 1 for v in ids):
                continue
            v0 = ids[0]  # every member shares the component's outside reach
            cin = (in_reach[v0] & ~members).bit_count()
            cout = (out_reach[v0] & ~members).bit_count()
            rep = min(ids, key=lambda v: (kappa[v], v))
            candidates.append((cin, cout, v0, rep))
        candidates.sort()
        return tuple(rep for *_, rep in candidates[: self.window])

    async def _tournament(self, query: str, passages: list[str], llm: ListwiseLLM) -> RerankResult:
        """BlitzRank on ``passages`` in the order given."""
        n = len(passages)
        adj = [0] * n  # adj[u]: bitmask of the v that u is known to beat directly
        rounds: list[list[Call]] = []
        previous: tuple[int, ...] | None = None

        def record(q: tuple[int, ...], call: Call) -> None:
            ranked = [q[i] for i in call.permutation]
            for i, u in enumerate(ranked):
                for v in ranked[i + 1 :]:
                    adj[u] |= 1 << v

        cover = n // self.window if (self.concurrent_cover and n > self.window) else 0
        if cover >= 2:
            blocks = [tuple(range(j * self.window, (j + 1) * self.window)) for j in range(cover)]
            calls = await llm.rank_many((query, [passages[i] for i in q]) for q in blocks)
            rounds.append(calls)
            for q, call in zip(blocks, calls):
                record(q, call)
            previous = blocks[-1]

        while True:
            comp_of, comp_masks = strongly_connected_components(adj, n)
            in_reach, out_reach = reachability(adj, n, comp_of, comp_masks)
            kappa = [(in_reach[v] | out_reach[v]).bit_count() for v in range(n)]
            in_count = [m.bit_count() for m in in_reach]
            order = sorted(range(n), key=lambda v: (in_count[v], v))
            top = order[: self.top_m]
            if all(kappa[v] == n - 1 for v in top):
                stop = "certified"
                break
            if sum(len(r) for r in rounds) >= self.max_calls:
                raise RuntimeError(
                    f"{self.name}: no certificate for the top {self.top_m} of {n} within "
                    f"{self.max_calls} calls; raise max_calls"
                )
            q = self._next_call(n, comp_masks, in_reach, out_reach, kappa)
            if q == previous:  # the same call again would reveal nothing new
                stop = "repeated_call"
                break
            previous = q
            call = await llm.rank(query, [passages[i] for i in q])
            rounds.append([call])
            record(q, call)

        return RerankResult(
            ranking=complete_ranking(top, n, rest_order=order),
            rounds=rounds,
            diagnostics={"stop": stop, "certified": stop == "certified"},
        )
