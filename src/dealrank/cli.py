"""Command-line interface: ``dealrank rerank``, ``dealrank deal`` and ``dealrank serve-vllm``."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO

from dealrank.llm.base import ListwiseLLM
from dealrank.partition import SPLITS, num_hands, split
from dealrank.schedules import BlitzRank, DealRank, HierarchicalDealRank, Schedule


def _read_queries(handle: TextIO) -> list[dict[str, Any]]:
    """Read JSONL queries with ``query`` and either ``candidates`` or ``passages``."""
    queries = []
    for line_no, line in enumerate(handle, 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if "candidates" in record:
            ids = [c.get("docid", str(i)) for i, c in enumerate(record["candidates"])]
            texts = [c["text"] for c in record["candidates"]]
        elif "passages" in record:
            texts = list(record["passages"])
            ids = list(range(len(texts)))
        else:
            raise SystemExit(f"line {line_no}: expected a 'candidates' or 'passages' field")
        queries.append({"qid": record.get("qid", line_no - 1), "query": record["query"], "ids": ids, "texts": texts})
    return queries


def _build_llm(args: argparse.Namespace) -> ListwiseLLM:
    if args.fake:
        from dealrank.llm.fake import FakeLLM

        return FakeLLM()
    if not args.model:
        raise SystemExit("--model is required (or --fake for an offline dry run)")
    from dealrank.llm.openai_compatible import OpenAICompatible

    common = {"max_concurrency": args.max_concurrency, "max_tokens": args.max_tokens}
    if args.openrouter:
        return OpenAICompatible.openrouter(args.model, **common)
    if args.vllm:
        return OpenAICompatible.vllm(args.model, base_url=args.base_url or "http://127.0.0.1:8000/v1", **common)
    # A key is sent to a custom --base-url only when --api-key-env names it, so that an
    # OpenAI key in the environment never reaches another server. Without --base-url,
    # the client reads OPENAI_API_KEY itself.
    api_key = None
    if args.api_key_env:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise SystemExit(f"--api-key-env: the environment variable {args.api_key_env} is not set")
    return OpenAICompatible(args.model, base_url=args.base_url, api_key=api_key, **common)


def _build_schedule(args: argparse.Namespace) -> Schedule:
    if args.schedule == "dealrank":
        return DealRank(window=args.window, quota=args.quota)
    if args.schedule == "hierarchical":
        return HierarchicalDealRank(window=args.window, first_quota=args.first_quota)
    return BlitzRank(window=args.window, top_m=args.top_m, concurrent_cover=args.concurrent_cover)


async def _rerank_all(
    schedule: Schedule, llm: ListwiseLLM, queries: list[dict[str, Any]], out: TextIO, concurrency: int
) -> None:
    sem = asyncio.Semaphore(concurrency)

    async def one(q: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            result = await schedule.arerank(q["query"], q["texts"], llm=llm)
        return {
            "qid": q["qid"],
            "ranking": result.reorder(q["ids"]),
            "rounds": result.num_rounds,
            "calls": result.num_calls,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "critical_path_s": round(result.critical_path_s(), 3),
        }

    for record in await asyncio.gather(*(one(q) for q in queries)):
        out.write(json.dumps(record) + "\n")


def _cmd_rerank(args: argparse.Namespace) -> int:
    with open(args.input) if args.input != "-" else sys.stdin as handle:
        queries = _read_queries(handle)
    llm = _build_llm(args)
    schedule = _build_schedule(args)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") if args.output else sys.stdout as out:
        asyncio.run(_rerank_all(schedule, llm, queries, out, args.query_concurrency))
    return 0


def _cmd_deal(args: argparse.Namespace) -> int:
    hands = args.hands if args.hands else num_hands(args.n, args.window)
    for h, group in enumerate(split(args.split, args.n, hands, seed=args.seed), 1):
        print(f"hand {h:>3}: " + " ".join(str(i + 1) for i in group))
    return 0


def _cmd_serve_vllm(args: argparse.Namespace) -> int:
    from dealrank.serve import vllm_command

    cmd = vllm_command(
        args.model,
        port=args.port,
        tensor_parallel_size=args.tensor_parallel_size,
        revision=args.revision,
        max_model_len=args.max_model_len,
    )
    if args.dry_run:
        print(shlex.join(cmd))
        return 0
    os.execvp(cmd[0], cmd)
    return 1  # pragma: no cover


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dealrank", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    rerank = sub.add_parser("rerank", help="rerank JSONL queries with an LLM")
    rerank.add_argument("input", help="JSONL with 'query' and 'candidates' [{docid, text}] or 'passages'; - for stdin")
    rerank.add_argument("-o", "--output", help="output JSONL (default: stdout)")
    rerank.add_argument("--schedule", choices=["dealrank", "hierarchical", "blitzrank"], default="dealrank")
    rerank.add_argument("--window", type=int, default=20, help="passages per call (default 20)")
    rerank.add_argument("--quota", type=int, default=None, help="DealRank: finalists per hand (default window/hands)")
    rerank.add_argument("--first-quota", type=int, default=2, help="hierarchical: survivors per first-round hand")
    rerank.add_argument("--top-m", type=int, default=10, help="BlitzRank: size of the certified top set")
    rerank.add_argument("--concurrent-cover", action="store_true", help="BlitzRank: run the opening calls at once")
    backend = rerank.add_mutually_exclusive_group()
    backend.add_argument("--openrouter", action="store_true", help="call OpenRouter (key in OPENROUTER_API_KEY)")
    backend.add_argument("--vllm", action="store_true", help="call a local vLLM server, thinking disabled")
    backend.add_argument("--fake", action="store_true", help="offline ranker that keeps the input order")
    rerank.add_argument("--model", help="model name as the endpoint expects it")
    rerank.add_argument("--base-url", help="OpenAI-compatible API root, e.g. http://127.0.0.1:8000/v1")
    rerank.add_argument(
        "--api-key-env",
        default=None,
        help="environment variable holding the key (default: OPENAI_API_KEY for OpenAI's API, no key for --base-url)",
    )
    rerank.add_argument("--max-tokens", type=int, default=1024)
    rerank.add_argument("--max-concurrency", type=int, default=32, help="most LLM requests in flight")
    rerank.add_argument("--query-concurrency", type=int, default=8, help="most queries in flight")
    rerank.set_defaults(func=_cmd_rerank)

    dealp = sub.add_parser("deal", help="print a split of N candidates (1-based retriever ranks)")
    dealp.add_argument("n", type=int)
    dealp.add_argument("--window", type=int, default=20)
    dealp.add_argument("--hands", type=int, default=None, help="default: ceil(n / window)")
    dealp.add_argument("--split", choices=SPLITS, default="deal")
    dealp.add_argument("--seed", type=int, default=0)
    dealp.set_defaults(func=_cmd_deal)

    serve = sub.add_parser("serve-vllm", help="start vllm serve with the paper's settings")
    serve.add_argument("model")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--tensor-parallel-size", type=int, default=2)
    serve.add_argument("--max-model-len", type=int, default=65536)
    serve.add_argument("--revision", default=None)
    serve.add_argument("--dry-run", action="store_true", help="print the command instead of running it")
    serve.set_defaults(func=_cmd_serve_vllm)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
