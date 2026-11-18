"""Download and normalise the first-stage pools and qrels.

    python -m repro.data bm25            # BM25 top 100, ten benchmarks, and their qrels (~450 MB)
    python -m repro.data deep            # BM25 top 1,000 of DL19, DL20, DBpedia, SciFact
    python -m repro.data dense --help    # Qwen3-Embedding-4B top 100 (GPU; see repro.dense)
    python -m repro.data check           # what is present, with query counts

Every pool is written as ``data/pools/<retriever>/<benchmark>.jsonl`` with one
query per line, ``{"qid", "query", "candidates": [{"docid", "score", "text"}]}``
in retriever order, and qrels as ``data/qrels/<benchmark>.txt`` (TREC format),
shared by every retriever (see :mod:`repro.records`).

Sources:

* BM25 top 100 and DL19/DL20 top 1,000: the files of ``castorini/rank_llm_data``
  on the Hugging Face hub, which inline the passage texts. Some BEIR pools
  hold fewer than 100 candidates for a few queries; they are kept as they are.
* qrels: ``castorini/eval`` (formerly ``castorini/anserini-tools``), at a pinned revision.
* BM25 top 1,000 of DBpedia and SciFact: no such file is published, so they
  are retrieved with Pyserini from the prebuilt BEIR flat indexes with
  Anserini's BM25 defaults (k1 = 0.9, b = 0.4), for the queries of the top-100
  pools. This step needs ``pip install pyserini`` and a Java 21 runtime, and
  downloads the two indexes (about 3 GB for DBpedia).

Downloads are kept in ``data/raw/`` and skipped when present.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

from repro import records
from repro.records import BENCHMARKS, DEEP_BENCHMARKS, DENSE_BENCHMARKS

RANK_LLM = (
    "https://huggingface.co/datasets/castorini/rank_llm_data/resolve/main/"
    "retrieve_results/BM25/retrieve_results_{name}_top{k}.jsonl"
)
#: castorini/eval (formerly castorini/anserini-tools), pinned to a revision whose qrels paths are stable.
QRELS_REVISION = "04cd1df032ad078a0a7cb2a9aa84eb743c2eaa25"
ANSERINI_QRELS = f"https://raw.githubusercontent.com/castorini/eval/{QRELS_REVISION}/topics-and-qrels/qrels.{{key}}.txt"

#: benchmark -> (rank_llm_data name, anserini qrels key, number of queries)
RANK_LLM_SETS: dict[str, tuple[str, str, int]] = {
    "dl19": ("dl19", "dl19-passage", 43),
    "dl20": ("dl20", "dl20-passage", 54),
    "covid": ("covid", "beir-v1.0.0-trec-covid.test", 50),
    "nfcorpus": ("nfc", "beir-v1.0.0-nfcorpus.test", 323),
    "signal": ("signal", "beir-v1.0.0-signal1m.test", 97),
    "news": ("news", "beir-v1.0.0-trec-news.test", 57),
    "robust04": ("robust04", "beir-v1.0.0-robust04.test", 249),
    "touche": ("touche", "beir-v1.0.0-webis-touche2020.test", 49),
    "dbpedia": ("dbpedia", "beir-v1.0.0-dbpedia-entity.test", 400),
    "scifact": ("scifact", "beir-v1.0.0-scifact.test", 300),
}

#: Number of queries of every benchmark (1,622 in all).
QUERY_COUNTS: dict[str, int] = {b: n for b, (_, _, n) in RANK_LLM_SETS.items()}

#: Benchmarks whose top 1,000 is retrieved with Pyserini, and their prebuilt indexes.
PYSERINI_DEEP: dict[str, str] = {
    "dbpedia": "beir-v1.0.0-dbpedia-entity.flat",
    "scifact": "beir-v1.0.0-scifact.flat",
}
BM25_K1, BM25_B = 0.9, 0.4


def raw_dir() -> Path:
    return records.DATA / "raw"


def fetch(url: str, dest: Path, attempts: int = 5) -> Path:
    """Download ``url`` to ``dest`` unless it is already there."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    print(f"  GET {url}", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(attempts):
        try:
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(dest)
            return dest
        except OSError as exc:
            if attempt == attempts - 1:
                raise
            print(f"  retrying after {type(exc).__name__}: {exc}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise AssertionError("unreachable")


def doc_text(doc: dict[str, Any]) -> str:
    """The passage text of a rank_llm_data candidate, whichever of its three shapes it has.

    >>> doc_text({"contents": " a b "})
    'a b'
    >>> doc_text({"title": "T", "text": "body"})
    'T body'
    >>> doc_text({"passage": "body", "title": ""})
    'body'
    """
    if "contents" in doc:
        return doc["contents"].strip()
    text = (doc["passage"] if "passage" in doc else doc.get("text") or "").strip()
    title = (doc.get("title") or "").strip()
    return f"{title} {text}".strip() if title else text


def normalise_rank_llm(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Pool records from the lines of a rank_llm_data retrieval file."""
    for line in lines:
        if not line.strip():
            continue
        rec = json.loads(line)
        yield {
            "qid": str(rec["query"]["qid"]),
            "query": rec["query"]["text"],
            "candidates": [
                {"docid": str(c["docid"]), "score": float(c.get("score", 0.0)), "text": doc_text(c["doc"])}
                for c in rec["candidates"]
            ],
        }


def write_pool(path: Path, pool: Iterable[dict[str, Any]], expected: int | None = None) -> int:
    """Write pool records atomically; checks the query count and that no qid repeats."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.part")
    seen: set[str] = set()
    with open(tmp, "w") as out:
        for rec in pool:
            if rec["qid"] in seen:
                raise ValueError(f"{path.name}: qid {rec['qid']} appears twice")
            seen.add(rec["qid"])
            out.write(json.dumps(rec) + "\n")
    if expected is not None and len(seen) != expected:
        tmp.unlink()
        raise ValueError(f"{path.name}: {len(seen)} queries, expected {expected}")
    tmp.rename(path)
    return len(seen)


def ingest_bm25(benchmarks: Sequence[str]) -> None:
    """BM25 top-100 pools and qrels."""
    for benchmark in benchmarks:
        print(f"[{benchmark}]", flush=True)
        pool_path = records.pool_path("bm25", benchmark)
        name, qrels_key, expected = RANK_LLM_SETS[benchmark]
        raw = fetch(RANK_LLM.format(name=name, k=100), raw_dir() / f"{name}_top100.jsonl")
        with open(raw) as handle:
            n = write_pool(pool_path, normalise_rank_llm(handle), expected)
        fetch(ANSERINI_QRELS.format(key=qrels_key), records.qrels_path(benchmark))
        print(f"  -> {pool_path.relative_to(records.DATA)}: {n} queries", flush=True)


def ingest_deep(benchmarks: Sequence[str]) -> None:
    """BM25 top-1,000 pools (``bm25-top1000``)."""
    for benchmark in benchmarks:
        print(f"[{benchmark} top 1000]", flush=True)
        path = records.pool_path("bm25-top1000", benchmark)
        if benchmark in PYSERINI_DEEP:
            n = write_pool(path, pyserini_bm25(benchmark, 1000), QUERY_COUNTS[benchmark])
        else:
            name, _, expected = RANK_LLM_SETS[benchmark]
            raw = fetch(RANK_LLM.format(name=name, k=1000), raw_dir() / f"{name}_top1000.jsonl")
            with open(raw) as handle:
                n = write_pool(path, normalise_rank_llm(handle), expected)
        print(f"  -> {path.relative_to(records.DATA)}: {n} queries", flush=True)


def pyserini_bm25(benchmark: str, k: int) -> Iterator[dict[str, Any]]:
    """BM25 top ``k`` from a prebuilt BEIR flat index, for the queries of the top-100 pool.

    Passage text is ``"{title} {text}"`` (or the text alone without a title),
    as in the BEIR top-100 pools.
    """
    from pyserini.search.lucene import LuceneSearcher

    searcher = LuceneSearcher.from_prebuilt_index(PYSERINI_DEEP[benchmark])
    searcher.set_bm25(BM25_K1, BM25_B)
    for query in records.load_pool("bm25", benchmark):
        candidates = []
        for hit in searcher.search(query["query"], k=k):
            raw = json.loads(searcher.doc(hit.docid).raw())
            title, text = (raw.get("title") or "").strip(), (raw.get("text") or "").strip()
            candidates.append(
                {"docid": hit.docid, "score": float(hit.score), "text": f"{title} {text}".strip() if title else text}
            )
        yield {"qid": query["qid"], "query": query["query"], "candidates": candidates}


def check() -> int:
    """Print every expected pool and qrels file with its query count; 1 if one is missing or wrong."""
    expected = {"bm25": BENCHMARKS, "dense": DENSE_BENCHMARKS, "bm25-top1000": DEEP_BENCHMARKS}
    status = 0
    for retriever, benchmarks in expected.items():
        for benchmark in benchmarks:
            path = records.pool_path(retriever, benchmark)
            if not path.exists():
                print(f"  missing  {path.relative_to(records.DATA)}")
                status = 1
                continue
            with open(path) as handle:
                n = sum(1 for line in handle if line.strip())
            ok = n == QUERY_COUNTS[benchmark]
            status |= not ok
            print(f"  {'ok' if ok else 'WRONG':7}  {path.relative_to(records.DATA)}: {n} queries")
    for benchmark in BENCHMARKS:
        path = records.qrels_path(benchmark)
        if not path.exists():
            print(f"  missing  {path.relative_to(records.DATA)}")
            status = 1
    return status


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m repro.data", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    bm25 = sub.add_parser("bm25", help="BM25 top-100 pools and qrels of the ten benchmarks")
    bm25.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS), choices=BENCHMARKS)
    deep = sub.add_parser("deep", help="BM25 top-1,000 pools (DBpedia and SciFact need pyserini and Java 21)")
    deep.add_argument("--benchmarks", nargs="+", default=list(DEEP_BENCHMARKS), choices=DEEP_BENCHMARKS)
    sub.add_parser("dense", help="dense top-100 pools (GPU); run python -m repro.dense --help", add_help=False)
    sub.add_parser("check", help="list the pools and qrels present")
    args, rest = parser.parse_known_args(argv)

    if args.command == "bm25":
        ingest_bm25(args.benchmarks)
    elif args.command == "deep":
        missing = [b for b in args.benchmarks if b in PYSERINI_DEEP and not records.pool_path("bm25", b).exists()]
        if missing:
            raise SystemExit(f"run `python -m repro.data bm25` first: the top-1,000 queries come from {missing}")
        ingest_deep(args.benchmarks)
    elif args.command == "dense":
        from repro import dense

        return dense.main(rest)
    elif args.command == "check":
        return check()
    if rest:
        parser.error(f"unrecognised arguments: {' '.join(rest)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
