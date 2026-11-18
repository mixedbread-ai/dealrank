"""Build the dense pools: Qwen3-Embedding-4B, exact top 100 over each full corpus.

This step is optional and heavy. It needs a GPU, ``transformers`` and
``torch`` (and ``vllm`` for the default, faster backend), and the corpora:

=====================  ===================  ==========  ===================
corpus                 benchmarks           passages    embeddings (fp16)
=====================  ===================  ==========  ===================
msmarco-passage        dl19, dl20            8,841,823   45 GB
beir/trec-covid        covid                   171,332   0.9 GB
beir/nfcorpus          nfcorpus                  3,633   19 MB
beir/scifact           scifact                   5,183   27 MB
beir/webis-touche2020  touche                  382,545   2.0 GB
beir/dbpedia-entity    dbpedia               4,635,922   24 GB
beir/robust04          robust04                528,036   2.7 GB
=====================  ===================  ==========  ===================

On one H100 the vLLM backend embeds about 700 passages per second, so all
corpora take about 7 GPU hours; ``embed`` takes ``--shard`` / ``--num-shards``
to spread a corpus over several GPUs. MS MARCO is read from the
``collection.tsv`` that ``ir_datasets`` downloads (``msmarco-passage``), the
BEIR corpora through ``ir_datasets``, and Robust04, which neither serves, from
Pyserini's prebuilt ``beir-v1.0.0-robust04.flat`` index (``extract-robust04``,
needs ``pyserini`` and Java 21).

Passage texts are composed exactly as in the BM25 pools, so a passage reads
the same whichever pool it comes from. Queries carry the model card's
instruction, ``"Instruct: {task}\\nQuery:{query}"``, and passages none;
vectors are last-token pooled and L2-normalised, and the search is an exact
inner-product top 100 (no approximate index). Queries come from the BM25 pool
of each benchmark, so the dense pool has the same queries in the same order.

    python -m repro.dense all --corpus beir/scifact           # every step on one GPU
    python -m repro.dense docids --corpus msmarco-passage     # fix the passage order first
    python -m repro.dense embed --corpus msmarco-passage --shard 0 --num-shards 8
    python -m repro.dense queries --corpus msmarco-passage    # after every shard is done
    python -m repro.dense search --corpus msmarco-passage
    python -m repro.dense pools --corpus msmarco-passage

Intermediate files live in ``data/dense/<corpus>/``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import sys
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from repro import records

MODEL = "Qwen/Qwen3-Embedding-4B"
REVISION = "5cf2132abc99cad020ac570b19d031efec650f2b"
DIM = 2560
TASK = "Given a web search query, retrieve relevant passages that answer the query"
TOP_K = 100

#: corpus -> (benchmarks it serves, token limit for passages and queries)
CORPORA: dict[str, tuple[tuple[str, ...], int]] = {
    "msmarco-passage": (("dl19", "dl20"), 512),
    "beir/trec-covid": (("covid",), 1024),
    "beir/nfcorpus": (("nfcorpus",), 1024),
    "beir/scifact": (("scifact",), 1024),
    "beir/webis-touche2020": (("touche",), 1024),
    "beir/dbpedia-entity": (("dbpedia",), 1024),
    "beir/robust04": (("robust04",), 1024),
}
ROBUST04_INDEX = "beir-v1.0.0-robust04.flat"
MSMARCO_COLLECTION = Path.home() / ".ir_datasets" / "msmarco-passage" / "collection.tsv"


def work_dir(corpus: str) -> Path:
    return records.DATA / "dense" / corpus.replace("/", "-")


def query_text(query: str) -> str:
    """The instructed query, in the exact form of the model card (no space after the colon)."""
    return f"Instruct: {TASK}\nQuery:{query}"


def compose(corpus: str, title: str | None, text: str | None) -> str:
    """A passage's text as the BM25 pools hold it.

    >>> compose("beir/trec-covid", "T", "body")
    'Title: T Content: body'
    >>> compose("beir/scifact", "", " body ")
    'body'
    """
    text = text or ""
    if corpus == "beir/trec-covid":
        return f"Title: {title or ''} Content: {text}".rstrip()
    title, text = (title or "").strip(), text.strip()
    return f"{title} {text}".strip() if title else text


def iter_docs(corpus: str, msmarco_collection: Path = MSMARCO_COLLECTION) -> Iterator[tuple[str, str]]:
    """(docid, text) of every passage, in a fixed order, first occurrence of each docid only."""
    seen: set[str] = set()
    for docid, text in _iter_raw(corpus, msmarco_collection):
        if docid not in seen:
            seen.add(docid)
            yield docid, text


def _iter_raw(corpus: str, msmarco_collection: Path) -> Iterator[tuple[str, str]]:
    if corpus == "msmarco-passage":
        with open(msmarco_collection, encoding="utf-8") as handle:
            for line in handle:
                pid, _, text = line.partition("\t")
                yield pid, text.rstrip("\n")
    elif corpus == "beir/robust04":
        with open(work_dir(corpus) / "docs.jsonl", encoding="utf-8") as handle:
            for line in handle:
                rec = json.loads(line)
                yield rec["docid"], rec["text"]
    else:
        import ir_datasets

        for doc in ir_datasets.load(corpus).docs_iter():
            yield doc.doc_id, compose(corpus, getattr(doc, "title", None), doc.text)


def extract_robust04() -> None:
    """Dump the Robust04 passages from Pyserini's prebuilt BEIR index to ``docs.jsonl``."""
    from pyserini.search.lucene import LuceneSearcher

    searcher = LuceneSearcher.from_prebuilt_index(ROBUST04_INDEX)
    out = work_dir("beir/robust04") / "docs.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".jsonl.part")
    n = 0
    with open(tmp, "w", encoding="utf-8") as handle:
        for i in range(searcher.num_docs):
            doc = searcher.doc(i)
            if doc is None or doc.raw() is None:
                continue
            raw = json.loads(doc.raw())
            text = compose("beir/robust04", raw.get("title"), raw.get("text"))
            handle.write(json.dumps({"docid": doc.docid(), "text": text}, ensure_ascii=False) + "\n")
            n += 1
    tmp.rename(out)
    print(f"{out}: {n} passages", flush=True)


def write_docids(corpus: str, msmarco_collection: Path) -> int:
    """Freeze the passage order: row ``i`` of the embeddings is line ``i`` of ``docids.txt``."""
    path = work_dir(corpus) / "docids.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".txt.part")
    n = 0
    with open(tmp, "w", encoding="utf-8") as handle:
        for docid, _ in iter_docs(corpus, msmarco_collection):
            handle.write(docid + "\n")
            n += 1
    tmp.rename(path)
    print(f"{path}: {n} passages", flush=True)
    return n


def load_docids(corpus: str) -> list[str]:
    with open(work_dir(corpus) / "docids.txt", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


class Encoder:
    """Qwen3-Embedding-4B, last-token pooled and L2-normalised, with vLLM or transformers."""

    def __init__(self, backend: str, max_len: int, gpu_memory_utilization: float = 0.85) -> None:
        from transformers import AutoTokenizer

        self.backend = backend
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, padding_side="left")
        if backend == "vllm":
            from vllm import LLM

            # max_model_len is above the truncation length: a prompt of exactly
            # max_model_len tokens can stall the scheduler.
            self.model: Any = LLM(
                model=MODEL,
                revision=REVISION,
                runner="pooling",
                max_model_len=max_len + 128,
                gpu_memory_utilization=gpu_memory_utilization,
                dtype="bfloat16",
                enable_prefix_caching=False,
                max_num_batched_tokens=65536,
                pooler_config={"pooling_type": "LAST"},
            )
        elif backend == "transformers":
            import torch
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(MODEL, revision=REVISION, dtype=torch.float16).to("cuda").eval()
        else:
            raise ValueError(f"unknown backend {backend!r}")

    def encode(self, texts: list[str], batch_size: int = 8) -> Any:
        """A float32 array of shape ``(len(texts), 2560)``, one unit vector per text."""
        import numpy as np

        if not texts:
            return np.zeros((0, DIM), dtype=np.float32)
        if self.backend == "vllm":
            from vllm.inputs import TokensPrompt

            ids = self.tokenizer(texts, truncation=True, max_length=self.max_len)["input_ids"]
            outputs = self.model.embed([TokensPrompt(prompt_token_ids=i) for i in ids])
            vectors = np.asarray([o.outputs.embedding for o in outputs], dtype=np.float32)
        else:
            import torch

            parts = []
            for start in range(0, len(texts), batch_size):
                batch = self.tokenizer(
                    texts[start : start + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.max_len,
                    return_tensors="pt",
                ).to("cuda")
                with torch.no_grad():
                    hidden = self.model(**batch).last_hidden_state[:, -1]  # left padding: the last token
                parts.append(hidden.float().cpu().numpy())
            vectors = np.concatenate(parts)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)
        return vectors


def embed_shard(corpus: str, shard: int, num_shards: int, encoder: Encoder, msmarco_collection: Path) -> None:
    """Embed rows ``[start, end)`` of the corpus into ``emb/shard_XXXX.npy`` (fp16)."""
    import numpy as np

    n_docs = len(load_docids(corpus))
    per = -(-n_docs // num_shards)
    start, end = min(shard * per, n_docs), min((shard + 1) * per, n_docs)
    out = work_dir(corpus) / "emb"
    out.mkdir(parents=True, exist_ok=True)
    npy, meta = out / f"shard_{shard:04d}.npy", out / f"shard_{shard:04d}.meta.json"
    if meta.exists() and json.loads(meta.read_text())["rows"] == end - start:
        print(f"{corpus} shard {shard}: done already", flush=True)
        return
    matrix = np.lib.format.open_memmap(npy, mode="w+", dtype=np.float16, shape=(end - start, DIM))
    written, buffer, t0 = 0, [], time.time()

    def flush() -> None:
        nonlocal written
        if buffer:
            matrix[written : written + len(buffer)] = encoder.encode(buffer).astype(np.float16)
            written += len(buffer)
            buffer.clear()
            rate = written / (time.time() - t0)
            print(f"  {written}/{end - start} passages, {rate:.0f}/s", flush=True)

    for i, (_, text) in enumerate(iter_docs(corpus, msmarco_collection)):
        if i >= end:
            break
        if i >= start:
            buffer.append(text)
            if len(buffer) >= 100_000:
                flush()
    flush()
    matrix.flush()
    if written != end - start:
        raise RuntimeError(f"wrote {written} rows, expected {end - start}")
    meta.write_text(json.dumps({"start": start, "end": end, "rows": end - start, "model": MODEL, "revision": REVISION}))


def embed_queries(corpus: str, encoder: Encoder) -> None:
    """Embed the instructed queries of every benchmark the corpus serves."""
    import numpy as np

    out = work_dir(corpus) / "queries"
    out.mkdir(parents=True, exist_ok=True)
    for benchmark in CORPORA[corpus][0]:
        pool = records.load_pool("bm25", benchmark)
        np.save(out / f"{benchmark}.npy", encoder.encode([query_text(q["query"]) for q in pool]))
        (out / f"{benchmark}.qids.json").write_text(json.dumps([str(q["qid"]) for q in pool]))


def search(corpus: str, k: int = TOP_K, block: int = 200_000) -> None:
    """Exact top ``k`` by inner product over every shard, for every benchmark of the corpus."""
    import numpy as np
    import torch

    shards = sorted((work_dir(corpus) / "emb").glob("shard_*.npy"))
    metas = [json.loads(p.with_suffix(".meta.json").read_text()) for p in shards]
    order = sorted(range(len(shards)), key=lambda i: metas[i]["start"])
    docids = load_docids(corpus)
    covered = 0
    for i in order:
        if metas[i]["start"] != covered:
            raise RuntimeError(f"{corpus}: the shards do not tile the corpus at row {covered}")
        covered = metas[i]["end"]
    if covered != len(docids):
        raise RuntimeError(f"{corpus}: the shards cover {covered} of {len(docids)} passages")

    for benchmark in CORPORA[corpus][0]:
        qdir = work_dir(corpus) / "queries"
        queries = torch.from_numpy(np.load(qdir / f"{benchmark}.npy")).to("cuda", torch.float16)
        qids = json.loads((qdir / f"{benchmark}.qids.json").read_text())
        nq = queries.shape[0]
        # The running top k stays in float32: fp16 would reorder near ties.
        best_s = torch.zeros((nq, 0), device="cuda", dtype=torch.float32)
        best_i = torch.zeros((nq, 0), device="cuda", dtype=torch.int64)
        for i in order:
            rows = np.load(shards[i], mmap_mode="r")
            base = metas[i]["start"]
            for s in range(0, rows.shape[0], block):
                chunk = torch.from_numpy(np.ascontiguousarray(rows[s : s + block])).to("cuda")
                scores = torch.cat([best_s, (queries @ chunk.T).float()], dim=1)
                index = torch.arange(base + s, base + s + chunk.shape[0], device="cuda").expand(nq, -1)
                best_s, sel = torch.topk(scores, min(k, scores.shape[1]), dim=1)
                best_i = torch.gather(torch.cat([best_i, index], dim=1), 1, sel)
        result = {
            qid: [[docids[int(r)], float(v)] for v, r in zip(best_s[j].tolist(), best_i[j].tolist())]
            for j, qid in enumerate(qids)
        }
        (work_dir(corpus) / f"top{k}-{benchmark}.json").write_text(json.dumps(result))
        print(f"{corpus}: top {k} for {nq} {benchmark} queries", flush=True)


def write_pools(corpus: str, msmarco_collection: Path, k: int = TOP_K) -> None:
    """``data/pools/dense/<benchmark>.jsonl``, in the BM25 pool's query order and schema."""
    results = {b: json.loads((work_dir(corpus) / f"top{k}-{b}.json").read_text()) for b in CORPORA[corpus][0]}
    needed = {d for res in results.values() for pairs in res.values() for d, _ in pairs}
    texts: dict[str, str] = {}
    for docid, text in iter_docs(corpus, msmarco_collection):
        if docid in needed:
            texts[docid] = text
            if len(texts) == len(needed):
                break
    if len(texts) != len(needed):
        raise RuntimeError(f"{corpus}: {len(needed) - len(texts)} retrieved passages have no text")
    from repro.data import write_pool

    for benchmark, res in results.items():
        bm25 = records.load_pool("bm25", benchmark)
        if {str(q["qid"]) for q in bm25} != set(res):
            raise RuntimeError(f"{benchmark}: the dense and BM25 pools have different queries")
        pool = (
            {
                "qid": str(q["qid"]),
                "query": q["query"],
                "candidates": [{"docid": d, "score": s, "text": texts[d]} for d, s in res[str(q["qid"])]],
            }
            for q in bm25
        )
        n = write_pool(records.pool_path("dense", benchmark), pool)
        print(f"  -> pools/dense/{benchmark}.jsonl: {n} queries", flush=True)


def _exit_without_engine(code: int = 0) -> None:
    """Exit after a vLLM run: kill vLLM's engine processes first, which would otherwise keep the GPU."""
    me = os.getpid()
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", "rb") as handle:
                parent = int(handle.read().rsplit(b")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        if parent == me:
            with contextlib.suppress(OSError):
                os.kill(int(entry), signal.SIGKILL)
    sys.stdout.flush()
    os._exit(code)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m repro.dense", description=__doc__.split("\n\n")[0])
    parser.add_argument("step", choices=["extract-robust04", "docids", "embed", "queries", "search", "pools", "all"])
    parser.add_argument("--corpus", choices=sorted(CORPORA), help="required for every step but extract-robust04")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--msmarco-collection", type=Path, default=MSMARCO_COLLECTION)
    args = parser.parse_args(argv)

    if args.step == "extract-robust04":
        extract_robust04()
        return 0
    if args.corpus is None:
        parser.error("--corpus is required")
    corpus, max_len = args.corpus, CORPORA[args.corpus][1]
    if args.step in ("docids", "all"):
        write_docids(corpus, args.msmarco_collection)
    if args.step in ("embed", "queries", "all"):
        encoder = Encoder(args.backend, max_len, args.gpu_memory_utilization)
        if args.step in ("embed", "all"):
            shards = range(args.num_shards) if args.step == "all" else [args.shard]
            for shard in shards:
                embed_shard(corpus, shard, args.num_shards, encoder, args.msmarco_collection)
        if args.step in ("queries", "all"):
            embed_queries(corpus, encoder)
        if args.backend == "vllm" and args.step != "all":
            _exit_without_engine(0)
    if args.step in ("search", "all"):
        search(corpus)
    if args.step in ("pools", "all"):
        write_pools(corpus, args.msmarco_collection)
    if args.step == "all" and args.backend == "vllm":
        _exit_without_engine(0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
