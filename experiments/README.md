# Reproducing the paper

This directory holds the code, configurations and commands behind every table, figure and
measured claim of *Deal With It: On Better Schedules for Listwise Reranking*. Each experiment is
one YAML file in `configs/`. One command runs it, writes one JSON record per query under `runs/`,
and resumes where it stopped if interrupted. The tables, figures and statistics are then computed
from those records.

**What is and is not in the repository.** The per-query run records of the paper are not shipped,
and neither are the pools. Checking a table or a measured claim therefore means running its
configurations again, at the cost given below, against API endpoints and a served model whose
answers will not be identical to ours, so expect agreement within the reported intervals, not to
the digit. What runs without any data, key or GPU is: Proposition 1's numbers
(`python -m repro.analysis.expected_cuts`), the offline tests, dry runs of every configuration,
and each figure drawn from the inputs shipped in `repro/figures/data/`. Those inputs are the
aggregate values the figure scripts computed from our records (`--from-runs --write-data`); drawing
from them redraws the paper's figures and does not re-derive their values.
[Conventions](#conventions) defines what each printed number is.

All commands below run from this directory (`experiments/`).

| Paper | Configurations | Then |
|---|---|---|
| Table 1 | `table1_gemini`, `table1_gemini_repeats`, `table1_local_timing` | `python -m repro.tables --table 1` |
| Table 2 | `table2_models` (Gemini 3 Flash's row comes from Table 1's runs) | `python -m repro.tables --table 2` |
| Table 3 | `table3_depth`, and for its 100-candidate row `table2_models` and `table1_local_timing` | `python -m repro.tables --table 3` |
| Figure 1 | `fig1_api_timing` and the Table 1 and Table 2 configurations | `python -m repro.figures.fig1_hero --from-runs` |
| Figure 2 | `table1_gemini`, `dense_gemini` | `python -m repro.figures.fig2_split --from-runs` |
| Figure 3 | `fig3_split_ablation`, `dense_gemini`, `table1_gemini`, `table1_gemini_repeats` | `python -m repro.figures.fig3_forest --from-runs` |
| Proposition 1 | none (no LLM call) | `python -m repro.analysis.expected_cuts` |
| Numbers quoted in the text | see [Claims in the text](#claims-in-the-text) | one `python -m repro.analysis.<name>` each |

Run each configuration with `python -m repro.run configs/<name>.yaml`, then the command in the last
column. Reproducing everything costs roughly USD 3,100 in API calls, about 100 hours on two H100
GPUs for the local runs and about 7 GPU hours to build the dense pools
(see [step 5](#5-cost-and-hardware)).

## Setup

```bash
pip install -e "..[experiments]"     # dealrank, plus openai, numpy, pyyaml, ir-measures, matplotlib, pytest
pip install blitzrank==0.1.0         # the BlitzRank authors' package
pip install --force-reinstall --no-deps pytrec-eval-terrier
export OPENROUTER_API_KEY=...        # every API model is called through OpenRouter
```

The Table 1, Figure 1 and Figure 3 configs run BlitzRank from its authors' package, and `repro.run`
refuses to start them without it. `blitzrank` depends on `pytrec-eval` and `ir-measures` on
`pytrec-eval-terrier`. Both install a module named `pytrec_eval`, and `ir-measures` rejects the
one from `pytrec-eval`. The third command therefore reinstalls `pytrec-eval-terrier` on top, which
serves both, since `blitzrank` uses the module only in its own evaluation code and the two modules
share that interface.

`configs/models.yaml` lists the seven LLMs, with their OpenRouter ids and routing, the prices used
for the cost estimates below, and the local model's revision and serving settings. Every model gets
the same RankGPT prompt and parser, temperature 0, seed 42, at most 1,024 output tokens, passages
truncated to 300 words, and reasoning disabled. The runs of the BlitzRank package keep these
settings but build the prompt and parse the answer with the package's own code (its RankGPT-style
listwise comparer), and use its retry policy and its LiteLLM transport. This covers every BlitzRank
number obtained with Gemini 3 Flash: the BlitzRank rows of Table 1 (nDCG@10, calls, rounds,
tokens), Gemini's row of Table 2, Figure 1(b) and the BlitzRank rows of Figure 3. Every other
schedule, and BlitzRank with the five other LLMs, uses this library's prompt and parser
(`../src/dealrank/prompt.py`).

The requirements are lower bounds, not a frozen environment: there is no lock file, and the paper's
local runs used vLLM 0.33 where the `vllm` extra accepts any later version. Latencies depend on the
serving stack and hardware, and the API models are served by providers that can change them
(`google/gemini-3-flash-preview` is a preview endpoint).

The paper writes *w* for the number of passages per call. BlitzRank's authors call it *k*; in this
code it is `window`.

## 1. Data

```bash
python -m repro.data bm25     # BM25 top 100 of the ten benchmarks, and their qrels
python -m repro.data deep     # BM25 top 1,000 of DL19, DL20, DBpedia and SciFact
python -m repro.data check    # lists every pool with its query count, and any missing file
```

Run `bm25` first, since the other pools take their queries from it.

| pool | benchmarks | queries | source | needs |
|---|---|---|---|---|
| `bm25` | DL19, DL20, TREC-COVID, NFCorpus, Signal, News, Robust04, Touché, DBpedia, SciFact | 1,622 | `castorini/rank_llm_data`; qrels from `castorini/eval` | about 450 MB of downloads |
| `bm25-top1000` | DL19, DL20, DBpedia, SciFact | 797 | DL19, DL20: `castorini/rank_llm_data`; DBpedia, SciFact: Pyserini over the prebuilt BEIR flat indexes (k1 = 0.9, b = 0.4) | `pip install pyserini`, Java 21, about 3 GB of index for DBpedia |
| `dense` | the ten but Signal and News | 1,468 | Qwen3-Embedding-4B, exact inner-product top 100 over each full corpus | a GPU, `torch`, `transformers`, `ir_datasets` (and `vllm`, the faster backend) |

The BM25 top 100 is shorter than 100 for 125 queries, 15 of them NFCorpus queries with no
candidate at all. These pools are kept as published. nDCG@10 is computed over all 1,622 queries
(a query with no candidate scores 0); the cost columns of Table 1 use the 1,497 whose pool holds a
full 100 candidates. On the short pools the schedules behave as their classes document: a pool of
at most 20 candidates is one call, DealRank with a quota of four sends fewer than 20 passages to
its final call when there are fewer than five hands, and BlitzRank's opening calls on dealt input
are not exactly the dealt hands for 32 queries (see `../src/dealrank/schedules/blitzrank.py`).

The pools are downloaded from the default branch of `castorini/rank_llm_data`, and the qrels from
`castorini/eval` (formerly `castorini/anserini-tools`) at a pinned revision; `repro.data` checks
the number of queries of each pool. The top-1,000 pools of DBpedia and SciFact are retrieved here
with Pyserini, while their top-100 pools are the published ones, so the first 100 of the former
need not equal the latter.

The dense pools are the only GPU step of the data. Their embeddings take about 75 GB in fp16
(45 GB for MS MARCO, 24 GB for DBpedia), and on one H100 all corpora take about 7 GPU hours.
`python -m repro.dense --help` lists the seven corpora.

```bash
python -m repro.dense extract-robust04                    # Robust04's corpus, from Pyserini (Java 21)
python -m repro.dense all --corpus beir/scifact           # every step for one corpus on one GPU

# A large corpus over several GPUs, step by step:
python -m repro.dense docids --corpus msmarco-passage     # fixes the passage order, once
python -m repro.dense embed --corpus msmarco-passage --shard 0 --num-shards 8   # one shard per GPU
python -m repro.dense queries --corpus msmarco-passage    # once every shard is done
python -m repro.dense search --corpus msmarco-passage
python -m repro.dense pools --corpus msmarco-passage
```

Robust04's corpus comes from the `extract-robust04` step and the other BEIR corpora from
`ir_datasets`, which downloads them on first use. MS MARCO is read from the `collection.tsv` that
`ir_datasets` keeps under `~/.ir_datasets/msmarco-passage/`. `repro.dense` does not download it, so
fetch it once before the `docids` step:

```bash
python -c "import ir_datasets; next(iter(ir_datasets.load('msmarco-passage').docs_iter()))"
```

or pass the `collection.tsv` of the MS MARCO passage ranking collection (8,841,823 passages) with
`--msmarco-collection`. The dense step needs packages that no extra of this package installs:
`torch`, `transformers`, `ir_datasets`, `pyserini` (with Java 21) for Robust04, and `vllm` for the
default backend (`--backend transformers` does without it). We give no pinned versions for them.

Every pool is written as `data/pools/<pool>/<benchmark>.jsonl`, one query per line,
`{"qid", "query", "candidates": [{"docid", "score", "text"}]}` in retriever order, and qrels as
`data/qrels/<benchmark>.txt` in TREC format. Set `DEALRANK_DATA` to keep the data elsewhere.

## 2. Serving the local model

Qwen3.8-27B runs on a local vLLM server, in bf16 at the pinned revision, with tensor parallelism
over two GPUs, prefix caching off and seed 42. The paper's runs used vLLM 0.33 on two H100s.

```bash
python scripts/serve_vllm.py qwen3.8-27b --gpus 0,1 --port 8000           # serves until stopped
python scripts/serve_vllm.py qwen3.8-27b --gpus 0,1 --port 8000 --dry-run # prints the command
```

The experiments call `http://127.0.0.1:8000/v1`. For another address, pass `--base-url` to
`repro.run` or set `DEALRANK_VLLM_URL`. Keep one server per pair of GPUs and nothing else on them
while timing.

## 3. Running an experiment

```bash
python -m repro.run configs/table1_gemini.yaml
python -m repro.run table1_gemini --benchmarks dl19 dl20 --max-queries 5    # a subset
python -m repro.run table2_models --llms kimi-k2.6                          # one LLM of a config
```

A config has quality grids, timing blocks, or both. Quality grids run every (LLM, schedule, pool)
cell with several queries in flight. Timing blocks run one query at a time and, for each query,
every method back to back in a seeded random order, after a warm-up whose results are discarded.
Under load, a timing block instead runs each method as a block with a fixed number of queries in
flight. `repro.run` runs the grids, then the timing blocks. `--skip-timing` skips the timing
blocks, and `python -m repro.timing <config> [--repeat N]` runs them alone.

Runs resume. Queries already recorded are skipped, and a failed query is logged and left
unrecorded, so running the command again retries it. The exit status is 1 if any query failed.
Resuming is keyed by the record file's path and the query id alone: a record does not store the
prompt, the model revision or the data it was made with, so use a fresh `DEALRANK_RUNS` directory
after changing any of them.

**Dry runs.** `--dry-run` replaces every LLM with `dealrank.FakeLLM`, which orders passages by a
hash of query and passage, makes no network call and writes to `runs-dry-run/`. It needs the pools
of step 1 and runs any config end to end in seconds. Without the `blitzrank` package, it skips the
cells that run it.

```bash
python -m repro.run configs/table2_models.yaml --dry-run --max-queries 2
```

## 4. Tables, figures and claims

Analyses read `runs/`, and the pools and qrels of step 1 to score them. They stop with the list of
missing record files if a config they need has not been run at all. They do not require every
query of a pool: a comparison uses the queries that all of its arms recorded, so a run cut short
with `--max-queries`, or a query that failed and was left unrecorded, shrinks the population
without stopping the analysis. Each record file that holds fewer queries than its pool is named in
a warning on standard error, as is a Table 3 timing arm with another number of repeats than the
paper's. Read those warnings before quoting an output. Every analysis prints Markdown, or JSON with
`--json`. Tables and figures are written to `out/` (set `DEALRANK_OUT` to change it).

### Table 1: schedules with Gemini 3 Flash

```bash
python -m repro.run configs/table1_gemini.yaml           # every row, one run each
python -m repro.run configs/table1_gemini_repeats.yaml   # runs 2 and 3, random-partition seeds 1 to 9
python -m repro.run configs/table1_local_timing.yaml     # p50 and p90 (local server)
python -m repro.tables --table 1                         # writes out/tables/main_gemini.tex
```

Without `--table`, `repro.tables` computes all three tables, so it also needs the configs of
Tables 2 and 3 below.

How each column of Table 1 is obtained:

- **nDCG@10.** DealRank, the sliding window, BlitzRank w=20 and BlitzRank w=20 on dealt input
  average each query over three runs. The random partition averages it over ten seeds, each
  drawing a fresh partition per query. Every other row is one run.
- **BlitzRank rows.** Their nDCG@10, calls, rounds and tokens come from the authors' package, run
  one call at a time; "BlitzRank w=20" is that run as it is. The two rows marked "concurrent"
  reuse those sequential records: the package returns the same ranking either way, and their
  rounds count the first, mutually disjoint calls as one round. BlitzRank w=10 is not given
  concurrent first calls, in this table or elsewhere, so its 13.4 rounds are its 13.4 calls.
- **Calls, rounds, tokens.** Means per benchmark, then over the ten benchmarks, on the 1,497
  queries with a full 100 candidates.
- **p50 and p90.** Seconds per DL19 query with Qwen3.8-27B on the local server, one query in
  flight, pooled over three repeats. BlitzRank w=20, with every call in turn and with concurrent
  first calls, is timed with the authors' package; BlitzRank w=10 and BlitzRank w=20 on dealt input
  with this library's implementation. The contiguous-chunk and random-partition rows make
  DealRank's calls on another split and show DealRank's timings.

### Table 2: six LLMs and GPT-5.5

```bash
python -m repro.run configs/table2_models.yaml    # six LLMs (Qwen3.8-27B needs the local server)
python -m repro.tables --table 2                  # writes out/tables/models.tex
```

Gemini 3 Flash's row is Table 1's, from `table1_gemini` and `table1_gemini_repeats`, with the
authors' BlitzRank package. The other LLMs use this library's BlitzRank. The average row averages
each query's difference over the six LLMs, and its intervals resample benchmarks and queries with
the six LLMs held fixed. GPT-5.5, the frontier-model check, runs DealRank and BlitzRank w=10 and is
printed below the six, outside the average.

`repro.tables` also writes `out/tables/tables.json`, with every printed number unrounded, and the
average of BlitzRank on dealt input, which Table 2 does not print and Figure 1 uses.

### Table 3: 500 and 1,000 candidates

```bash
python -m repro.run configs/table3_depth.yaml   # needs the BM25 top-1,000 pools and the local server
python -m repro.tables --table 3                # writes out/tables/depth.tex
python -m repro.analysis.depth                  # the same numbers, with differences and ratios
```

The 100-candidate row takes its nDCG@10 from the Qwen3.8-27B runs of `table2_models` and its
seconds and rounds from `table1_local_timing`. `repro.analysis.depth` also prints each schedule's
nDCG@10 difference to DealRank with its interval, and the per-query ratio of its seconds to
DealRank's: at 1,000 candidates, 1.26 [1.25, 1.28] for BlitzRank with concurrent first calls and
11.2 [11.1, 11.4] for the sliding window.

At 500 and 1,000 candidates, hierarchical DealRank and BlitzRank with concurrent first calls are
timed three times on DL19. The two sequential schedules, the sliding window and BlitzRank with
every call in turn, are timed once, in blocks of their own. The top-1,000 pools of DBpedia and
SciFact are retrieved here, while the 100-candidate row uses the published top-100 pools.

### Figure 1: dealing against chunking, and latency

```bash
python -m repro.run configs/fig1_api_timing.yaml   # Gemini 3 Flash API, one query in flight
python -m repro.figures.fig1_hero --from-runs      # out/figures/fig1_hero.{pdf,png}
python -m repro.analysis.latency                   # the numbers behind panel (b)
```

Panel (b) is the median critical path per query, the sum over rounds of each round's slowest call,
over DL19, DL20, TREC-COVID and Touché and three repeats. The segments of a bar are not medians of
each round: they are the mean duration of each round, rescaled so that they sum to the median
critical path. Its nDCG@10 column is the mean over the six LLMs of Table 2: DealRank's mean, and
each other method's paired difference from it, with its 95% interval, as in Table 2's average row.
The certification shares and BlitzRank's rounds are those of Table 1's three runs on 100-candidate
queries, so `--from-runs` and `repro.analysis.latency` also need the Table 1 and Table 2 configs.
In this panel DealRank and the sliding window call the API through this library's OpenAI client and
BlitzRank through its package's LiteLLM transport.

### Figure 2: cuts against retriever quality

```bash
python -m repro.run configs/dense_gemini.yaml      # needs the dense pools
python -m repro.figures.fig2_split --from-runs     # out/figures/fig2_split.{pdf,png}
python -m repro.analysis.cuts                      # the pooled values, and the other measured cuts
```

Each point counts how many of the sliding window's top ten a split puts over its group's quota of
four, on the queries with 100 candidates: one small point per benchmark and retriever (ten over
BM25, eight over the dense retriever), and one large point per retriever for the pooled value with
its 95% interval. It needs only the sliding window's rankings of `table1_gemini` and
`dense_gemini`. The solid lines are Proposition 1 (`repro.analysis.expected_cuts`) and the dashed
lines the Mallows model of `repro.analysis.mallows`; neither is fitted to the points.

### Figure 3: dealing inside other schedules

```bash
python -m repro.run configs/dense_gemini.yaml          # needs the dense pools
python -m repro.run configs/fig3_split_ablation.yaml   # needs the dense pools
python -m repro.figures.fig3_forest --from-runs        # out/figures/fig3_forest.{pdf,png}
python -m repro.analysis.splits                        # every row's difference and interval
```

Figure 3 also reads the records of `table1_gemini` and `table1_gemini_repeats`. Its nDCG@10
differences use every query that both arms of a row recorded; its differences in calls use those
with a full 100 candidates. The "quota binds" column of `repro.analysis.splits` follows the paper's
grouping.

Without `--from-runs`, each figure script draws from the inputs shipped in `repro/figures/data/`,
which hold the paper's values. `--data FILE` reads another input, `--write-data FILE` saves the
inputs used, and `--out STEM` sets the output path.

### Conventions

What each number printed by the analyses is.

**Repeats.** "Three runs" holds for DealRank, the sliding window, BlitzRank w=20 and BlitzRank
w=20 on dealt input with Gemini 3 Flash in Table 1, for the API timing of Figure 1(b), and for the
local timing of Table 1, of the four benchmarks and of the two concurrent schedules of Table 3.
Every other cell is one run: the other rows of Table 1, the five other LLMs and GPT-5.5 in Table 2,
TourRank on its two other LLMs, the quality cells of Table 3, the timing of the two sequential
schedules at 500 and 1,000 candidates, the sliding-window rankings behind Figure 2, and the arms of
Figure 3, except that over BM25 its DealRank arm and its two BlitzRank w=20 arms are Table 1's
three runs. The random partition has ten seeds with Gemini 3 Flash and three in the quota sweep,
whose dealt arms are one run each.

**Prompt.** The runs of the BlitzRank package use the package's own RankGPT-style prompt builder,
parser, retries and transport ([Setup](#setup)). All other runs share this library's RankGPT prompt
and parser.

**Latency.** Two measures are used. The *critical path* is the sum over rounds of each round's
slowest call, where a call's latency is that of its successful attempt: a failed attempt and the
wait before the retry are not in it. *Wall-clock* is the client's time for the whole query and
includes them. Figure 1(b) reports median critical paths, and `repro.analysis.latency` prints the
median wall-clock next to each. Tables 1 and 3, the local ratios and the throughput use wall-clock,
from runs that allow a single attempt per call. The segments of Figure 1(b)'s bars are mean round
durations rescaled to the median critical path. Paired ratios of seconds are geometric means of
per-query ratios, each query averaged over its repeats, with the bootstrap of `repro/stats.py` on
their logarithms. The throughput ratios with 8 and 32 queries in flight have Student t intervals
over three repeats of each level, each on its own server instance, not the bootstrap of the nDCG@10
intervals.

**Populations.** nDCG@10 uses all queries of a pool, including the 125 BM25 pools shorter than
100; costs, certification shares and cut counts use the queries with a full 100 candidates.
Certification shares, calls and rounds are means per benchmark, then over benchmarks. The cuts
forced by a split (2.0, 2.8 and about 0.1, and the points of Figure 2) are means pooled over
queries, from one sliding-window run per retriever, with a plain bootstrap over queries. The 12% of
relevant passages in over-full random groups is a share of passages, pooled over queries and
seeds, while the 8.0 and 16.5 points average within a query, then per benchmark, then over
benchmarks. The nDCG@20 difference of `repro.analysis.ndcg_at_k` is the sliding window minus
DealRank.

**Intervals.** Every nDCG@10 interval comes from `repro/stats.py`: 10,000 replicates that resample
queries within each benchmark and then benchmarks. One query resample is drawn per benchmark per
replicate and reused when the benchmark is drawn more than once. Runs and seeds are averaged per
query before resampling, and Table 2's average holds the six LLMs fixed, so the intervals describe
resampled queries and benchmarks, not other LLMs or other runs. Table 3's intervals resample four
benchmarks.

#### Baselines as run here

All baselines are reimplemented in `../src/dealrank/schedules/` under this library's prompt, except
BlitzRank with Gemini 3 Flash, which runs its authors' package. Their settings, where the paper
does not give them:

| schedule | settings | differs from the original in |
|---|---|---|
| Sliding window | window 20, stride 10, one bottom-up pass | |
| TourRank | one tournament (`r = 1`), stage plan (5, 5, 1, 1, 1) groups advancing (50, 20, 10, 5, 2), ties by retriever rank | the model returns a permutation with the RankGPT prompt and the top *m* are its first *m* entries, where TourRank's prompt asks for the top *m*; the order inside a group is otherwise unused, as in TourRank's point system |
| TDPart | window 20, pivot at rank 10, budget 20 | all tail windows of a level run as one round (the released loop runs them in turn and stops when the budget is full). The budget is the original parameter *b*, equal to the window in the original evaluation: passages that beat the pivot beyond the first 20 are ranked below the cut, so TDPart has no per-group quota but its result can still depend on how the tail is cut into windows |
| JointRank | window 20, four replicates (20 calls in one round), seeded random equi-replicate design, PageRank with damping 0.85 | |
| ListK | top 10, window 20, 16 pivots in groups of two, six sort pivots | the pivots are the candidates at the retriever's next ranks, in place of ListK's embedding-based choice of pivots |
| BlitzRank | `w` = 20 or 10, certified top 10 | this library's version is a reimplementation; first calls issued concurrently are this repository's addition, to the package through `repro/blitzrank_package.py` and to the library through `concurrent_cover` |

### Claims in the text

Section numbers in this table are the paper's.

| claim | runs needed | command |
|---|---|---|
| Proposition 1's expected cuts: chunks 6, 2.9 and 1.1 against 0, 0.05 and 0.10 for dealing; a random partition's 0.149, its 12.7% of lists, and 0.69 and 1.68 at quotas of two and one; Proposition 2's 6 and 30 calls (Sections 2 and 4) | none | `python -m repro.analysis.expected_cuts` |
| The noisy ranker's measured curve: 51,955 first-round calls, concave up to x = 7, dealing optimal under it at every depth from 10 to 100 (Section 2) | `table1_gemini`, `dense_gemini` | `python -m repro.analysis.retention` |
| BlitzRank's certification call on 59% of 100-candidate queries, 14% on dealt input, with no significant change in nDCG@10 (abstract, Sections 1 and 4) | `table1_gemini`, `table1_gemini_repeats` | `python -m repro.analysis.certification` |
| Dealing inside five other schedules: chunks' cost of 2.7 points in DealRank, TourRank's 2.2 and 4.5 points, BlitzRank's 0.46 to 0.57 fewer calls, at most 0.6 points where no quota binds; TourRank 1.44 [0.49, 2.72] points behind DealRank across three LLMs (Sections 1 and 4, Figure 3) | the Figure 3 configs, `table2_models` | `python -m repro.analysis.splits` |
| Chunks' cost of 4.4 points over the dense retriever against 2.3 over BM25 on the same queries, a difference of 2.1 [0.2, 4.1]; the sliding window's top ten that a split puts over quota (2.0, 2.8, about 0.1); a random partition's 0.07 more over quota and 0.07 [-0.01, +0.15] fewer at the final call; its over-full groups (12%, 8.0 and 16.5 points) (Section 4, Figure 2) | `table1_gemini`, `table1_gemini_repeats`, `dense_gemini` | `python -m repro.analysis.cuts` |
| A random partition against dealing over ten seeds, -0.3 [-0.8, +0.2] (Section 4) | `table1_gemini`, `table1_gemini_repeats` | `python -m repro.tables --table 1` (the "Random partition" row) |
| A query's nDCG@10 varies more across three seeds of a random partition than across three DealRank runs, by 0.72 [0.43, 1.05] points of standard deviation (Section 4) | `table1_gemini`, `table1_gemini_repeats` | `python -m repro.analysis.seed_stability` |
| A random partition against dealing at quotas of four, two and one with Qwen3.8-27B: 1.3 [0.4, 2.4] at a quota of one over the dense retriever (Section 4) | `quota_sweep` | `python -m repro.analysis.quota` |
| nDCG@20 of the sliding window against DealRank (Section 4) | `table1_gemini`, `table1_gemini_repeats` | `python -m repro.analysis.ndcg_at_k` |
| API latency of 3.4, 4.0 and 13.4 seconds (3.5 on dealt input), within 0.3 seconds of wall-clock; local latency on four benchmarks, 1.20 [1.14, 1.26], 1.03 [1.01, 1.05] and 3.31 [2.96, 3.69] times DealRank's; 90th percentiles 0.36 and 1.23 seconds above the median; throughput with 8 and 32 queries in flight, 1.09 [1.06, 1.13] and 1.15 [1.10, 1.20] times concurrent BlitzRank's, 1.92 [1.86, 1.97] and 1.54 [1.51, 1.58] times the sliding window's, 1.01 [0.99, 1.02] and 1.05 [1.01, 1.10] times BlitzRank's on dealt input (Section 4) | `fig1_api_timing`, `local_latency`, `load`, and the Table 1 and Table 2 configs | `python -m repro.analysis.latency` |
| 500 and 1,000 candidates: Table 3, and the latency ratios of 1.26 and 11.2 at 1,000 (Section 4) | `table3_depth`, `table2_models`, `table1_local_timing` | `python -m repro.analysis.depth` |
| BlitzRank's reported results reproduced within 0.3 points at w=20. The published numbers are transcribed in `repro/analysis/published.py` (Section 3, footnote) | `table1_gemini`, `table1_gemini_repeats` | `python -m repro.analysis.published` |
| TourRank with ten tournaments, about 130 calls and 481k tokens per query (Section 3, footnote) | `table1_gemini` | ten times the calls and tokens of Table 1's TourRank row: an extrapolation, since no configuration runs ten tournaments |

These rows read configs that run on the local model, besides those of the tables:

```bash
python -m repro.run configs/quota_sweep.yaml     # BM25 and dense pools
python -m repro.run configs/local_latency.yaml   # DL19, DL20, TREC-COVID and Touché, one query in flight
python -m repro.timing configs/load.yaml --repeat 1   # 8 queries in flight; repeats 2 and 3 on fresh servers
python -m repro.timing configs/load.yaml --repeat 4   # 32 queries in flight; repeats 5 and 6 on fresh servers
```

The paper's three repeats of each load level each ran on its own server instance, so the interval
over repeats includes the variation between instances (see the header of `configs/load.yaml`).

## 5. Cost and hardware

The estimates below are rough, at the prices in `configs/models.yaml`. The API quality grids run 12
queries in flight, and local times are for two H100s.

| config | reproduces | LLM calls | cost or GPU time |
|---|---|---|---|
| `table1_gemini` | Table 1, nDCG@10 and cost columns | about 144,000 | about USD 360 |
| `table1_gemini_repeats` | Table 1, runs 2 and 3 and seeds 1 to 9 | about 168,000 | about USD 475 |
| `table1_local_timing` | Table 1, p50 and p90; Table 3's 100-candidate seconds | about 13,000 (local) | about 2 hours |
| `table2_models` | Table 2; TourRank on two more LLMs | about 500,000 API and 104,000 local | about USD 1,350, and about 9 hours |
| `table3_depth` | Table 3 | about 282,000 (local) | about 45 hours |
| `fig1_api_timing` | Figure 1(b) | about 16,000 | about USD 40, about 4.5 hours |
| `dense_gemini` | Figures 2 and 3, dense-retriever ablation | about 50,000 | about USD 140 |
| `fig3_split_ablation` | Figure 3 | about 342,000 | about USD 760 |
| `quota_sweep` | quota sweep (Section 4) | about 455,000 (local) | about 40 hours |
| `local_latency` | local latency on four benchmarks (Section 4) | about 16,000 (local) | about 4 hours |
| `load` | throughput under load (Section 4) | local | about 35 minutes per repeat, six repeats |
| dense pools | Figures 2 and 3, dense-retriever ablation, quota sweep | | about 7 hours on one H100 |

Each config's header gives its own figures.

## 6. Layout of runs

```
runs/<experiment>/<llm>/<schedule>/<pool>/<benchmark>[-n<depth>][.r<repeat>].jsonl
```

`<experiment>` is the config's file stem, `<llm>` a key of `configs/models.yaml`, and `<schedule>`
the name of the schedule built from its spec (`dealrank-w20-q4`, `sliding-window-w20-s10`,
`blitzrank-package-w20-m10-dealt`, ...). `-n<depth>` marks the deeper pools of `table3_depth`, and
`.r<repeat>` a repeated run, either the second and third runs of `table1_gemini_repeats` or a
timing repeat. For example:

```
runs/table1_gemini/gemini-3-flash/dealrank-w20-q4/bm25/dl19.jsonl
runs/table1_gemini_repeats/gemini-3-flash/dealrank-w20-q4/bm25/dl19.r2.jsonl
runs/table3_depth/qwen3.8-27b/dealrank-hier-w20-q2/bm25-top1000/dl19-n1000.r1.jsonl
```

Each line holds one query, with the ranking, the number of candidates, rounds, calls and tokens,
every call's start and end time grouped by round, the critical path and wall-clock seconds, and
the schedule's diagnostics. Timed records add a `timing` field (repeat, position in the method
order, seed, queries in flight). Runs under load also write one line per block to
`runs/load/blocks.jsonl`, with its queries per second. Set `DEALRANK_RUNS` to keep runs elsewhere.

A schedule's name carries the arguments the paper varies. It does not carry TourRank's shuffle
seed, or JointRank's design seed and damping, and a record stores neither the configuration nor
the data it was made from. Runs made with other values of those belong in another `DEALRANK_RUNS`.

`repro/paper_rows.py` names, for every row of every table and figure, the experiment, LLM, schedule
spec and repeats it reads, and `tests/test_coverage.py` checks that some config produces each of
them.

## 7. Adding a schedule

A schedule is a spec in a config's `schedules:` list, a mapping with a `type` and the schedule's
arguments, as `dealrank.build_schedule` accepts it:

```yaml
schedules:
  - {type: dealrank, window: 20, quota: 4}
  - {type: dealrank, window: 10, quota: 2, final_size: 20}
  - {type: tourrank, tournaments: 1, split: chunks}
```

`type: blitzrank_package` runs the authors' package instead of the library. A new kind of schedule
is a subclass of `dealrank.schedules.Schedule` with a unique `name`, registered in
`dealrank.schedules.SCHEDULES`, and its runs then land under its name. To put it in a table, add
its spec to `repro/paper_rows.py`. Check a config before spending on it:

```bash
python -m repro.run configs/my_experiment.yaml --dry-run --max-queries 2
```

## Tests

```bash
pytest experiments/tests    # from the repository root, offline, about 20 seconds
```

The `experiments` extra installs pytest. The tests build tiny fixture pools, dry-run every config,
and check the records, resumption, the timing harness, the data normalisers, that every config
parses, that every record file the analyses read is produced by some config, the bootstrap, the
nDCG implementation against `ir_measures`, the table generator on synthetic records, Proposition
1's numbers and the figure rendering. Without optional packages some are skipped: the adapter of
the BlitzRank package (tested against mocked API responses) and the dry-run cells that use it
need `blitzrank`, one config test needs `openai`, and the figure tests need `matplotlib`.

What the tests do not cover: they check plumbing and arithmetic on synthetic or dry-run records,
not agreement with the paper's numbers, which needs the runs. Of the analysis modules,
`expected_cuts`, the tables and `quota` are exercised; `certification`, `cuts`, `splits`,
`retention`, `seed_stability`, `published`, `ndcg_at_k`, `latency` and `depth` have no test of
their own beyond the check that their input files are produced by some config.
