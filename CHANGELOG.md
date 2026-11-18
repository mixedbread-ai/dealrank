# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [0.1.0]

First public release, accompanying the paper "Deal With It: On Better Schedules for Listwise
Reranking" ([arXiv:2611.22392](https://arxiv.org/abs/2611.22392)).

### Added

- `deal`, `chunks` and `random_partition` splits, `dealt_order` and `count_cuts`.
- `DealRank`, the two-round dealt schedule, and `HierarchicalDealRank` for 500 to 1,000 candidates.
- `BlitzRank`, with dealt input by default and optional concurrent opening calls.
- The paper's baselines under the same prompt and parser: sliding window, TourRank, TDPart,
  JointRank, ListK's quickselect, a single call over the top 20, and the retriever's order.
- LLM backends for OpenAI-compatible endpoints (OpenAI, OpenRouter, vLLM), LiteLLM and an offline
  `FakeLLM`, with concurrent calls within a round and an optional SQLite response cache.
- The `dealrank` command line (`rerank`, `deal`, `serve-vllm`).
- `experiments/`: configurations and scripts that reproduce every table and figure of the paper.
