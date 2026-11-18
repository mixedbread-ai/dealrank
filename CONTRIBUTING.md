# Contributing

Bug reports, questions and pull requests are welcome.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,openai,experiments]"
pip install pre-commit && pre-commit install   # optional: runs ruff on every commit
```

## Before opening a pull request

```bash
ruff check .
ruff format --check .
pytest
```

The tests run offline in under a minute. Every schedule is exercised with `dealrank.FakeLLM`,
which needs no network and no API key. Please add a test for any change in behaviour.

## Changes to schedules

The schedules in `src/dealrank/schedules/` reproduce the runs reported in the paper, including
their prompt, parser, tie-breaks and seeds. A change that alters which calls a schedule makes, or in
what order, changes published results. Such changes are welcome when they fix a bug, but please
say so in the pull request and keep the previous behaviour available behind a parameter when it is
needed to reproduce the paper.

## Style

Code is formatted and linted with ruff (configuration in `pyproject.toml`). Public functions and
classes carry type hints and docstrings. Comments explain what the code does and why, not how it
came to be.
