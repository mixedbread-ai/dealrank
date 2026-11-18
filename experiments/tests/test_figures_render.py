"""Each figure renders from its shipped input JSON."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("numpy")


@pytest.mark.parametrize("module", ["fig1_hero", "fig2_split", "fig3_forest"])
def test_renders_from_json(module: str, tmp_path: Path) -> None:
    fig = importlib.import_module(f"repro.figures.{module}")
    fig.main(["--data", str(fig.DEFAULT_DATA), "--out", str(tmp_path / module)])
    assert (tmp_path / f"{module}.pdf").stat().st_size > 10_000
    assert (tmp_path / f"{module}.png").stat().st_size > 10_000


@pytest.mark.parametrize("module", ["fig1_hero", "fig2_split", "fig3_forest"])
def test_shipped_data_holds_only_what_the_figure_draws(module: str) -> None:
    # The shipped inputs are what ``--from-runs --write-data`` writes: only the fields the figure draws.
    fig = importlib.import_module(f"repro.figures.{module}")
    data = json.loads(Path(fig.DEFAULT_DATA).read_text())
    assert data and fig.drawn(data) == data
