"""Figure 3 (one column): dealing inside other schedules.

    python -m repro.figures.fig3_forest [--data JSON | --from-runs] [--out STEM]

A forest plot of dealt minus usual split (nDCG@10 points, 95% CI), one row per
schedule and the part of it that is dealt, two small multiples on one shared
scale (BM25 | Qwen3-Embedding-4B, the dense retriever). Rows are grouped by whether a group's quota can cut a
top-ten passage: where it can, on a tinted band (DealRank's own hands as the
reference, TourRank with contiguous groups at every stage, BlitzRank
``w = 20``'s input); where it cannot (TourRank's first stage, BlitzRank
``w = 10``'s input, TDPart's tail windows, ListK's bins, JointRank's design).

Marks are coloured by the paired 95% interval: orange where dealing gains,
grey where the interval crosses zero, dark where dealing loses. A
significant change in calls per query is written in italics under its row.
A cell set to ``null`` is drawn as "not run".

The input JSON (``data/fig3.json`` holds the paper's; ``--from-runs`` computes
it with :func:`repro.analysis.splits.figure3_data`) has ``rows``, each with
``key``, ``name``, ``part``, ``group`` (``binds`` or ``free``), ``reference``
and, per retriever, ``bm25`` / ``dense`` and ``calls_bm25`` / ``calls_dense``
as ``{diff, lo, hi}``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from repro import records
from repro.figures import style as fs

STEM = "fig3_forest"
DEFAULT_DATA = fs.DATA / "fig3.json"


def data_from_runs() -> dict[str, Any]:
    from repro.analysis.splits import figure3_data

    return drawn(figure3_data())


def drawn(data: dict[str, Any]) -> dict[str, Any]:
    """``data`` with only the fields the figure draws: each cell's difference and interval."""

    def cell(c: dict[str, Any] | None) -> dict[str, Any] | None:
        return None if c is None else {k: c[k] for k in ("diff", "lo", "hi")}

    cells = ("bm25", "dense", "calls_bm25", "calls_dense")
    rows = []
    for r in data["rows"]:
        rows.append({**{k: v for k, v in r.items() if k not in cells}, **{k: cell(r.get(k)) for k in cells}})
    return {"rows": rows}


def draw(D: dict[str, Any]) -> plt.Figure:
    """Draw Figure 3 from its input data."""
    ROWS = D["rows"]
    POOLS = (("bm25", "BM25"), ("dense", "Qwen3-Embedding-4B"))
    GROUPS = (("binds", "Quota can bind"), ("free", "No quota can bind"))

    FW = fs.COLWIDTH
    X0, X1 = -5.0, 7.2  # shared x range (points of nDCG@10)
    TICKS = [-4, -2, 0, 2, 4, 6]
    PITCH, HEAD = 0.116, 0.14  # row pitch, group-header pitch (in)
    SUB = 0.092  # the calls line under a row
    TOP, FOOT = 0.0, 0.19  # the first header line also carries the panel titles; axis below
    n_rows = len(ROWS)

    def sig(c):
        return c and (c["lo"] > 0 or c["hi"] < 0) and abs(c["diff"]) >= 0.1

    def has_calls(r):
        return any(sig(r.get(f"calls_{p}")) for p, _ in POOLS)

    n_sub = sum(1 for r in ROWS if has_calls(r))
    groups = [g for g, _ in GROUPS if any(r["group"] == g for r in ROWS)]
    FH = TOP + len(groups) * HEAD + n_rows * PITCH + n_sub * SUB + FOOT
    LAB_W = 1.36  # label column
    GAP = 0.12  # between the two panels
    PW = (FW - LAB_W - GAP - 0.04) / 2  # panel width

    fig = plt.figure(figsize=(FW, FH))
    ov = fs.inch_axes(fig)  # overlay in inches for labels, bands and rules
    ov.set_zorder(0)
    axes = []
    for i in range(2):
        x = LAB_W + i * (PW + GAP)
        ytop = FH - TOP - HEAD + 0.012  # the plot starts under the first header line (panel titles)
        a = fig.add_axes([x / FW, FOOT / FH, PW / FW, (ytop - FOOT) / FH])
        a.set_xlim(X0, X1)
        a.set_ylim(FOOT, ytop)  # y in inches from the bottom of the canvas
        a.set_zorder(2)
        a.patch.set_visible(False)
        a.set_yticks([])
        for sp in ("left", "right", "top"):
            a.spines[sp].set_visible(False)
        a.spines["bottom"].set_color(fs.RULE)
        a.set_xticks(TICKS)
        a.set_xticklabels([f"{t:+d}".replace("-", "−") if t else "0" for t in TICKS])
        a.tick_params(axis="x", length=2.5, pad=1.5)
        a.axvline(0, color="#9a988f", lw=0.6, zorder=1)
        axes.append(a)

    # ---- row positions ------------------------------------------------------------------
    y = FH - TOP
    ys, heads = {}, {}
    for g in groups:
        heads[g] = y - HEAD / 2
        y -= HEAD
        for r in ROWS:
            if r["group"] == g:
                ys[r["key"]] = y - PITCH / 2
                y -= PITCH
                if has_calls(r):
                    y -= SUB
    assert abs(y - FOOT) < 1e-9

    # band behind the rows whose quota can bind (the one tinted region)
    if "binds" in groups:
        top = heads["binds"] - HEAD / 2 + 0.012  # the band holds the rows; its header sits on it
        lowest = min(ys[r["key"]] - (SUB if has_calls(r) else 0) for r in ROWS if r["group"] == "binds")
        bot = lowest - PITCH / 2 - 0.008
        fs.pill(ov, 0.0, bot, FW, top - bot, fs.DEAL_TINT, r=0.04, z=0.2)

    # group headers
    # (the band alone separates the groups: a rule would cut its rounded corners)
    for g, label in GROUPS:
        if g not in heads:
            continue
        ov.text(0.06, heads[g], label, ha="left", va="center_baseline", fontsize=fs.FS, fontweight="bold", color=fs.INK)

    # panel titles
    for a, (_pool, title) in zip(axes, POOLS):
        bb = a.get_position()
        tt = ov.text(
            (bb.x0 + bb.width / 2) * FW,
            heads[groups[0]],
            title,
            ha="center",
            va="center_baseline",
            fontsize=fs.FS,
            color=fs.INK,
            fontweight="bold",
        )
        # a title wider than its panel (the dense retriever's name) ends flush with the panel's right edge
        fig.canvas.draw()
        wd = tt.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
        if wd > bb.width * FW:
            tt.set_x((bb.x0 + bb.width) * FW)
            tt.set_ha("right")

    # ---- rows -----------------------------------------------------------------------------
    GAIN, NONE, LOSS = fs.DEAL, fs.GREY, fs.INK2  # by significance of the paired 95% CI
    VAL = dict(fontsize=fs.FS_MIN, color=fs.DEAL_DARK, fontweight="bold", va="center_baseline", zorder=5)
    UNIT = (X1 - X0) / PW  # data units per inch
    for r in ROWS:
        yc = ys[r["key"]]
        ov.text(
            0.13,
            yc,
            f"{r['name']}, {r['part']}",
            ha="left",
            va="center_baseline",
            fontsize=fs.FS,
            color=fs.INK,
            fontweight="bold" if r.get("reference") else "normal",
        )
        for a, (pool, _) in zip(axes, POOLS):
            cell = r.get(pool)
            if cell is None:  # not run (or not complete): no mark, just say so
                a.text(
                    1.0,
                    yc,
                    "not run",
                    ha="left",
                    va="center_baseline",
                    fontsize=fs.FS_MIN,
                    color=fs.MUTED,
                    fontstyle="italic",
                )
                continue
            c = GAIN if cell["lo"] > 0 else LOSS if cell["hi"] < 0 else NONE
            assert cell["lo"] > X0 and cell["hi"] < X1, (r["key"], pool, cell)  # no interval is clipped
            a.plot([cell["lo"], cell["hi"]], [yc, yc], color=c, lw=1.3, solid_capstyle="round", zorder=3)
            ring = "white"
            a.plot(cell["diff"], yc, "o", ms=5.6, mfc=ring, mec=ring, mew=0, zorder=3.9)
            a.plot(cell["diff"], yc, "o", ms=4.2, mfc=c, mec=c, mew=0, zorder=4)
            if r.get("reference") or r["key"] == "tourrank_all":
                # the headline gain, beside the interval: right of it where it fits, else left
                t = a.text(cell["hi"] + 0.07 * UNIT, yc, f"{cell['diff']:+.1f}", ha="left", **VAL)
                fig.canvas.draw()
                bb = t.get_window_extent(fig.canvas.get_renderer())
                if a.transData.inverted().transform((bb.x1, 0))[0] > X1:
                    t.set_x(cell["lo"] - 0.07 * UNIT)
                    t.set_ha("right")
            calls = r.get(f"calls_{pool}")
            if sig(calls):
                # a significant change in calls per query, on its own line under the row
                d = calls["diff"]
                a.text(
                    X1 - 0.1,
                    yc - (PITCH + SUB) / 2 + 0.02,
                    f"{abs(d):.2f} {'fewer' if d < 0 else 'more'} calls",
                    ha="right",
                    va="center_baseline",
                    fontsize=fs.FS_MIN,
                    fontstyle="italic",
                    color=fs.DEAL_DARK if d < 0 else fs.INK2,
                    zorder=5,
                    # too wide to start right of zero: a patch in the row's background hides the zero line
                    bbox=dict(
                        boxstyle="square,pad=0.04", ec="none", fc=fs.DEAL_TINT if r["group"] == "binds" else fs.PAPER
                    ),
                )

    # axis name on the tick-label line, in the label column
    fig.canvas.draw()
    tl = axes[0].get_xticklabels()[0].get_window_extent(fig.canvas.get_renderer())
    ov.text(
        0.06,
        (tl.y0 + tl.y1) / 2 / fig.dpi,
        "Δ nDCG@10, dealt better →",
        ha="left",
        va="center",
        fontsize=fs.FS_MIN,
        color=fs.INK,
    )

    fs.check_layout(fig)
    return fig


def load_data(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--data", type=Path, default=DEFAULT_DATA, help="JSON of the figure's inputs")
    source.add_argument("--from-runs", action="store_true", help="compute the inputs from the run records")
    parser.add_argument("--write-data", type=Path, help="also write the inputs used to this JSON file")
    parser.add_argument("--out", type=Path, default=records.OUT / "figures" / STEM, help="output path stem")
    args = parser.parse_args(argv)
    data = data_from_runs() if args.from_runs else load_data(args.data)
    if args.write_data:
        args.write_data.parent.mkdir(parents=True, exist_ok=True)
        args.write_data.write_text(json.dumps(data, indent=1) + "\n")
    fs.apply()
    fig = draw(data)
    print("saved", fs.save(fig, args.out))


if __name__ == "__main__":
    main()
