"""Figure 2 (one column): chunks cut more of the top ten as the retriever improves; dealing cuts about 0.1.

    python -m repro.figures.fig2_split [--data JSON | --from-runs] [--out STEM]

* top panel: contiguous chunks (blue), 0 to 6 cuts; bottom panel: dealing
  (orange), 0 to 0.2 cuts, with the random split as a dotted line.
* solid lines: Proposition 1, exact (:mod:`repro.analysis.expected_cuts`),
  drawn against the mean retriever rank the model implies, ``(R + 1) / 2``.
* dashed lines: the Mallows model of retriever ranks with query-varying
  dispersion (:mod:`repro.analysis.mallows`), traced over its mean dispersion,
  ``mallows_curve``. Nothing about the cuts enters it.
* points: how many of the sliding window's top ten (Gemini 3 Flash) the split
  puts over their group's quota of four, at the mean retriever rank of that
  top ten, on the queries with 100 candidates. Hue is the split, fill the
  retriever (hollow BM25, filled dense), size the level (small: one benchmark,
  ``pools``, eighteen per panel; large: pooled with its 95% interval,
  ``points``).

The input JSON (``data/fig2.json`` holds the paper's; ``--from-runs`` computes
it with :func:`repro.analysis.cuts.figure2_data`) has ``config``
(``N, H, w, q, k``), ``random_partition``, ``points`` (``key``, ``split``,
``retriever``, ``value``, ``ci``, ``mean_rank_top10``), ``pools``
(``retriever``, ``benchmark``, ``split``, ``mean_rank_top10``, ``measured``)
and ``mallows_curve`` (``mean_rank_top10``, ``chunk``, ``deal``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from repro import records
from repro.analysis import expected_cuts
from repro.figures import style as fs

STEM = "fig2_split"
DEFAULT_DATA = fs.DATA / "fig2.json"


def data_from_runs() -> dict[str, Any]:
    from repro.analysis.cuts import figure2_data

    return drawn(figure2_data())


def drawn(data: dict[str, Any]) -> dict[str, Any]:
    """``data`` with only the fields the figure draws."""

    def keep(d: dict[str, Any], *keys: str) -> dict[str, Any]:
        return {k: d[k] for k in keys if k in d}

    return {
        "config": data["config"],
        "random_partition": keep(data["random_partition"], "e_cut"),
        "points": [keep(p, "key", "split", "retriever", "value", "ci", "mean_rank_top10") for p in data["points"]],
        "pools": [keep(p, "retriever", "benchmark", "split", "mean_rank_top10", "measured") for p in data["pools"]],
        "mallows_curve": [keep(m, "mean_rank_top10", "chunk", "deal") for m in data["mallows_curve"]],
    }


def draw(D: dict[str, Any]) -> plt.Figure:
    """Draw Figure 2 from its input data."""
    N, H, W_, Q, K = (D["config"][k] for k in ("N", "H", "w", "q", "k"))
    L_RAND = float(expected_cuts.l_rand(H, W_, K, Q))
    RS = np.arange(K, N + 1)
    LD = np.array([float(expected_cuts.l_deal(int(R), H, K, Q)) for R in RS])
    LC = np.array([float(expected_cuts.l_chunk(int(R), H, W_, K, Q)) for R in RS])
    XR = (RS + 1) / 2  # mean rank of a uniform k-subset of the top R
    assert abs(L_RAND - D["random_partition"]["e_cut"]) < 1e-9
    assert np.all(LD <= L_RAND + 1e-12) and np.all(LC >= L_RAND - 1e-12)  # Proposition 1's sandwich
    assert LC[0] == K - Q and LD[H * Q - K] == 0

    pts = {p["key"]: p for p in D["points"]}
    pools = D["pools"]
    mc = sorted(D["mallows_curve"], key=lambda r: r["mean_rank_top10"])
    COL = {"chunk": fs.CHUNK, "deal": fs.DEAL, "random": fs.GREY}

    def tint(c, t):
        """Mix a colour with white (t = share of white)."""
        return mcolors.to_hex([(1 - t) * v + t for v in mcolors.to_rgb(c)])

    FW, FH = fs.COLWIDTH, 1.72
    fig = plt.figure(figsize=(FW, FH))
    L, Rm = 0.27, 0.06  # margins (in)
    AW = FW - L - Rm
    Y_ZO, H_ZO = 0.295, 0.52  # zoom panel
    Y_AX, H_AX = 0.925, 0.50  # main panel; the key sits above it (gutter 0.11 in)
    ax = fig.add_axes([L / FW, Y_AX / FH, AW / FW, H_AX / FH])
    zo = fig.add_axes([L / FW, Y_ZO / FH, AW / FW, H_ZO / FH])
    X0, X1 = 52.3, 3.6
    A = dict(fontsize=fs.FS, color=fs.INK2)
    B = dict(fontsize=fs.FS, color=fs.INK)  # panel titles: regular weight, as the headers of Figs 1 and 3
    ZMAX = 0.2  # top of the zoom: next 0.1 above every dealt/random point
    while any(
        v > ZMAX
        for v in [r["measured"] for r in pools if r["split"] != "chunk"]
        + [p["value"] for p in D["points"] if p["split"] != "chunk"]
    ):
        ZMAX = round(ZMAX + 0.1, 1)
    bg = fs.inch_axes(fig)
    bg.set_zorder(-1)
    for (y0, h0), panel_fc in (((Y_AX, H_AX), fs.CHUNK_TINT), ((Y_ZO, H_ZO), fs.DEAL_TINT)):
        fs.pill(bg, L - 0.03, y0 - 0.025, AW + 0.05, h0 + 0.045, panel_fc, r=0.05, z=0)
    for a in (ax, zo):
        a.set_xlim(X0, X1)
        a.tick_params(length=0)
        for sp in ("left", "bottom"):
            a.spines[sp].set_visible(False)
        a.grid(axis="y", color=fs.GRID, lw=0.5, zorder=0)
        a.axhline(0, color=fs.RULE, lw=0.6, zorder=0.5)  # the baseline is zero itself
    ZLIM = (-0.011, ZMAX + 0.052)  # headroom above ZMAX for the panel title
    STRETCH = 25  # the dealing panel's y axis is this many times finer
    Y_LO = -0.12
    ax.set_ylim(Y_LO, Y_LO + STRETCH * (ZLIM[1] - ZLIM[0]) * H_AX / H_ZO)
    ax.set_yticks([0, 2, 4, 6])
    ax.set_xticks([])
    zo.set_ylim(*ZLIM)
    zt = [round(0.1 * i, 1) for i in range(round(ZMAX * 10) + 1)]
    zo.set_yticks(zt)
    zo.set_yticklabels(["0"] + [f"{t:g}" for t in zt[1:]])
    zo.set_xticks([50, 40, 30, 20, 10])
    fig.text(
        0.02 / FW,
        (FH - 0.012) / FH,
        "Top-10 passages the quota cuts (mean per query)",
        ha="left",
        va="top",
        fontsize=fs.FS,
        color=fs.INK,
    )
    xl = fig.text(
        L / FW,
        0.022 / FH,
        "mean retriever rank of the sliding window’s top 10",
        ha="left",
        va="bottom",
        fontsize=fs.FS,
        color=fs.INK,
    )
    fig.canvas.draw()
    xl_end = xl.get_window_extent(fig.canvas.get_renderer()).x1 / fig.dpi
    fig.text((xl_end + 0.07) / FW, 0.022 / FH, "better →", ha="left", va="bottom", **A)  # beside the label

    # ---- marks --------------------------------------------------------------------
    # Shape carries the split, fill the retriever (hollow = BM25, filled = dense),
    # size the level: small and pale = one benchmark, large = pooled over benchmarks.
    MARK = {"chunk": "o", "deal": "o", "random": "s"}  # one marker: hue and panel carry the split
    SZ = {"o": 1.0, "D": 0.82, "s": 0.9}
    SMALL = {"chunk": tint(fs.CHUNK, 0.22), "deal": tint(fs.DEAL_DARK, 0.12), "random": fs.GREY}
    SZ_SMALL = {"o": 1.0, "D": 0.95, "s": 0.9}
    MS_BIG, MS_SMALL = 5.6, 2.7
    HOLLOW = "bm25"

    def retriever(p):
        """Retriever of a point, read from its label when it has none (else drawn filled)."""
        r = (p.get("retriever") or "").lower()
        if not r:
            lab = (p.get("label") or "").lower()
            r = "bm25" if "bm25" in lab else ""
        return r

    def face(retr, colour):
        return "white" if retr.lower() == HOLLOW else colour

    def big(a, p, ms=MS_BIG):
        """Pooled value: large marker in the full hue on a white ring, with its 95% interval."""
        x, y, c, m = p["mean_rank_top10"], p["value"], COL[p["split"]], MARK[p["split"]]
        if p["split"] == "random":
            c = fs.MUTED  # the measured squares are darker than the model line
        if p.get("ci"):
            a.plot([x, x], p["ci"], color=c, lw=1.1, zorder=5, solid_capstyle="butt")
        a.plot(x, y, m, ms=(ms + 2.2) * SZ[m], mfc="white", mec="white", mew=0, zorder=5.9, clip_on=False)
        a.plot(x, y, m, ms=ms * SZ[m], mfc=face(retriever(p), c), mec=c, mew=1.25, zorder=6, clip_on=False)
        return ms * SZ[m] / 2 + 0.75  # outer radius in points

    def small(a, r):
        m, c = MARK[r["split"]], SMALL[r["split"]]
        a.plot(
            r["mean_rank_top10"],
            r["measured"],
            m,
            ms=(MS_SMALL + 1.0) * SZ_SMALL[m],
            mfc="white",
            mec="white",
            mew=0,
            zorder=3.7,
        )
        a.plot(
            r["mean_rank_top10"],
            r["measured"],
            m,
            ms=MS_SMALL * SZ_SMALL[m],
            mfc=face(r["retriever"], c),
            mec=c,
            mew=0.75,
            zorder=3.8,
            alpha=0.8,
        )

    def tw(t):
        """Width of a text in x data units."""
        fig.canvas.draw()
        bb = t.get_window_extent(fig.canvas.get_renderer())
        inv = t.axes.transData.inverted()
        return abs(inv.transform((bb.x1, 0))[0] - inv.transform((bb.x0, 0))[0])

    def pt_x(a, pts_):
        """Points -> x data units of axes a."""
        return pts_ / 72 / (AW / abs(X1 - X0))

    def pt_y(a, pts_):
        y0, y1 = a.get_ylim()
        return pts_ / 72 / (a.get_position().height * FH / (y1 - y0))

    def pooled_label(a, p, x, y, nd, anchor, r_pt, name=True, ha="right"):
        """'<retriever> pooled <value>' ending (or starting) at x, tied to the pooled marker by a leader.

        The label carries no marker sample, so nothing but data sits in the plot."""
        px, py = p["mean_rank_top10"], p["value"]
        t = a.text(
            x,
            y,
            (p["retriever"] + " " if name else "") + f"$\\mathbf{{{py:.{nd}f}}}$",
            ha=ha,
            va="center",
            fontsize=fs.FS,
            color=fs.INK,
            zorder=7,
        )
        xl, xr = (x, x - tw(t)) if ha == "left" else (x + tw(t), x)
        ax_, ay_ = {"right": (xr - pt_x(a, 1.5), y), "below": (px, y - pt_y(a, 4.2)), "left": (xl + pt_x(a, 1.5), y)}[
            anchor
        ]
        a.annotate(
            "",
            (px, py),
            xytext=(ax_, ay_),
            zorder=5.5,
            arrowprops=dict(arrowstyle="-", color=fs.INK2, lw=0.55, shrinkA=0, shrinkB=r_pt + 1.2),
        )

    mx = [r["mean_rank_top10"] for r in mc]
    DASH = (0, (3.2, 2.2))
    LW, LW_DASH = 1.45, 1.1  # Prop. 1 (solid) and Mallows (dashed)
    # ---- main panel: the whole range ---------------------------------------------
    # the random line is drawn in the zoom only: at this scale it would sit on the dealing line
    # dealing is drawn in its own panel below only: at this scale it would sit on the baseline
    ax.plot(XR, LC, color=fs.CHUNK, lw=LW, zorder=4)
    ax.plot(XR, LD, color=fs.DEAL, lw=1.0, alpha=0.55, zorder=3.6, solid_capstyle="butt")  # dealing, at this scale
    ax.text(16.0, 0.42, "dealing", ha="center", va="bottom", fontsize=fs.FS_MIN, color=fs.DEAL_DARK, zorder=3.6)
    ax.plot(mx, [r["chunk"] for r in mc], color=fs.CHUNK, lw=LW_DASH, ls=DASH, zorder=3.5, dash_capstyle="butt")
    for r in pools:
        if r["split"] == "chunk":
            small(ax, r)
    rad = {}
    for p in D["points"]:
        if p["split"] == "chunk":
            rad[p.get("key")] = big(ax, p)
    pb, pdn = pts["bm25_chunk"], pts["dense_chunk"]
    pooled_label(ax, pb, pb["mean_rank_top10"] + 2.2, pb["value"] + 1.15, 1, "right", rad["bm25_chunk"])
    pooled_label(ax, pdn, pdn["mean_rank_top10"] - 0.9, pdn["value"] + 1.4, 1, "below", rad["dense_chunk"])
    ax.plot([50.9, 48.4], [5.75, 5.75], color=fs.CHUNK, lw=2.4, solid_capstyle="round", zorder=6)  # hue swatch
    ax.text(47.4, 5.75, "Chunks cut more as retrieval improves", ha="left", va="center", **B)
    ax.text(X1 + 0.3, 6.0 - pt_y(ax, 5.6), "max", ha="right", va="center", **A)
    # ---- key, above the plot: lines are models of the true top 10, points are measured ----
    hd = fs.inch_axes(fig)
    hd.set_zorder(5)
    KX, KY = 0.02, FH - 0.21  # inches: left edge (as the title), first row
    PT = 1 / 72

    def twi(t):
        fig.canvas.draw()
        return t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi

    def head(y, bold, rest):
        t = hd.text(KX, y, f"$\\mathbf{{{bold}}}${rest}", ha="left", va="center", fontsize=fs.FS_MIN, color=fs.INK)
        return KX + twi(t)

    def key_row(y, x, items):
        for kind, arg, label in items:
            if kind == "line":
                hd.plot(
                    [x, x + 11.5 * PT],
                    [y, y],
                    color=fs.INK2,
                    lw=arg[1],
                    ls=arg[0],
                    dash_capstyle="butt",
                    solid_capstyle="butt",
                )
                x += 14.5 * PT
            else:  # (split whose shape and hue it shows, retriever whose fill)
                split, retr = arg
                m, c = MARK[split], KEYCOL[split]
                if kind == "big":  # size key: a pooled marker is large
                    hd.plot(x + 3.4 * PT, y, m, ms=MS_BIG * SZ[m], mfc=fs.GREY_LIGHT, mec=c, mew=1.25)
                    x += 9.4 * PT
                else:
                    hd.plot(x + 2.2 * PT, y, m, ms=MS_SMALL * SZ_SMALL[m], mfc=face(retr, c), mec=c, mew=0.8)
                    x += 7.0 * PT
            x += twi(hd.text(x, y, label, ha="left", va="center", fontsize=fs.FS, color=fs.INK2)) + 7.0 * PT
        return x - 7.5 * PT

    KEYCOL = {"chunk": fs.CHUNK, "deal": fs.DEAL, "random": fs.MUTED, "ink": fs.INK2}
    MARK["ink"] = "o"
    SZ.setdefault("o", 1.0)
    KY = FH - 0.205
    xe = key_row(
        KY,
        KX,
        [
            ("line", ("-", LW), "uniform (Prop. 1)"),
            ("line", (DASH, LW_DASH), "Mallows fit"),
            ("dot", ("ink", "bm25"), "BM25"),
            ("dot", ("ink", "dense"), "dense"),
            ("big", ("ink", "bm25"), "pooled"),
        ],
    )
    assert xe < FW - 0.03, xe  # the key stays on the canvas

    # ---- zoom panel: dealing against random -----------------------------------------
    zo.plot(XR, np.full_like(XR, L_RAND), color=fs.MUTED, lw=1.2, ls=(0, (0.1, 2.2)), dash_capstyle="round", zorder=3)
    zo.plot(XR, LD, color=fs.DEAL, lw=LW, zorder=4.2)
    zo.plot(mx, [r["deal"] for r in mc], color=fs.DEAL, lw=LW_DASH, ls=DASH, zorder=4.1, dash_capstyle="butt")
    for r in pools:
        if r["split"] != "chunk" and r["measured"] < ZMAX:
            small(zo, r)
    for p in D["points"]:
        if p["split"] != "chunk" and p["value"] < ZMAX:
            rad[p.get("key")] = big(zo, p)
    zo.text(
        X1 + 0.4,
        L_RAND + pt_y(zo, 1.9),
        f"random split {L_RAND:.3f}",
        ha="right",
        va="baseline",
        fontsize=fs.FS,
        color=fs.INK,
        zorder=2.5,  # no descenders: it fits under the 0.2 gridline;
        bbox=dict(fc=fs.PAPER, ec="none", pad=0.4) if ZMAX > 0.2 else None,
    )  # in a taller zoom it masks that line
    zo.plot([50.9, 48.4], [ZMAX + 0.03] * 2, color=fs.DEAL, lw=2.4, solid_capstyle="round", zorder=6)
    zo.text(47.4, ZMAX + 0.03, f"Dealing cuts about 0.1 (y stretched {STRETCH}×)", ha="left", va="center", **B)

    def side_label(a, p, x, y, nd, side, r_pt):
        """'<retriever> <value>' whose near end sits at x (side: 'left' = the label lies visually left of
        x, toward worse retrieval), tied to the marker by a straight leader from that end."""
        t = a.text(
            x,
            y,
            f"{p['retriever']} $\\mathbf{{{p['value']:.{nd}f}}}$",
            ha="right" if side == "left" else "left",
            va="center",
            fontsize=fs.FS,
            color=fs.INK,
            zorder=7,
        )
        a.annotate(
            "",
            (p["mean_rank_top10"], p["value"]),
            xytext=(x, y),
            zorder=5.5,
            arrowprops=dict(arrowstyle="-", color=fs.INK2, lw=0.55, shrinkA=1.5, shrinkB=r_pt + 1.2),
        )
        return t

    side_label(zo, pts["bm25_deal"], 25.6, 0.028, 2, "right", rad["bm25_deal"])
    if "dense_deal" in pts:
        side_label(zo, pts["dense_deal"], 18.4, 0.121, 2, "right", rad["dense_deal"])

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
