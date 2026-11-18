"""Figure 1 (full text width): dealing against chunking, and latency per query.

    python -m repro.figures.fig1_hero [--data JSON | --from-runs] [--out STEM]

(a) The same 100 candidates split into five calls of 20; the retriever's top
    20 are dark. Dealt hands each hold four of them, the quota that advances,
    so DealRank takes two rounds on every query. Contiguous chunks put all
    twenty in one call; the sixteen beyond the quota are drawn lighter and
    struck through, and one stroke brackets them and leads to BlitzRank's
    certification call, dashed and hatched because only ``certify_share`` of
    queries make it.
(b) Latency per query: bar length is ``path_s`` (the median critical path), one block
    per sequential round (hatched: certification), with seconds, mean rounds
    and nDCG@10 (the mean over the LLMs of Table 2, with each method's
    paired difference from DealRank) as columns; a hatched block carries its
    row's ``certify_share``.

Every number drawn comes from the input JSON (``data/fig1.json`` holds the
paper's; ``--from-runs`` computes it with
:func:`repro.analysis.latency.figure1_data`). Its fields:

* ``setup``: ``retriever``, ``hands``, ``window``, ``quota``, ``head`` (how
  many top ranks are drawn dark) and ``certify_share`` (0-1) for panel (a).
* ``panel_b.rows``: one bar per method, top to bottom (3 to 6 fit), each with
  ``key`` (panel (a) prints the rounds of ``dealrank`` and
  ``blitzrank_concurrent``), ``label``, ``role`` (``ours``, ``blitz`` or
  ``other``: bold label and grouping), ``split`` (``dealt``, ``chunks`` or
  ``window``: orange, blue or grey), ``path_s``, ``rounds``, ``ndcg10``,
  optional ``segments`` (``{s, kind}`` per round, ``kind`` = ``round`` or
  ``certify``, summing to ``path_s``), ``quiet`` (a lighter context row),
  ``certify_share``, ``ndcg_minus_dealrank`` (difference and 95% interval)
  and ``rounds_max``.
* ``panel_b.unit_note``: the key's label for a solid block; ``panel_b.ndcg_note``:
  the line under the nDCG@10 header.

Running text uses proportional figures, number columns tabular ones (see
:func:`repro.figures.style.proportional_figures`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from repro import records
from repro.figures import style as fs

STEM = "fig1_hero"
DEFAULT_DATA = fs.DATA / "fig1.json"


def data_from_runs() -> dict[str, Any]:
    from repro.analysis.latency import figure1_data

    return drawn(figure1_data())


#: Row fields the analysis computes that the figure does not draw.
UNDRAWN_ROW_FIELDS = ("wall_s",)


def drawn(data: dict[str, Any]) -> dict[str, Any]:
    """``data`` with only the fields the figure draws."""
    rows = [{k: v for k, v in r.items() if k not in UNDRAWN_ROW_FIELDS} for r in data["panel_b"]["rows"]]
    return {**data, "panel_b": {**data["panel_b"], "rows": rows}}


def draw(DATA: dict[str, Any]) -> plt.Figure:
    """Draw Figure 1 from its input data."""
    SET = DATA["setup"]
    NH, WIN, QUOTA, HEAD = SET["hands"], SET["window"], SET["quota"], SET["head"]
    NCAND = NH * WIN
    RETR = SET["retriever"]
    CERT = SET["certify_share"]
    PB = DATA["panel_b"]
    ROWS = PB["rows"]
    BY = {r["key"]: r for r in ROWS}

    DH = 0.136  # height trimmed from the canvas by the tight hand pitch
    W, H = fs.TEXTWIDTH, 2.3 - DH
    fig = plt.figure(figsize=(W, H))
    ax = fs.inch_axes(fig)
    rough = fs.Rough(ax, seed=11)
    STRONG, WEAK = fs.RANK_RAMP[0], fs.RANK_RAMP[3]
    DEAL_PALE, CHUNK_PALE = "#f6c4ab", "#a9c4e8"  # an advanced passage outside ranks 1-HEAD
    FS_A = 8.0  # labels the reader must read: Biolinum runs small, so 8 pt, not 7.5
    CUT = fs.RANK_RAMP[1]  # a top card the quota cuts: lighter, struck through

    def text(x, y, s, size=fs.FS, color=fs.INK, weight="normal", ha="left", va="baseline", tabular=False, **kw):
        """Running text gets proportional figures (no gap around a 1); number columns keep tabular ones."""
        t = ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, **kw)
        if not tabular:
            slack = fs.proportional_figures(t)  # inches the drawn text is narrower than measured
            if not kw.get("rotation"):
                t.set_x(x + slack * {"left": 0, "center": 0.5, "right": 1}[ha])
        return t

    def width(t):
        return t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi - getattr(t, "pnum_slack", 0.0)

    def run(x, y, parts, **kw):
        """Consecutive text pieces on one baseline; returns the end x (inches)."""
        fig.canvas.draw()
        for s, k in parts:
            x += width(text(x, y, s, **{**kw, **k}))
        return x

    def fmt(v, nd=1):
        return f"{v:.{nd}f}"

    def rounds_of(row):
        """(fewest, most) rounds the schedule drawn in (a) can take: one per block, the certification
        block being conditional; without blocks, the rounded mean."""
        segs = row.get("segments")
        if not segs:
            return (round(row["rounds"]),) * 2
        return sum(s["kind"] != "certify" for s in segs), len(segs)

    # =============================================================================
    # (a) dealing vs chunking
    # =============================================================================
    AX1 = 4.50  # right edge of panel (a)
    LANE1_TOP = 2.085 - DH
    LANE_BOT = 0.03
    PITCH_Y = 0.122  # hand to hand
    BOX_H = 0.104
    PAD_BOT = 0.055  # last hand to the lane border
    HEAD_Y = 0.140  # lane title baseline below the lane top
    OFF1 = 0.245 + BOX_H / 2  # lane top to the first hand's centre (DealRank)
    OFF2 = 0.324 + BOX_H / 2  # BlitzRank: one more line, for what its first chunk cuts
    LANE1_H = OFF1 + (NH - 1) * PITCH_Y + BOX_H / 2 + PAD_BOT
    LANE2_H = OFF2 + (NH - 1) * PITCH_Y + BOX_H / 2 + PAD_BOT
    LANE2_TOP = LANE_BOT + LANE2_H
    LANE_GAP = LANE1_TOP - LANE1_H - LANE2_TOP
    assert 0.055 <= LANE_GAP <= 0.09, LANE_GAP
    PIP_W, PIP_H, PIP_DX = 0.043, 0.072, 0.0605
    X_LAB = 0.555  # right edge of the rank labels
    X_HAND = 0.775  # left edge of the hand boxes
    HAND_W = WIN * PIP_DX + 0.03
    X_FIN = X_HAND + HAND_W + 0.16  # the passages that advance
    X_BOX = X_FIN + QUOTA * PIP_DX + 0.185
    GB_W, GB_H = 0.36, 0.40  # final / merge / certify box
    X_CERT = X_BOX + GB_W + 0.12
    X_OUT = X_CERT + GB_W + 0.125

    def card_x(j):
        """Left edge of card j (0-based) in a hand box."""
        return X_HAND + 0.015 + j * PIP_DX + (PIP_DX - PIP_W) / 2

    def lane(top, height, off, hues, tint, dark, hands, labels):
        """One schedule: five calls of `WIN` -> `QUOTA` advance from each -> one call."""
        rough.rrect(
            0.012, top - height, AX1 - 0.024, height, r=0.09, fc=tint, ec=dark, lw=0.8, amp=0.005, z=1, wavelength=1.1
        )
        ys = [top - off - j * PITCH_Y for j in range(NH)]
        adv = []  # colours of the passages each hand advances
        for yc, ranks, lab in zip(ys, hands, labels):
            text(X_LAB, yc, lab, size=FS_A, color=fs.INK2, ha="right", va="center_baseline")
            rough.rrect(
                X_HAND,
                yc - BOX_H / 2,
                HAND_W,
                BOX_H,
                r=0.03,
                fc="white",
                ec=fs.INK2,
                lw=0.65,
                amp=0.003,
                z=2,
                wavelength=0.6,
            )
            seen, cut_x = 0, []
            for j, r in enumerate(ranks):
                if r <= HEAD:
                    seen += 1
                    if seen > QUOTA:  # a top-HEAD card beyond the quota: faded and struck through, i.e. cut
                        fs.pill(ax, card_x(j), yc - PIP_H / 2, PIP_W, PIP_H, CUT, r=0.011, z=3)
                        cut_x.append(card_x(j))
                        continue
                fs.pill(ax, card_x(j), yc - PIP_H / 2, PIP_W, PIP_H, STRONG if r <= HEAD else WEAK, r=0.011, z=3)
            if cut_x:
                ax.plot(
                    [cut_x[0] - 0.006, cut_x[-1] + PIP_W + 0.006],
                    [yc, yc],
                    color=fs.CHUNK,
                    lw=1.2,
                    solid_capstyle="round",
                    zorder=3.5,
                )
            rough.arrow((X_HAND + HAND_W + 0.03, yc), (X_FIN - 0.03, yc), color=fs.INK2, lw=0.75, head=0.04, amp=0)
            n_top = min(QUOTA, sum(1 for r in ranks if r <= HEAD))  # top-HEAD passages that advance
            adv.append([hues[0]] * n_top + [hues[1]] * (QUOTA - n_top))
            for j in range(QUOTA):
                fs.pill(ax, X_FIN + j * PIP_DX, yc - PIP_H / 2, PIP_W, PIP_H, adv[-1][j], r=0.011, z=3)
        # round-1 bracket
        xb = X_HAND - 0.058
        rough.line(
            [
                (xb + 0.025, ys[0] + BOX_H / 2),
                (xb, ys[0] + BOX_H / 2 - 0.02),
                (xb, ys[-1] - BOX_H / 2 + 0.02),
                (xb + 0.025, ys[-1] - BOX_H / 2),
            ],
            color=dark,
            lw=0.8,
            amp=0.002,
        )
        text(xb - 0.060, ys[NH // 2], "round 1", size=FS_A, color=fs.INK2, ha="center", va="center", rotation=90)
        # the one call that ranks the passages that advanced
        by = ys[0] + BOX_H / 2 - GB_H
        rough.rrect(X_BOX, by, GB_W, GB_H, r=0.05, fc="white", ec=fs.INK2, lw=0.75, amp=0.003, z=2)
        gx0 = X_BOX + (GB_W - QUOTA * 0.072 + (0.072 - PIP_W)) / 2
        gys = [ys[0] - i * (GB_H - 0.124) / (NH - 1) for i in range(NH)]
        for yc, gy, cols in zip(ys, gys, adv):
            for j in range(QUOTA):
                fs.pill(ax, gx0 + j * 0.072, gy - 0.026, PIP_W, 0.052, cols[j], r=0.011, z=3)
            ax.plot(
                [X_FIN + QUOTA * PIP_DX + 0.012, X_BOX - 0.02],
                [yc, gy],
                color=cols[0],
                lw=0.7,
                zorder=1.5,
                solid_capstyle="round",
            )
        return ys, by

    LANE_NAMES = ("DealRank deals", "BlitzRank on chunks")
    X_SUB = 0.0

    def lane_head(top, name, claim):
        """Bold lane name, then its one-line claim; the claims of both lanes start at X_SUB."""
        nonlocal X_SUB
        names = [text(0.09, top - HEAD_Y, n, weight="bold", size=fs.FS_HEAD, visible=(n == name)) for n in LANE_NAMES]
        fig.canvas.draw()
        X_SUB = 0.09 + max(width(t) for t in names) + 0.10
        for t in names:
            if not t.get_visible():
                t.remove()
        text(X_SUB, top - HEAD_Y, claim)

    def box_label(x, by, name, sub, ha="center"):
        """Stage name and its round, under a call box."""
        text(x, by - 0.118, name, size=FS_A, ha=ha)
        text(x, by - 0.236, sub, size=FS_A, color=fs.INK2, ha=ha)

    def outcome(ym, big, *subs, x=None):
        x = X_OUT if x is None else x
        text(x, ym + 0.03, big, size=10, weight="bold", va="center_baseline")
        for i, sub in enumerate(subs):
            text(x, ym - 0.115 - 0.115 * i, sub, size=FS_A, color=fs.INK2, va="center_baseline")

    # ---- lane 1: DealRank -------------------------------------------------------
    deal_hands = [[i + 1 + NH * j for j in range(WIN)] for i in range(NH)]
    deal_labels = [", ".join(str(h[j]) for j in range(3)) + ", …" for h in deal_hands]
    _, by1 = lane(LANE1_TOP, LANE1_H, OFF1, (fs.DEAL, DEAL_PALE), fs.DEAL_TINT, fs.DEAL_DARK, deal_hands, deal_labels)
    per_hand = sum(1 for r in deal_hands[0] if r <= HEAD)
    assert all(sum(1 for r in h if r <= HEAD) == per_hand for h in deal_hands) and per_hand <= QUOTA
    lane_head(LANE1_TOP, "DealRank deals", f"every hand holds {per_hand} of ranks 1–{HEAD}, so all {HEAD} can advance")
    box_label(X_BOX + GB_W / 2, by1, "final call", "round 2")
    ym1 = by1 + GB_H / 2
    X_OUT1 = X_BOX + GB_W + 0.30  # DealRank ends after round 2: a short arrow to its outcome
    rough.arrow((X_BOX + GB_W + 0.04, ym1), (X_OUT1 - 0.045, ym1), color=fs.INK2, lw=0.85, amp=0)
    r_lo, r_hi = rounds_of(BY["dealrank"])
    assert r_lo == r_hi
    outcome(ym1, f"{r_hi} rounds", "on every query", x=X_OUT1)

    # ---- lane 2: BlitzRank's contiguous chunks ----------------------------------
    chunk_hands = [[i * WIN + 1 + j for j in range(WIN)] for i in range(NH)]
    chunk_labels = [f"{h[0]}–{h[-1]}" for h in chunk_hands]
    ys2, by2 = lane(
        LANE2_TOP, LANE2_H, OFF2, (fs.CHUNK, CHUNK_PALE), fs.CHUNK_TINT, fs.CHUNK, chunk_hands, chunk_labels
    )
    in_first = sum(1 for r in chunk_hands[0] if r <= HEAD)
    over = in_first - QUOTA
    lane_head(LANE2_TOP, "BlitzRank on chunks", f"all of ranks 1–{HEAD} land in one chunk")
    box_label(X_BOX + GB_W / 2, by2, "merge", "round 2")
    ym2 = by2 + GB_H / 2
    rough.arrow((X_BOX + GB_W + 0.03, ym2), (X_CERT - 0.03, ym2), color=fs.INK2, lw=0.85, amp=0)
    rough.rrect(
        X_CERT, by2, GB_W, GB_H, r=0.05, fc="white", ec=fs.INK2, lw=0.75, amp=0.003, z=2, dashes=(3.2, 2.2)
    )  # dashed: a call made only on some queries
    rough.hachure(X_CERT + 0.02, by2 + 0.02, GB_W - 0.04, GB_H - 0.04, r=0.035, color=fs.CHUNK, lw=0.7, gap=0.05, z=2.5)
    c_lo, c_hi = rounds_of(BY["blitzrank_concurrent"])
    c_max = BY["blitzrank_concurrent"].get("rounds_max", c_hi)  # over all queries, not just the timed ones
    box_label(X_CERT + GB_W / 2, by2, "certify", f"round {c_hi}")
    rough.arrow((X_CERT + GB_W + 0.03, ym2), (X_OUT - 0.04, ym2), color=fs.INK2, lw=0.85, amp=0)
    # a 100-candidate query takes the certification round on certify_share of queries
    outcome(ym2, f"{c_lo}–{c_max} rounds" if c_max > c_lo else f"{c_max} rounds", f"3 on {100 * CERT:.0f}%")
    # what chunk 1 holds back goes to the certification call: ONE stroke brackets the cards beyond the
    # quota (the count sits on it), steps up clear of the merge box and turns down into the dashed box
    top2 = ys2[0] + BOX_H / 2
    x0c, x1c = card_x(QUOTA), card_x(WIN - 1) + PIP_W
    yk = top2 + 0.030
    text((x0c + x1c) / 2, yk + 0.037, f"the quota cuts {over}", ha="center")  # on its bracket
    ya = yk + 0.056
    xc, tip = X_CERT + GB_W / 2, by2 + GB_H + 0.012
    RISE, R_EL = 0.10, 0.05
    u = np.linspace(0, 1, 14)
    turn = np.linspace(0, np.pi / 2, 14)
    stroke = (
        [(x0c, yk - 0.017), (x0c + 0.012, yk), (x1c, yk)]
        + [(x1c + 0.03 + RISE * t, yk + (ya - yk) * t * t * (3 - 2 * t)) for t in u]
        + [(xc - R_EL + R_EL * np.sin(a), ya - R_EL * (1 - np.cos(a))) for a in turn]
        + [(xc, tip)]
    )
    rough.polyarrow(stroke, color=fs.CHUNK, lw=0.85, head=0.04)
    ax.plot([x1c - 0.012, x1c], [yk, yk - 0.017], color=fs.CHUNK, lw=0.85, solid_capstyle="round", zorder=4)

    # ---- panel title ------------------------------------------------------------
    TY = 2.185 - DH
    text(0.0, TY, "(a)", weight="bold", size=fs.FS_HEAD)
    x = run(0.22, TY, [(f"The {RETR} top {NCAND} in {NH} calls of {WIN}, each advancing a quota of {QUOTA}      ", {})])
    fs.pill(ax, x, TY - 0.005, PIP_W, PIP_H, STRONG, r=0.011)
    run(x + 0.065, TY, [(f"retriever ranks 1–{HEAD}", {})])

    # =============================================================================
    # (b) latency (median critical path), one block per round
    # =============================================================================
    BX0 = 4.70
    X_NDCG, X_RND, X_SEC = W - 0.01, W - 0.57, W - 0.98
    smax = max(r["path_s"] for r in ROWS)
    # the time axis ends on a labelled tick at or beyond the longest bar (2 to 4 intervals)
    STEP, AMAX = min(
        ((st, st * -(-smax // st)) for st in (1, 2, 5, 10, 20, 50, 4, 3, 30) if 2 <= -(-smax // st) <= 4),
        key=lambda c: c[1],
    )
    BAR_MAX = X_SEC - 0.34 - BX0
    SC = BAR_MAX / AMAX
    BAR_H, GAP = 0.112, 0.012
    # bar colours by split, as in (a): (full, quiet)
    FILL = {"dealt": (fs.DEAL, "#f2a07e"), "chunks": (fs.CHUNK, "#7fa6da"), "window": (fs.GREY, "#cdcbc3")}
    QUIET_INK = "#2f2e2b"
    Y_TOP, Y_FOOT = LANE1_TOP - 0.085, 0.185  # room for the source line under the headers
    pitch = (Y_TOP - Y_FOOT) / len(ROWS)
    ROW_INK = 0.245  # cap top of the label to the bottom of the bar
    SEC_HEAD = "median (s)"  # what path_s is: the statistic the bar and its number show

    text(BX0 - 0.14, TY, "(b)", weight="bold", size=fs.FS_HEAD)
    text(BX0 + 0.08, TY, "Latency")
    for xcol, head in ((X_SEC, SEC_HEAD), (X_RND, "rounds"), (X_NDCG, "nDCG@10")):
        text(xcol, TY, head, size=FS_A, color=fs.INK2, ha="right")
    TY2 = TY - 0.112  # where each column's numbers come from
    text(X_SEC, TY2, "timed on 4 sets", size=fs.FS_MIN, color=fs.MUTED, ha="right", style="italic")
    text(X_NDCG, TY2, PB["ndcg_note"], size=fs.FS_MIN, color=fs.MUTED, ha="right", style="italic")

    hatch_named = False
    for i, r in enumerate(ROWS):
        y0 = Y_TOP - i * pitch  # top rule of the row; none between two rows of one method
        if not (i and r["role"] == "blitz" and ROWS[i - 1]["role"] == "blitz"):
            ax.plot([BX0, W - 0.005], [y0, y0], color=fs.GRID, lw=0.6, zorder=1)
        yb = y0 - (pitch + ROW_INK) / 2  # bar bottom: label + bar sit mid-row
        yl = yb + BAR_H + 0.05  # label baseline
        quiet = bool(r.get("quiet"))
        ink = QUIET_INK if quiet else fs.INK
        text(BX0, yl, r["label"], size=8.5, weight="bold" if r["role"] == "ours" else "normal", color=ink)
        segs = r.get("segments")
        if segs:
            assert abs(sum(s["s"] for s in segs) - r["path_s"]) < 0.05, r["key"]
        else:
            n = max(1, round(r["rounds"]))
            segs = [{"s": r["path_s"] / n, "kind": "round"} for _ in range(n)]
        fill = FILL[r["split"]][1 if quiet else 0]
        yv = yb + BAR_H / 2
        x = BX0
        for s in segs:
            w = s["s"] * SC
            if s["kind"] == "certify":
                fs.pill(ax, x, yb, w - GAP, BAR_H, "white", r=0.02, z=3, ec=fill, lw=0.7)
                rough.hachure(x, yb, w - GAP, BAR_H, r=0.02, color=fill, lw=0.7, gap=0.036, z=3.5)
                share = r.get("certify_share")
                lab = (
                    f"certify {100 * share:.0f}%" if share is not None else "certify" if not hatch_named else None
                )  # name the hatched block where it first appears
                if lab:
                    text(x + w + 0.03, yv, lab, size=FS_A, color=fs.INK2, va="center_baseline")
                hatch_named = True
            else:
                fs.pill(ax, x, yb, w - GAP, BAR_H, fill, r=0.02, z=3)
            x += w
        dci = r.get("ndcg_minus_dealrank")  # difference to DealRank and its 95% CI, on the label line
        if dci:

            def sg(v):
                return f"{v:+.1f}".replace("-", "\u2212")

            text(
                X_NDCG,
                yl,
                f"{sg(dci[0])} [{sg(dci[1])}, {sg(dci[2])}]",
                size=fs.FS_MIN,
                ha="right",
                color=fs.MUTED,
                tabular=True,
            )
        for xcol, key in ((X_SEC, "path_s"), (X_RND, "rounds"), (X_NDCG, "ndcg10")):
            text(
                xcol,
                yv,
                fmt(r[key], 1),
                size=8.5,
                ha="right",
                va="center_baseline",
                color=ink,
                tabular=True,
                weight="bold" if r["role"] == "ours" and key == "path_s" else "normal",
            )

    # time axis under the bars
    ax.plot([BX0, W - 0.005], [Y_FOOT, Y_FOOT], color=fs.GRID, lw=0.6, zorder=1)
    ax.plot([BX0, BX0 + BAR_MAX], [Y_FOOT, Y_FOOT], color=fs.GREY, lw=0.7, zorder=1.2)
    t = 0
    while t <= AMAX:
        ax.plot([BX0 + t * SC] * 2, [Y_FOOT, Y_FOOT - 0.028], color=fs.GREY, lw=0.7)
        lab = text(BX0 + t * SC, 0.04, f"{t:g}", size=fs.FS_MIN, color=fs.MUTED, ha="center")
        t += STEP
    fig.canvas.draw()
    text(BX0 + AMAX * SC + width(lab) / 2 + 0.025, 0.04, "s", size=fs.FS_MIN, color=fs.MUTED)
    # key, right of the axis: a solid block is `unit_note`; a hatched one is a round only some queries take
    KEY_W, KEY_H, KEY_Y = 0.085, 0.085, 0.04 - 0.012
    key = [("hatch", "certify")] if hatch_named else []
    key.append(("solid", PB["unit_note"]))
    x = W - 0.01
    for kind, lab in key:  # laid out right to left
        t = text(x, 0.04, lab, size=FS_A, color=fs.INK2, ha="right")
        fig.canvas.draw()
        x -= width(t) + 0.035 + KEY_W
        if kind == "hatch":
            fs.pill(ax, x, KEY_Y, KEY_W, KEY_H, "white", r=0.018, z=3, ec=fs.MUTED, lw=0.7)
            rough.hachure(x, KEY_Y, KEY_W, KEY_H, r=0.018, color=fs.MUTED, lw=0.7, gap=0.036, z=3.5)
        else:
            fs.pill(ax, x, KEY_Y, KEY_W, KEY_H, fs.GREY, r=0.018, z=3)
        x -= 0.12

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
