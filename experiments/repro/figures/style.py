"""House style of the paper's figures: palette, type, layout checks and a small hand-drawn kit.

Canvases are built at their printed size (ACM sigconf: column 3.33 in, text
7.0 in) and included at scale 1.0, so every font size below is the printed
size; nothing is smaller than ``FS_MIN`` = 7.5 pt.

Type: Linux Biolinum O, the sans-serif companion of the Linux Libertine
family that the ACM article class uses, bundled in ``fonts/`` under the SIL
Open Font License. Without those files the figures fall back to DejaVu Sans,
and the layout checks only warn, since text widths change.

Colour: two hues plus greys, by role, the same in every figure.

    DEAL   orange  dealing / DealRank
    CHUNK  blue    contiguous chunks / BlitzRank
    greys          everything else (other baselines, random split, structure)

The pair keeps a clear contrast under common colour-vision deficiencies and
in greyscale (144 vs 93 of 255), and every use also carries a second cue
(direct label, dash pattern, marker shape or hatching).

The hand-drawn kit (``Rough``) gives schematic outlines and arrows a slight,
seeded wobble; data marks (bars, lines, points) are never wobbled.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import itertools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch

FONTS = Path(__file__).resolve().parent / "fonts"
DATA = Path(__file__).resolve().parent / "data"

# ---- colour ----------------------------------------------------------------
DEAL = "#eb6834"
DEAL_DARK = "#b9471a"  # outlines on DEAL_TINT
DEAL_TINT = "#fdeee5"
CHUNK = "#1c5cab"
CHUNK_TINT = "#e8f0fa"
CHUNK_MARK = "#4a86d4"  # one step lighter: small chunk marks beside the dark rank cards (grey 125 vs 42 in print)
INK = "#0b0b0b"  # primary text
INK2 = "#3d3c38"  # secondary text, outlines
MUTED = "#6f6d67"  # axis text (>= 5:1 on white)
GREY = "#a9a7a0"  # neutral marks (other baselines)
GREY_LIGHT = "#d6d4cc"
RULE = "#c3c2b7"  # baselines / axes
GRID = "#e6e5df"
PAPER = "#ffffff"
# candidate strength: one grey ramp, retriever ranks 1-20 (darkest) ... 81-100
RANK_RAMP = ["#2b2a27", "#74726c", "#a5a39c", "#cbc9c1", "#e6e4dd"]

COLWIDTH = 3.33
TEXTWIDTH = 7.0

FS = 8.0
FS_MIN = 7.5  # smallest text anywhere
FS_HEAD = 8.5

FONT = "Linux Biolinum O"
FALLBACK = "DejaVu Sans"
#: Whether the house font was found; set by :func:`apply`.
HOUSE_FONT = False
#: Whether Matplotlib can apply OpenType features (3.11 and later, so Python 3.11 and later).
FONT_FEATURES = hasattr(matplotlib.text.Text, "set_fontfeatures")


def apply() -> None:
    """Set the house style for every figure drawn afterwards."""
    global HOUSE_FONT
    files = sorted(FONTS.glob("LinBiolinum_*.otf"))
    for f in files:
        fm.fontManager.addfont(str(f))
    HOUSE_FONT = bool(files) or any(f.name == FONT for f in fm.fontManager.ttflist)
    family = FONT if HOUSE_FONT else FALLBACK
    if not HOUSE_FONT:
        warnings.warn(f"{FONT} not found in {FONTS}; drawing with {FALLBACK}", stacklevel=2)
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": FS,
            "text.color": INK,
            "mathtext.fontset": "custom",
            "mathtext.rm": family,
            "mathtext.it": f"{family}:italic",
            "mathtext.bf": f"{family}:bold",
            "mathtext.cal": f"{family}:italic",
            "mathtext.sf": family,
            "mathtext.tt": family,
            "mathtext.default": "regular",
            "axes.labelsize": FS,
            "axes.labelcolor": INK,
            "axes.edgecolor": RULE,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.facecolor": "none",
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "xtick.labelsize": FS_MIN,
            "ytick.labelsize": FS_MIN,
            "xtick.color": RULE,
            "ytick.color": RULE,
            "xtick.labelcolor": MUTED,
            "ytick.labelcolor": MUTED,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.pad": 2,
            "ytick.major.pad": 2,
            "axes.labelpad": 2,
            "legend.frameon": False,
            "legend.fontsize": FS_MIN,
            "lines.linewidth": 1.5,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "figure.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "pdf.fonttype": 42,
        }
    )


def save(fig: Any, stem: Path | str) -> Path:
    """Write a vector PDF at the canvas size (no tight crop: the canvas is the print size) and a 300-dpi PNG."""
    base = Path(stem)
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{base}.pdf")
    fig.savefig(f"{base}.png", dpi=300)
    return Path(f"{base}.pdf")


def check_layout(fig: Any) -> None:
    """Fail if any text is below ``FS_MIN``, leaves the canvas or overlaps other text.

    With the fallback font, or a Matplotlib that cannot apply font features,
    text widths differ from the paper's, so the checks warn instead of failing.
    """
    try:
        assert min_fontsize(fig) >= FS_MIN, "text below the minimum size"
        assert_text_inside(fig)
        assert_no_text_overlap(fig)
    except AssertionError as err:
        if HOUSE_FONT and FONT_FEATURES:
            raise
        warnings.warn(f"layout check without the paper's font setup: {err}", stacklevel=2)


def proportional_figures(t):
    """Draw text `t` with proportional figures (OpenType pnum) and return how much narrower, in
    inches, it is drawn than Matplotlib measures it: Matplotlib 3.11 applies font features when it
    draws but not when it measures, so right-aligned, centred or chained text must be shifted by
    this slack.  The slack is also stored as t.pnum_slack.  Before Matplotlib 3.11 the text keeps
    tabular figures and the slack is 0."""
    if not FONT_FEATURES:
        t.pnum_slack = 0.0
        return 0.0
    from matplotlib.ft2font import LoadFlags

    t.set_fontfeatures(["pnum"])
    font = fm.get_font(fm.findfont(t.get_fontproperties()))
    font.set_size(t.get_fontsize(), 72)

    def w(features):
        font.set_text(t.get_text(), 0.0, flags=LoadFlags.NO_HINTING, features=features)
        return font.get_width_height()[0] / 64.0 / 72.0

    t.pnum_slack = w(None) - w(["pnum"])
    return t.pnum_slack


def _text_bbox(t, r):
    """Extent of the text itself (not an annotation's leader), as drawn: see proportional_figures."""
    bb = matplotlib.text.Text.get_window_extent(t, r)
    slack = getattr(t, "pnum_slack", 0.0) * t.figure.dpi
    if slack and not t.get_rotation():
        bb = matplotlib.transforms.Bbox.from_extents(bb.x0, bb.y0, bb.x1 - slack, bb.y1)
    return bb


def min_fontsize(fig):
    sizes = [t.get_fontsize() for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]
    return min(sizes)


def assert_no_text_overlap(fig, ignore=()):
    """Raise if two visible texts overlap (0.5 pt tolerance)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pad = 0.5 * fig.dpi / 72
    boxes = []
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text().strip() or t in ignore:
            continue
        bb = _text_bbox(t, r)
        if bb.width <= 0:
            continue
        boxes.append((t.get_text(), bb.expanded(1, 1).padded(-pad)))
    bad = [(a, b) for i, (a, ba) in enumerate(boxes) for (b, bb) in boxes[i + 1 :] if ba.overlaps(bb)]
    if bad:
        raise AssertionError("text overlap: " + "; ".join(f"{a!r} x {b!r}" for a, b in bad))


def assert_text_inside(fig, margin_pt=1.0):
    """Raise if any visible text leaves the canvas."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.get_size_inches() * fig.dpi
    m = margin_pt * fig.dpi / 72
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        bb = _text_bbox(t, r)
        if bb.x0 < -m or bb.y0 < -m or bb.x1 > W + m or bb.y1 > H + m:
            raise AssertionError(f"text off canvas: {t.get_text()!r} {bb}")


# ---- hand-drawn kit (coordinates in inches on an equal-aspect axes) ---------
class Rough:
    """Seeded wobble so every build is byte-stable."""

    def __init__(self, ax, seed=7):
        self.ax = ax
        self.rng = np.random.default_rng(seed)

    # -- geometry
    def _wobble(self, pts, amp, wavelength=0.55, closed=False):
        pts = np.asarray(pts, float)
        d = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
        t = np.gradient(pts, axis=0)
        n = np.c_[-t[:, 1], t[:, 0]]
        n /= np.maximum(np.hypot(n[:, 0], n[:, 1]), 1e-9)[:, None]
        ph = self.rng.uniform(0, 2 * np.pi, 3)
        wl = wavelength * self.rng.uniform(0.8, 1.25, 3) * np.array([1.0, 0.47, 2.1])
        if closed:  # whole periods around the outline, so its two ends meet without a step
            wl = d[-1] / np.maximum(1, np.round(d[-1] / wl))
        noise = (
            np.sin(2 * np.pi * d / wl[0] + ph[0])
            + 0.5 * np.sin(2 * np.pi * d / wl[1] + ph[1])
            + 0.7 * np.sin(2 * np.pi * d / wl[2] + ph[2])
        ) / 2.2
        return pts + n * (amp * noise)[:, None]

    @staticmethod
    def _rrect_pts(x, y, w, h, r, step=0.02):
        r = min(r, w / 2, h / 2)
        pts = []
        corners = [(x + w - r, y + h - r, 0), (x + r, y + h - r, 90), (x + r, y + r, 180), (x + w - r, y + r, 270)]

        # start mid top edge, go counter-clockwise
        def seg(p, q):
            n = max(2, int(np.hypot(q[0] - p[0], q[1] - p[1]) / step))
            return [(p[0] + (q[0] - p[0]) * i / n, p[1] + (q[1] - p[1]) * i / n) for i in range(n)]

        def arc(cx, cy, a0):
            a = np.radians(np.linspace(a0, a0 + 90, 13))
            return list(zip(cx + r * np.cos(a), cy + r * np.sin(a)))

        start = (x + w * 0.42, y + h)
        pts += seg(start, (x + r, y + h))
        pts += arc(*corners[1][:2], 90)
        pts += seg((x, y + h - r), (x, y + r))
        pts += arc(*corners[2][:2], 180)
        pts += seg((x + r, y), (x + w - r, y))
        pts += arc(*corners[3][:2], 270)
        pts += seg((x + w, y + r), (x + w, y + h - r))
        pts += arc(*corners[0][:2], 0)
        pts += seg((x + w - r, y + h), start) + [start]
        return np.array(pts)

    # -- primitives
    def line(self, pts, color=INK2, lw=0.8, amp=0.006, z=3, alpha=1.0, wavelength=0.55, **kw):
        p = self._wobble(self._densify(pts), amp, wavelength)
        self.ax.plot(
            p[:, 0],
            p[:, 1],
            color=color,
            lw=lw,
            zorder=z,
            alpha=alpha,
            solid_capstyle="round",
            solid_joinstyle="round",
            **kw,
        )
        return p

    @staticmethod
    def _densify(pts, step=0.02):
        pts = np.asarray(pts, float)
        out = [pts[0]]
        for p, q in itertools.pairwise(pts):
            n = max(1, int(np.hypot(*(q - p)) / step))
            out += [p + (q - p) * i / n for i in range(1, n + 1)]
        return np.array(out)

    def rrect(
        self, x, y, w, h, r=0.05, fc="none", ec=INK2, lw=0.8, amp=0.006, z=2, passes=1, wavelength=0.55, dashes=None
    ):
        """Rounded box: crisp soft fill, hand-drawn outline (dashes=(on, off) in points: dashed)."""
        if fc != "none":
            self.ax.add_patch(
                FancyBboxPatch(
                    (x, y),
                    w,
                    h,
                    boxstyle=f"round,pad=0,rounding_size={min(r, w / 2, h / 2)}",
                    fc=fc,
                    ec="none",
                    zorder=z,
                )
            )
        if ec != "none":
            base = self._rrect_pts(x, y, w, h, r)
            for i in range(passes):
                p = self._wobble(base, amp, wavelength, closed=True)
                self.ax.plot(
                    p[:, 0],
                    p[:, 1],
                    color=ec,
                    lw=lw,
                    zorder=z + 0.1,
                    alpha=1.0 if i == 0 else 0.55,
                    solid_capstyle="round",
                    solid_joinstyle="round",
                    **({"dashes": dashes, "dash_capstyle": "round"} if dashes else {}),
                )

    def arrow(self, p0, p1, color=INK2, lw=0.9, bow=0.0, head=0.055, amp=0.004, z=4, spread=27):
        """Open-headed arrow with a gently bowed, hand-drawn shaft."""
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        d = p1 - p0
        L = np.hypot(*d)
        nrm = np.array([-d[1], d[0]]) / max(L, 1e-9)
        ctrl = (p0 + p1) / 2 + nrm * bow
        t = np.linspace(0, 1, max(8, int(L / 0.02)))[:, None]
        pts = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * ctrl + t**2 * p1
        pts = self._wobble(pts, amp if L > 0.25 else 0.0)
        pts[-1] = p1
        self.ax.plot(
            pts[:, 0], pts[:, 1], color=color, lw=lw, zorder=z, solid_capstyle="round", solid_joinstyle="round"
        )
        tang = p1 - ctrl
        ang = np.arctan2(tang[1], tang[0])
        for s in (-1, 1):
            a = ang + np.pi + s * np.radians(spread)
            q = p1 + head * np.array([np.cos(a), np.sin(a)])
            self.ax.plot([p1[0], q[0]], [p1[1], q[1]], color=color, lw=lw, zorder=z, solid_capstyle="round")

    def route(self, pts, color=INK2, lw=0.9, head=0.05, amp=0.003, z=4, spread=27, rounds=3):
        """Arrow along a polyline with rounded corners (Chaikin), open head at the end
        (head=0: no head)."""
        p = np.asarray(pts, float)
        for _ in range(rounds):
            q = [p[0]]
            for a, b in itertools.pairwise(p):
                q += [0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b]
            q.append(p[-1])
            p = np.array(q)
        end = p[-1].copy()
        p = self._wobble(self._densify(p), amp)
        p[-1] = end
        self.ax.plot(p[:, 0], p[:, 1], color=color, lw=lw, zorder=z, solid_capstyle="round", solid_joinstyle="round")
        if not head:  # a plain line: the first half of an arrow that carries its label
            return
        tang = np.asarray(pts[-1], float) - np.asarray(pts[-2], float)
        ang = np.arctan2(tang[1], tang[0])
        for s in (-1, 1):
            a = ang + np.pi + s * np.radians(spread)
            q = end + head * np.array([np.cos(a), np.sin(a)])
            self.ax.plot([end[0], q[0]], [end[1], q[1]], color=color, lw=lw, zorder=z, solid_capstyle="round")

    def polyarrow(self, pts, color=INK2, lw=0.9, head=0.045, z=4, spread=27):
        """Arrow along the given points, open head at the end.  Drawn crisp: a stroke that must
        bracket, step and turn within a few points of height has no room for a wobble."""
        p = np.asarray(pts, float)
        self.ax.plot(p[:, 0], p[:, 1], color=color, lw=lw, zorder=z, solid_capstyle="round", solid_joinstyle="round")
        ang = np.arctan2(*(p[-1] - p[-2])[::-1])
        for s in (-1, 1):
            b = ang + np.pi + s * np.radians(spread)
            self.ax.plot(
                [p[-1, 0], p[-1, 0] + head * np.cos(b)],
                [p[-1, 1], p[-1, 1] + head * np.sin(b)],
                color=color,
                lw=lw,
                zorder=z,
                solid_capstyle="round",
            )

    def hachure(self, x, y, w, h, r=0.03, color=INK2, lw=0.6, gap=0.035, z=3, amp=0.0):
        """Diagonal hand strokes clipped to a rounded box (the 'maybe' texture)."""
        clip = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={min(r, w / 2, h / 2)}",
            fc="none",
            ec="none",
            transform=self.ax.transData,
        )
        self.ax.add_patch(clip)
        k = np.arange(-h, w + gap, gap)
        for x0 in k:
            pts = np.array([[x + x0, y], [x + x0 + h, y + h]])
            if amp:
                pts = self._wobble(self._densify(pts), amp, 0.3)
            (ln,) = self.ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, zorder=z, solid_capstyle="round")
            ln.set_clip_path(clip)


def pill(ax, x, y, w, h, fc, r=None, z=3, ec="none", lw=0.0):
    """Crisp rounded block (data mark / card)."""
    r = min(h / 2, w / 2) if r is None else min(r, h / 2, w / 2)
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z)
    ax.add_patch(p)
    return p


def inch_axes(fig):
    """Full-canvas axes whose data units are inches."""
    W, H = fig.get_size_inches()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    return ax
