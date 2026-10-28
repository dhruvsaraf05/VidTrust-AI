"""Figures for the IEEE paper draft. Every number is read from the evaluation
outputs in backend/ -- nothing here is typed in or illustrative.

    cd paper && python make_figures.py

Writes fig1_architecture.pdf, fig2_roc.pdf, fig3_normalisation.pdf and
fig4_robustness.pdf (plus PNG previews). Vector PDF is what the LaTeX build
uses.

vidtrust_ieee.tex includes fig1 and fig4. fig2 and fig3 were cut to fit the
six-page conference limit once the operating-envelope section was added; they
are still produced because REPORT.md and the slide deck use them, and a
journal-length version would want fig2 back. After touching fig1, run
check_fig1_layout.py: it fails on any overlapping text.
"""
from __future__ import annotations

import csv
import numpy as np
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.text import Text
from matplotlib.transforms import Bbox

BACKEND = Path(__file__).resolve().parent.parent / "backend"
OUT = Path(__file__).resolve().parent

# Series colours validated for colour-vision deficiency separation; the three
# curves additionally differ in dash pattern so the figure survives grayscale
# printing, which is how a conference proceedings is often read.
BLUE, RED, GREEN = "#2F5BB3", "#B23A3A", "#1F8A70"
INK, INK2, RULE, SUNK = "#16171A", "#55575C", "#C9C6C0", "#ECEAE6"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,          # embed TrueType; IEEE PDF-eXpress rejects Type 3
    "ps.fonttype": 42,
})


def rows(name):
    with (BACKEND / name).open(newline="", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r["expected"] in ("AI", "REAL")]


NAIVE = rows("evaluation_report_public.csv")
NORM = rows("evaluation_report_public_normalised.csv")
ABL = json.loads((BACKEND / "ablation.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Fig. 1 -- architecture
# ---------------------------------------------------------------------------
# Fig. 1 is drawn on an auto-aspect axes, so a data unit stretches with the
# figure height while text stays at its point size. Any text offset written in
# data units therefore drifts whenever the figure is resized; shrinking it from
# 2.5in to 1.9in left the signal boxes with 11pt above their text and none
# below. So text positions given at creation are only first guesses: after
# layout, _place() measures every row and positions it inside its box.
def _shift(ax, artists, dx_px, dy_px):
    """Move texts and patches by a displacement given in display pixels."""
    inverse = ax.transData.inverted()
    x0, y0 = inverse.transform((0.0, 0.0))
    x1, y1 = inverse.transform((dx_px, dy_px))
    for artist in artists:
        if isinstance(artist, Text):
            x, y = artist.get_position()
            artist.set_position((x + x1 - x0, y + y1 - y0))
        else:
            artist.set_x(artist.get_x() + x1 - x0)
            artist.set_y(artist.get_y() + y1 - y0)


def _extent(artists, renderer):
    return Bbox.union([a.get_window_extent(renderer) for a in artists])


def _place(fig, ax, frame, rows, gaps_pt=None):
    """Lay rows of artists out top to bottom inside frame, from measured size.

    gaps_pt=None spaces the rows evenly, top and bottom margins included.
    Otherwise consecutive rows are separated by the given gaps, in points, and
    the whole stack is centred. Each row moves as a unit.
    """
    renderer = fig.canvas.get_renderer()
    box = frame.get_window_extent(renderer)
    extents = [_extent(row, renderer) for row in rows]
    if gaps_pt is None:
        gap = (box.height - sum(e.height for e in extents)) / (len(rows) + 1)
        gaps = [gap] * (len(rows) - 1)
        top = box.y1 - gap
    else:
        gaps = [g * fig.dpi / 72.0 for g in gaps_pt]
        stack = sum(e.height for e in extents) + sum(gaps)
        top = box.y1 - (box.height - stack) / 2
    for i, (row, extent) in enumerate(zip(rows, extents)):
        _shift(ax, row, 0.0, top - extent.y1)
        top -= extent.height + (gaps[i] if i < len(gaps) else 0.0)


def architecture(path):
    # 1.9in, not the 2.5in this started at: a full-width float costs twice its
    # height in column inches, and the paper is against a hard six-page limit.
    # Box geometry is in data units; every label inside a box is positioned by
    # the placement pass at the end, so the height can change without the text
    # drifting. paper/check_fig1_layout.py fails on any overlap or off-centre
    # box; run it after changing anything here.
    fig, ax = plt.subplots(figsize=(7.0, 1.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 42)
    ax.axis("off")

    # Text inside the small boxes: title-to-first-line and line-to-line gaps,
    # in points, so they hold at any figure height.
    TITLE_GAP_PT, LINE_GAP_PT = 2.6, 1.6
    small_boxes = []     # (frame, title, lines), centred by _place below

    def frame_at(x, y, w, h):
        frame = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.35,rounding_size=0.8",
            linewidth=0.8, edgecolor=INK, facecolor="white", zorder=2)
        ax.add_patch(frame)
        return frame

    def box(x, y, w, h, title, lines):
        frame = frame_at(x, y, w, h)
        heading = ax.text(x + w / 2, y + h / 2, title, ha="center", va="top",
                          fontsize=8, fontweight="bold", color=INK, zorder=3)
        body = [ax.text(x + w / 2, y + h / 2, line, ha="center", va="top",
                        fontsize=6.2, color=INK2, zorder=3) for line in lines]
        small_boxes.append((frame, heading, body))
        return frame, heading

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=7, linewidth=0.8,
                                     color=INK2, zorder=1,
                                     shrinkA=0, shrinkB=0))

    # geometry
    rows_y = (28.5, 14.8, 1.1)          # classifier / provenance / frequency
    centres = [y + 6.0 for y in rows_y]
    mid = centres[1]
    sig_x, sig_w = 22.0, 30.0

    # input, and the bus that fans it to all three signals
    box(0.5, 8.8, 13, 12, "Input", ["image or video", "(upload or URL)"])
    ax.plot([14.2, 18.0], [14.8, 14.8], color=INK2, lw=0.8, zorder=1)
    ax.plot([18.0, 18.0], [centres[0], centres[2]], color=INK2, lw=0.8, zorder=1)
    for cy in centres:
        arrow(18.0, cy, sig_x - 0.4, cy)
    ax.text(16.4, 39.4, "video: 1 fps,\n$\\leq$ 60 frames", ha="center",
            va="top", fontsize=5.8, color=INK2, linespacing=1.2)

    box(sig_x, rows_y[0], sig_w, 12, "Classifier   $w = 0.60$",
        ["SwinV2, pretrained, inference only", "scored per frame for video"])
    box(sig_x, rows_y[1], sig_w, 12, "Provenance   $w = 0.25$",
        ["EXIF / XMP / C2PA fingerprints", "read once per file"])
    box(sig_x, rows_y[2], sig_w, 12, "Frequency   $w = 0.15$",
        ["FFT high-frequency energy ratio", "every 5th frame  (UNFITTED)"])

    # signals converge on the fusion block
    fuse_x, fuse_w = 58.0, 21.0
    for cy in centres:
        ax.plot([sig_x + sig_w + 0.4, 55.0], [cy, cy], color=INK2, lw=0.8,
                zorder=1)
    ax.plot([55.0, 55.0], [centres[0], centres[2]], color=INK2, lw=0.8, zorder=1)
    arrow(55.0, mid, fuse_x - 0.4, mid)

    # The fusion and verdict boxes use the full height the three-signal stack
    # occupies: at 20.4 units the display-style fraction collided with both the
    # box title and the subtext. Their contents are spaced evenly by _place.
    tall_y, tall_h = 3.0, 36.0
    fusion = frame_at(fuse_x, tall_y, fuse_w, tall_h)
    fusion_rows = [
        [ax.text(fuse_x + fuse_w / 2, 30.0, "Availability-aware fusion",
                 ha="center", va="top", fontsize=8, fontweight="bold",
                 color=INK, zorder=3)],
        [ax.text(fuse_x + fuse_w / 2, 21.0,
                 r"$c=\dfrac{\sum_{k\in A} s_k w_k}{\sum_{k\in A} w_k}$",
                 ha="center", va="center", fontsize=8, color=INK, zorder=3)],
        [ax.text(fuse_x + fuse_w / 2, 10.0,
                 "$A$ = signals that ran\nunavailable $\\neq$ score 0.0\n"
                 "$A = \\varnothing \\Rightarrow c = 0.5$",
                 ha="center", va="center", fontsize=6.0, color=INK2,
                 linespacing=1.55, zorder=3)],
    ]

    arrow(fuse_x + fuse_w + 0.4, mid, 83.6, mid)

    # verdict bands: each is a colour swatch with its name and rule beside it
    verdict = frame_at(84.0, tall_y, 15.0, tall_h)
    verdict_title = ax.text(91.5, 30.0, "Verdict", ha="center", va="top",
                            fontsize=8, fontweight="bold", color=INK, zorder=3)
    bands = [("AI_GENERATED", "$c \\geq 0.65$", RED),
             ("UNCERTAIN", "$0.35 < c < 0.65$", "#8A6210"),
             ("LIKELY_REAL", "$c \\leq 0.35$", GREEN)]
    band_rows = []
    for i, (name, rule, colour) in enumerate(bands):
        y = 26.0 - i * 8.6
        swatch = FancyBboxPatch((85.4, y - 2.2), 1.1, 4.4,
                                boxstyle="round,pad=0.05,rounding_size=0.3",
                                linewidth=0, facecolor=colour, zorder=3)
        ax.add_patch(swatch)
        label = ax.text(87.4, y + 1.25, name, ha="left", va="center",
                        fontsize=5.8, color=INK, zorder=3)
        detail = ax.text(87.4, y - 1.75, rule, ha="left", va="center",
                         fontsize=5.4, color=INK2, zorder=3)
        band_rows.append((swatch, [label, detail]))

    fig.tight_layout(pad=0.15)

    # --- placement pass ----------------------------------------------------
    # Measured at 300 dpi: text extents shift slightly with rasterisation, and
    # 300 dpi is both the PNG's resolution and the closest to the vector PDF.
    fig.set_dpi(300)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    for frame, heading, body in small_boxes:
        _place(fig, ax, frame, [[heading]] + [[line] for line in body],
               gaps_pt=[TITLE_GAP_PT] + [LINE_GAP_PT] * (len(body) - 1))

    _place(fig, ax, fusion, fusion_rows)

    # Within each band, centre the name-and-rule pair on its swatch; then
    # centre the whole column of bands left-to-right in the box.
    for swatch, words in band_rows:
        mark, text = _extent([swatch], renderer), _extent(words, renderer)
        _shift(ax, words, 0.0, (mark.y0 + mark.y1) / 2 - (text.y0 + text.y1) / 2)
    column = [a for swatch, words in band_rows for a in [swatch] + words]
    box_extent, column_extent = (verdict.get_window_extent(renderer),
                                 _extent(column, renderer))
    _shift(ax, column, (box_extent.x0 + box_extent.x1) / 2
           - (column_extent.x0 + column_extent.x1) / 2, 0.0)
    _place(fig, ax, verdict,
           [[verdict_title]] + [[swatch] + words for swatch, words in band_rows])

    fig.savefig(path, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print("wrote", path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig. 2 -- ROC, both conditions
# ---------------------------------------------------------------------------
def roc_points(scores, truth):
    """Empirical ROC with ties collapsed into one segment."""
    pairs = sorted(zip(scores, truth), key=lambda p: -p[0])
    n_pos, n_neg = sum(truth), len(truth) - sum(truth)
    xs, ys, tp, fp, i = [0.0], [0.0], 0, 0, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            tp += pairs[j][1]
            fp += 1 - pairs[j][1]
            j += 1
        xs.append(fp / n_neg)
        ys.append(tp / n_pos)
        i = j
    return xs, ys


def area(xs, ys):
    return sum((xs[k + 1] - xs[k]) * (ys[k + 1] + ys[k]) / 2
               for k in range(len(xs) - 1))


def roc(path):
    series = [("Ensemble (fused)", "confidence", BLUE, "-"),
              ("Classifier only", "model_score", RED, (0, (5, 2))),
              ("Frequency only", "frequency_score", GREEN, (0, (1.6, 1.6)))]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)

    for ax, (title, data) in zip(axes, [("(a) naive", NAIVE),
                                        ("(b) resolution-normalised", NORM)]):
        truth = [1 if r["expected"] == "AI" else 0 for r in data]
        ax.plot([0, 1], [0, 1], color=RULE, lw=0.8, ls="--", zorder=1)
        for label, col, colour, dash in series:
            xs, ys = roc_points([float(r[col]) for r in data], truth)
            auc = area(xs, ys)
            ax.plot(xs, ys, color=colour, lw=1.5, ls=dash, zorder=3,
                    solid_capstyle="round", label=f"{label}  {auc:.3f}")
            print(f"  {title:26} {label:18} AUC {auc:.4f}")
        ax.set_xlim(-0.012, 1.012)
        ax.set_ylim(-0.012, 1.012)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(labelsize=7, colors=INK2, length=2.5, width=0.6)
        ax.grid(True, color=SUNK, lw=0.5, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(RULE)
        ax.set_xlabel("False positive rate", fontsize=8, color=INK)
        ax.set_title(title, fontsize=8.5, color=INK, pad=4)
        leg = ax.legend(loc="lower right", fontsize=6.8, frameon=True,
                        framealpha=1, edgecolor=RULE, labelcolor=INK,
                        handlelength=2.4, borderpad=0.45,
                        title="AUC", title_fontsize=6.8)
        leg.get_frame().set_linewidth(0.6)

    axes[0].set_ylabel("True positive rate", fontsize=8, color=INK)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print("wrote", path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig. 3 -- what normalisation actually moved
# ---------------------------------------------------------------------------
def normalisation(path):
    """ECDF of the classifier score, per class, in both conditions.

    The point of the figure: the generated curve is unchanged (those images are
    natively 512x512, so the centre crop is a no-op on them) while the real
    curve shifts left. The whole normalised gain is fewer false positives.
    """
    def ecdf(values):
        v = sorted(values)
        n = len(v)
        xs, ys = [0.0], [0.0]
        for i, x in enumerate(v, start=1):
            xs += [x, x]
            ys += [(i - 1) / n, i / n]
        xs.append(1.0)
        ys.append(1.0)
        return xs, ys

    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    # The two generated curves are identical to the last decimal, so the
    # normalised one is drawn first and the naive dashes sit on top of it --
    # otherwise the overlap looks like a single series.
    combos = [
        (NORM, "AI", RED, "-", "generated, normalised"),
        (NAIVE, "AI", RED, (0, (4, 2.4)), "generated, naive"),
        (NAIVE, "REAL", BLUE, (0, (4, 2.4)), "real, naive"),
        (NORM, "REAL", BLUE, "-", "real, normalised"),
    ]
    for data, cls, colour, dash, label in combos:
        xs, ys = ecdf([float(r["model_score"]) for r in data
                       if r["expected"] == cls])
        ax.plot(xs, ys, color=colour, ls=dash, lw=1.4, label=label, zorder=3,
                solid_capstyle="round")

    ax.axvline(0.5, color=INK2, lw=0.7, ls=":", zorder=2)
    ax.text(0.515, 0.045, "0.5", fontsize=6.2, color=INK2)
    ax.set_xlim(-0.012, 1.012)
    ax.set_ylim(0, 1.005)
    ax.set_xlabel("Classifier score  $s_{\\mathrm{model}}$", fontsize=8, color=INK)
    ax.set_ylabel("Cumulative fraction", fontsize=8, color=INK)
    ax.tick_params(labelsize=7, colors=INK2, length=2.5, width=0.6)
    ax.grid(True, color=SUNK, lw=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    leg = ax.legend(loc="center right", fontsize=6.2, frameon=True,
                    framealpha=1, edgecolor=RULE, labelcolor=INK,
                    handlelength=2.2, borderpad=0.4)
    leg.get_frame().set_linewidth(0.6)

    fig.tight_layout(pad=0.3)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print("wrote", path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig. 4 -- the operating envelope
# ---------------------------------------------------------------------------
def robustness(path):
    """Two things AUC alone will not tell you.

    (a) Under JPEG the ranking holds up while the DECISIONS collapse: the
        ensemble's AUC is flat, and the share of generated images still clearing
        the 0.65 threshold falls from 173/200 to 95/200. A detector whose
        ordering is intact but whose operating point has slid out from under it
        is broken in the only way a user experiences.
    (b) Without the resolution control, two of three conditions invert: the
        measurement says the perturbation HELPS. That is the confound, drawn.
    """
    data = json.loads((BACKEND / "robustness.json").read_text(encoding="utf-8"))
    matched, native = data["arms"]["matched"], data["arms"]["native"]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.25))

    # --- (a) the JPEG sweep ------------------------------------------------
    ax = axes[0]
    order = ["clean", "jpeg95", "jpeg75", "jpeg50", "jpeg30"]
    labels = ["none", "q95", "q75", "q50", "q30"]
    x = range(len(order))

    series = [
        ("Ensemble AUC", [matched[c]["auc"]["ensemble"] for c in order], BLUE, "-", "o"),
        ("Classifier AUC", [matched[c]["auc"]["model"] for c in order], RED, (0, (5, 2)), "s"),
        ("Frequency AUC", [matched[c]["auc"]["frequency"] for c in order], GREEN, (0, (1.6, 1.6)), "^"),
    ]
    for label, values, colour, dash, marker in series:
        ax.plot(x, values, color=colour, ls=dash, lw=1.5, marker=marker,
                markersize=4, label=label, zorder=3)

    # The FUSED decision, not the classifier score alone: true positives over
    # all 200 generated images, which is the quantity Table IV reports and the
    # quantity a user of the system actually experiences. (The classifier-only
    # count differs by one file at two conditions; mixing the two definitions
    # across a figure and a table would be the kind of quiet inconsistency this
    # paper is arguing against.)
    detection = [matched[c]["at_frozen_band"]["confusion"]["tp"]
                 / matched[c]["classifier_mean"]["generated_total"] for c in order]
    ax.plot(x, detection, color=INK, ls="-", lw=2.2, marker="D", markersize=4.5,
            label="Generated images still called AI", zorder=4)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylim(0.3, 1.0)
    ax.set_xlabel("JPEG quality applied before analysis", fontsize=8, color=INK)
    ax.set_ylabel("rate", fontsize=8, color=INK)
    ax.set_title("(a) the ranking holds, the decisions do not", fontsize=8.5,
                 color=INK, pad=4)
    ax.tick_params(labelsize=7, colors=INK2, length=2.5, width=0.6)
    ax.grid(True, color=SUNK, lw=0.5, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    leg = ax.legend(loc="lower left", fontsize=6.2, frameon=True, framealpha=1,
                    edgecolor=RULE, labelcolor=INK, handlelength=2.2)
    leg.get_frame().set_linewidth(0.6)

    # --- (b) the confound --------------------------------------------------
    ax = axes[1]
    conditions = ["jpeg50", "noise10", "messaging"]
    width = 0.35
    positions = np.arange(len(conditions))

    def change(arm, condition, signal):
        return arm[condition]["auc"][signal] - arm["clean"]["auc"][signal]

    # The CLASSIFIER, not the ensemble. It carries 0.60 of the weight, it shows
    # the largest effect, and it is where the inversion is starkest: controlled,
    # JPEG 50 costs it 0.110 AUC; uncontrolled, it appears to GAIN 0.029.
    # (The ensemble inverts under JPEG 50 too, and the frequency signal inverts
    # under noise -- but no single signal inverts on all three, so plotting one
    # signal and labelling the flips is the honest version of this chart.)
    matched_delta = [change(matched, c, "model") for c in conditions]
    native_delta = [change(native, c, "model") for c in conditions]

    ax.bar(positions - width / 2, matched_delta, width, color=BLUE,
           label="matched (controlled)", zorder=3)
    ax.bar(positions + width / 2, native_delta, width, color="#B8863B",
           label="native (uncontrolled)", zorder=3)
    ax.axhline(0, color=INK, lw=0.9, zorder=4)

    # Mark only the bars that actually change sign.
    for index, (before, after) in enumerate(zip(matched_delta, native_delta)):
        if (before < 0) != (after < 0):
            ax.annotate("sign\nflips", xy=(positions[index] + width / 2, after),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", va="bottom", fontsize=5.8,
                        color=INK, linespacing=1.1, zorder=5)

    # Headroom so the "sign flips" labels sit inside the axes rather than
    # climbing into the title.
    lowest = min(matched_delta + native_delta)
    highest = max(matched_delta + native_delta)
    ax.set_ylim(lowest - 0.02, highest + 0.055)

    ax.set_xticks(positions)
    ax.set_xticklabels(["JPEG 50", "noise $\\sigma$=10", "messaging"], fontsize=7.5)
    ax.set_ylabel("change in classifier AUC", fontsize=8, color=INK)
    ax.set_title("(b) the same measurement, without the control", fontsize=8.5,
                 color=INK, pad=4)
    ax.tick_params(labelsize=7, colors=INK2, length=2.5, width=0.6)
    ax.grid(True, axis="y", color=SUNK, lw=0.5, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    leg = ax.legend(loc="lower left", fontsize=6.4, frameon=True, framealpha=1,
                    edgecolor=RULE, labelcolor=INK)
    leg.get_frame().set_linewidth(0.6)

    fig.tight_layout(pad=0.4)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print("wrote", path)
    for condition in order:
        tp = matched[condition]["at_frozen_band"]["confusion"]["tp"]
        total = matched[condition]["classifier_mean"]["generated_total"]
        print(f"  {condition:<12} ens AUC {matched[condition]['auc']['ensemble']:.4f}"
              f"   detected {tp}/{total}")
    plt.close(fig)


if __name__ == "__main__":
    architecture(OUT / "fig1_architecture.pdf")
    roc(OUT / "fig2_roc.pdf")
    normalisation(OUT / "fig3_normalisation.pdf")
    robustness(OUT / "fig4_robustness.pdf")
