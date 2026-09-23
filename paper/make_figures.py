"""Figures for the IEEE paper draft. Every number is read from the evaluation
outputs in backend/ -- nothing here is typed in or illustrative.

    cd paper && python make_figures.py

Writes fig1_architecture.pdf, fig2_roc.pdf, fig3_normalisation.pdf (plus PNG
previews). Vector PDF is what the LaTeX build uses.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

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
def architecture(path):
    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 42)
    ax.axis("off")

    def box(x, y, w, h, title, lines, face="white"):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.35,rounding_size=0.8",
            linewidth=0.8, edgecolor=INK, facecolor=face, zorder=2))
        ax.text(x + w / 2, y + h - 3.1, title, ha="center", va="top",
                fontsize=8, fontweight="bold", color=INK, zorder=3)
        for i, line in enumerate(lines):
            ax.text(x + w / 2, y + h - 7.0 - i * 3.3, line, ha="center",
                    va="top", fontsize=6.2, color=INK2, zorder=3)

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

    box(fuse_x, 10.6, fuse_w, 20.4, "Availability-aware fusion", [])
    ax.text(fuse_x + fuse_w / 2, 21.4,
            r"$c=\dfrac{\sum_{k\in A} s_k w_k}{\sum_{k\in A} w_k}$",
            ha="center", va="center", fontsize=8, color=INK, zorder=3)
    ax.text(fuse_x + fuse_w / 2, 13.9,
            "$A$ = signals that ran\nunavailable $\\neq$ score 0.0\n"
            "$A = \\varnothing \\Rightarrow c = 0.5$",
            ha="center", va="center", fontsize=6.0, color=INK2,
            linespacing=1.4, zorder=3)

    arrow(fuse_x + fuse_w + 0.4, mid, 83.6, mid)

    # verdict bands
    ax.add_patch(FancyBboxPatch(
        (84.0, 10.6), 15.0, 20.4, boxstyle="round,pad=0.35,rounding_size=0.8",
        linewidth=0.8, edgecolor=INK, facecolor="white", zorder=2))
    ax.text(91.5, 27.9, "Verdict", ha="center", va="top", fontsize=8,
            fontweight="bold", color=INK, zorder=3)
    bands = [("AI_GENERATED", "$c \\geq 0.65$", RED),
             ("UNCERTAIN", "$0.35 < c < 0.65$", "#8A6210"),
             ("LIKELY_REAL", "$c \\leq 0.35$", GREEN)]
    for i, (name, rule, colour) in enumerate(bands):
        y = 21.4 - i * 4.4
        ax.add_patch(FancyBboxPatch((85.4, y - 1.4), 1.1, 2.8,
                                    boxstyle="round,pad=0.05,rounding_size=0.3",
                                    linewidth=0, facecolor=colour, zorder=3))
        ax.text(87.4, y + 0.55, name, ha="left", va="center", fontsize=5.8,
                color=INK, zorder=3)
        ax.text(87.4, y - 1.05, rule, ha="left", va="center", fontsize=5.4,
                color=INK2, zorder=3)

    fig.tight_layout(pad=0.15)
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


if __name__ == "__main__":
    architecture(OUT / "fig1_architecture.pdf")
    roc(OUT / "fig2_roc.pdf")
    normalisation(OUT / "fig3_normalisation.pdf")
