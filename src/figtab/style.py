"""Shared visual grammar for every manuscript figure.

One encoding is used across the whole manuscript. The exact solver is always charcoal, the
tabu search is always the problem-specific blue, and each method family keeps its hue in
every figure. Methods inside a family are separated by line style and marker so that no
distinction rests on hue alone and every panel survives grayscale printing. Figures are
drawn at the final column width, in a sans-serif face, on a white ground with sparse grid
lines subordinate to the data.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.analysis.panel import CLASS_COLOR, CLASS_LABEL, FAMILY_COLOR, color, display  # noqa: E402,F401

WIDTH = 6.3      # double-column width of the preprint layout, in inches
HALF = 3.15      # single-column width
INK = "#111111"
MUTED = "#5A5A5A"
GREY = "#C8C8C4"
FAINT = "#E4E4E0"


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 8.0,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "text.color": INK,
        "axes.titlesize": 8.5,
        "axes.titleweight": "normal",
        "axes.labelsize": 8.0,
        "axes.labelcolor": INK,
        "axes.edgecolor": MUTED,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": FAINT,
        "grid.alpha": 1.0,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 7.0,
        "legend.handlelength": 1.9,
        "legend.columnspacing": 1.0,
        "legend.labelspacing": 0.3,
        "legend.borderpad": 0.2,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.2,
        "lines.solid_capstyle": "round",
        "patch.linewidth": 0.6,
    })


def panel_label(ax, text: str, title: str = "", dx: float = -0.02,
                dy: float = 1.04) -> None:
    """Panel letter in the upper-left corner, aligned the same way in every figure.

    When a title is given the letter is set as part of the title, which keeps the two from
    colliding and matches the convention of the journal.
    """
    if title:
        ax.set_title(f"{text} {title}", fontsize=8, loc="left")
        return
    ax.text(dx, dy, text, transform=ax.transAxes, fontsize=8.5, fontweight="bold",
            va="bottom", ha="left", color=INK)


def save(fig, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
