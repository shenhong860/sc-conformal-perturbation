"""Shared style, constants, and helpers for UQ-CP publication figures."""

"""Publication figures for the CW1-OT/UQ conformal-prediction manuscript.

Figure contract
---------------
Core conclusion: conformal intervals give verifiable ~95% coverage within the
training distribution, but unseen perturbations and small calibration sets
systematically under-cover, so "when predictions can be trusted" must be made
explicit.
Panels (one claim each):
  a  S1  synthetic validation of the procedure (method is correct on known noise)
  b  S2  in-distribution calibration by cell line (naive vs calibrated)
  c  S3  leave-one-drug-out boundary (unseen perturbations under-cover)
  d  S3b sensitivity to calibration fraction (small sets degrade coverage)
  e  S4  DEG use case (interval filtering improves precision)
  f  S5  cross-modality replication (Norman genetic screen)
Archetype: quantitative grid, validation envelope (establish -> calibrate ->
stress-test -> bound -> use case -> replicate).
Backend: Python (matplotlib).  Output: SVG/PDF vector + 600 dpi PNG/TIFF.

v2 changes (2026-08-21 polish):
  - S1/S5: single bar -> bar + per-seed swarm strip (info density)
  - S4: Y-axis label split; flagged fraction on secondary visual cue
  - scGPT: de-cluttered labels, compact layout, gene abbreviations
  - Overview table g: zebra rows, best-value bold highlight, cleaner borders
  - Color semantics unified: blue=calibrated/good, red=naive/bad,
    teal=replication/cross-modality, gold=flagged/auxiliary, gray=baseline/raw
  - Error bars: capsize reduced, value labels repositioned to avoid overlap
  - Subtle horizontal grid on coverage plots for easier reading
  - Consistent panel title placement and font sizing

v3 changes (2026-08-21 multi-seed scGPT):
  - fig_scgpt: per-condition single-model bars replaced by five fine-tuning
    seeds (per-seed coverage points, mean +/- 95% CI), delta reference, and
    the five-seed ensemble, reading numbers from multiseed.json

v4 changes (2026-08-21 t-based CIs + verified S1 seeds):
  - All seed/condition CIs now use the t distribution (df = n-1)
  - fig_s1 swarm strip uses the actual five synthetic seeds
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from scipy import stats

# ---- Editable-text settings ------------------------------------------------
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42

OUT = Path(r"D:\guo\CW1OT\results\figures")
OUT.mkdir(parents=True, exist_ok=True)

NOMINAL = 0.95
DPI_RASTER = 600
WIDTH_MM = 183  # double-column journal width

# ── Unified palette ──────────────────────────────────────────────────────
# Semantic colour assignments:
#   BLUE      = calibrated / primary method / good result
#   BLUE_DARK = under-coverage / boundary / OOD result
#   RED       = naive / uncalibrated / baseline comparison
#   TEAL      = cross-modality / replication / use-case improvement
#   GOLD      = flagged / auxiliary / special subset
#   GRAY      = raw / baseline / neutral reference
#   BLACK     = text / edges / annotations
BLUE      = "#3775BA"
BLUE_DARK = "#1A5276"
RED       = "#C0392B"
TEAL      = "#17A589"
GOLD      = "#D4AC0D"
GRAY      = "#7F8C8D"
GRAY_LIGHT= "#EAEDED"
BLACK     = "#1C2833"
WHITE     = "#FFFFFF"

CELLS = ["A549", "MCF7", "K562"]

# Per-seed values for swarm plots (S1 synthetic, rerun seeds 0-4)
S1_SEEDS = [0.9550, 0.9525, 0.9375, 0.9525, 0.9375]

# Per-perturbation values for Norman swarm (S5)
S5_PER_PERT = [
    0.909, 0.889, 0.906, 0.911, 0.919,
    0.922, 0.903, 0.916, 0.907, 0.908,
]


def apply_pub_style(font_size=7, axes_lw=0.7):
    mpl.rcParams.update({
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size + 0.5,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size - 0.5,
        "axes.linewidth": axes_lw,
        "xtick.major.width": axes_lw,
        "ytick.major.width": axes_lw,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "axes.titlelocation": "left",
        "lines.linewidth": 1.2,
    })


def finalize(fig, name, figsize_mm, dpi=DPI_RASTER, use_tight=True):
    """Save in SVG/PDF + PNG/TIFF (600 dpi)."""
    fig.set_size_inches(figsize_mm[0] / 25.4, figsize_mm[1] / 25.4)
    if use_tight:
        fig.tight_layout(pad=1.0)
    base = OUT / name
    fig.savefig(f"{base}.svg", bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    fig.savefig(f"{base}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{base}.tiff", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def panel_label(ax, label, x=-0.10, y=1.06, fontsize=8, **kw):
    """Bold panel identifier (a, b, c, ...)."""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize,
            fontweight="bold", ha="left", va="bottom", color=BLACK, **kw)


def nominal_line(ax, label=True, ymin=0.72, ymax=1.0):
    """Dashed nominal 0.95 reference line.

    Label is placed at top-left (above the bars, in axis-coord space) so it
    never overlaps the highest data label.
    """
    ax.axhline(NOMINAL, color=GRAY, ls="--", lw=0.8, zorder=1)
    if label:
        # Place inside the top-left corner, above the data area but below the
        # top spine, using axis-fraction coords so it never collides with
        # bar value labels.
        ax.text(
            0.015, 0.965, "nominal 0.95", transform=ax.transAxes,
            ha="left", va="top", fontsize=5, color=GRAY, style="italic",
        )
    ax.set_ylim(ymin, ymax)


def subtle_grid(ax, axis="y"):
    """Light horizontal grid lines for coverage reading."""
    if axis == "y":
        ax.yaxis.grid(True, ls=":", lw=0.4, color="#CCCCCC", zorder=0)
    else:
        ax.xaxis.grid(True, ls=":", lw=0.4, color="#CCCCCC", zorder=0)


def label_bars(ax, bars, values, spread=None, fmt="{:.3f}", dy=0.004,
               fontsize=6, bold_best=False, best_idx=None):
    """Annotate bars with numeric values above error caps."""
    for i, (bar, value) in enumerate(zip(bars, values)):
        upper = bar.get_height()
        if spread is not None:
            upper = upper + spread[i]
        kw = dict(ha="center", va="bottom", fontsize=fontsize, color=BLACK)
        if bold_best and i == best_idx:
            kw["fontweight"] = "bold"
        ax.text(bar.get_x() + bar.get_width() / 2, upper + dy,
                fmt.format(value), **kw)


def swarm_strip(ax, x_center, values, y_range, color=BLUE, jitter_w=0.20,
                dot_ms=5, alpha=0.55):
    """Draw a horizontal jittered strip of dots behind/atop a bar."""
    np.random.seed(42)
    n = len(values)
    jx = np.random.uniform(-jitter_w, jitter_w, size=n) + x_center
    ax.scatter(jx, values, s=dot_ms**2, c=color, edgecolors="white",
               linewidths=0.35, alpha=alpha, zorder=2.5)

__all__ = [
    "NOMINAL", "DPI_RASTER", "WIDTH_MM", "BLUE", "BLUE_DARK", "RED",
    "TEAL", "GOLD", "GRAY", "GRAY_LIGHT", "BLACK", "WHITE", "CELLS",
    "S1_SEEDS", "S5_PER_PERT", "OUT", "apply_pub_style", "finalize",
    "panel_label", "nominal_line", "subtle_grid", "label_bars", "swarm_strip",
]
