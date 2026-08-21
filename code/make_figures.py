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

# Per-seed values for swarm plots (S1 synthetic)
S1_SEEDS = [0.945, 0.948, 0.951, 0.943, 0.949]

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


# ═══════════════════════════════════════════════════════════════════════════
# Individual panels
# ═══════════════════════════════════════════════════════════════════════════

def fig_s1():
    """Synthetic validation: bar + per-seed swarm strip."""
    fig, ax = plt.subplots()
    mean, lo, hi = 0.947, 0.938, 0.956

    # Swarm strip (per-seed values) behind bar
    swarm_strip(ax, 0, S1_SEEDS, (lo - 0.01, hi + 0.01), color=BLUE,
                jitter_w=0.18, dot_ms=6, alpha=0.5)

    # Summary bar
    bars = ax.bar([0], [mean], width=0.45, color=BLUE, edgecolor=BLACK,
                  linewidth=0.6, zorder=3, alpha=0.85)
    ax.errorbar(0, mean, yerr=[[mean - lo], [hi - mean]], fmt="none",
                ecolor=BLACK, elinewidth=0.8, capsize=3, capthick=0.8, zorder=4)

    # Annotations — value above, CI label INSIDE bar (white text on colour)
    ax.text(0, hi + 0.008, f"{mean:.3f}", ha="center", va="bottom",
            fontsize=7, fontweight="bold", color=BLACK)
    ax.text(0, 0.905, f"95% CI [{lo:.3f}, {hi:.3f}]", ha="center",
            va="center", fontsize=5.5, color=WHITE, fontweight="bold")

    nominal_line(ax, ymin=0.86, ymax=1.0)
    subtle_grid(ax)
    ax.set_ylabel("Coverage")
    ax.set_xticks([0])
    ax.set_xticklabels(["Synthetic\n(5 seeds)"])
    ax.set_title("Method validation on synthetic data", loc="left", fontsize=6.5,
                 color=GRAY, pad=4)
    panel_label(ax, "a")
    finalize(fig, "fig_s1", (80, 60))


def fig_s2():
    """In-distribution: naive vs calibrated by cell line."""
    fig, ax = plt.subplots()
    x = np.arange(len(CELLS))
    naive = [0.9164, 0.9438, 0.9394]
    cal   = [0.9513, 0.9498, 0.9527]
    naive_sem = [0.0042, 0.0038, 0.0051]
    cal_ci   = [0.0032, 0.0026, 0.0077]
    w = 0.34

    b1 = ax.bar(x - w/2, naive, w, label="Naive", color=RED, edgecolor=BLACK,
                linewidth=0.6, zorder=3,
                yerr=naive_sem,
                error_kw=dict(elinewidth=0.6, capsize=2, capthick=0.6,
                              ecolor=BLACK))
    b2 = ax.bar(x + w/2, cal, w, label="Calibrated", color=BLUE,
                edgecolor=BLACK, linewidth=0.6, zorder=3,
                yerr=cal_ci,
                error_kw=dict(elinewidth=0.6, capsize=2, capthick=0.6,
                              ecolor=BLACK))

    label_bars(ax, b1, naive, spread=naive_sem, fmt="{:.3f}", dy=0.003)
    label_bars(ax, b2, cal, spread=cal_ci, fmt="{:.3f}", dy=0.003)

    nominal_line(ax, ymin=0.84, ymax=0.985)
    subtle_grid(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(CELLS)
    ax.set_ylabel("Coverage")
    ax.legend(loc="lower right", fontsize=5.5,
              handletextpad=0.4, columnspacing=0.8)

    # Compact footnote
    ax.text(0.02, 0.02,
            "Error bars: naive SE / calibrated 95% CI (5 seeds)",
            transform=ax.transAxes, fontsize=5, va="bottom", ha="left",
            color=GRAY, style="italic")

    panel_label(ax, "b")
    finalize(fig, "fig_s2", (88, 60))


def fig_s3():
    """Leave-one-drug-out: unseen perturbations under-cover."""
    fig, ax = plt.subplots()
    vals = [0.8534, 0.8429, 0.9094]
    errs = [0.0028, 0.0024, 0.0069]

    # Gradient from dark blue (worst) to teal (best among OOD)
    colors = [BLUE_DARK, BLUE, TEAL]

    bars = ax.bar(CELLS, vals, width=0.52, color=colors, edgecolor=BLACK,
                  linewidth=0.6, zorder=3, yerr=errs,
                  error_kw=dict(elinewidth=0.7, capsize=2.5, capthick=0.7,
                                ecolor=BLACK))
    label_bars(ax, bars, vals, spread=errs, fmt="{:.3f}", dy=0.003)

    nominal_line(ax, ymin=0.78, ymax=0.975)
    subtle_grid(ax)

    # Under-coverage annotation (between two bars)
    mid_x = 0.5
    ax.annotate("", xy=(mid_x, NOMINAL - 0.005),
                xytext=(mid_x, np.mean(vals) + 0.005),
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.9,
                                connectionstyle="arc3,rad=0.15"))
    ax.text(mid_x + 0.18, (NOMINAL + np.mean(vals)) / 2, "under-cover",
            fontsize=6, color=RED, rotation=90, va="center", ha="left",
            fontweight="bold")

    # Footnote in upper-LEFT, NOT behind the bars
    ax.text(0.02, 0.965, "Mean \u00b1 95% CI (5 seeds)",
            transform=ax.transAxes, fontsize=5, va="top", ha="left",
            color=GRAY, style="italic",
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8,
                      pad=1.2))
    ax.set_ylabel("Coverage")
    panel_label(ax, "c")
    finalize(fig, "fig_s3", (82, 60))


def fig_s3b():
    """Coverage vs calibration-fraction sensitivity."""
    fig, ax = plt.subplots()
    fracs = [25, 10, 5]
    cov  = [0.9522, 0.9303, 0.9275]
    sem  = [0.0032, 0.0056, 0.0050]

    # Shaded confidence band
    ax.fill_between(fracs, [c - s for c, s in zip(cov, sem)],
                    [c + s for c, s in zip(cov, sem)],
                    color=BLUE, alpha=0.12, zorder=1)

    ax.errorbar(fracs, cov, yerr=sem, fmt="none", ecolor=BLACK,
                elinewidth=0.7, capsize=2.5, capthick=0.7, zorder=2)
    ax.plot(fracs, cov, "-o", color=BLUE, lw=1.3, ms=5, zorder=3,
            markeredgecolor=WHITE, markeredgewidth=0.6)

    for fx, cy in zip(fracs, cov):
        ax.text(fx, cy + 0.007, f"{cy:.3f}", ha="center", va="bottom",
                fontsize=6, color=BLACK)

    nominal_line(ax, ymin=0.86, ymax=0.98)
    subtle_grid(ax)
    ax.set_xlabel("Calibration set size (% of data)")
    ax.set_ylabel("Coverage")
    ax.set_xticks(fracs)
    ax.set_xticklabels(["25%", "10%", "5%"])
    ax.set_xlim(28, 2)
    ax.invert_xaxis()  # natural left-to-right: large -> small

    ax.text(0.02, 0.02, "Mean +/- 95% CI over five seeds",
            transform=ax.transAxes, fontsize=5, va="bottom", ha="left",
            color=GRAY, style="italic")
    panel_label(ax, "d", x=-0.08, y=1.08)
    finalize(fig, "fig_s3b", (82, 60))


def fig_s4():
    """DEG prioritization use case: interval filtering."""
    fig, ax = plt.subplots()
    labels = ["Top-50\nraw", "Top-50\nfiltered", "Recall", "Flagged\nfraction"]
    vals  = [0.6570, 0.6699, 0.6800, 0.0955]
    errs  = [0.0061, 0.0091, 0.0103, 0.0026]
    colors = [GRAY, TEAL, BLUE, GOLD]
    x = np.arange(4)

    bars = ax.bar(x, vals, color=colors, edgecolor=BLACK,
                  linewidth=0.6, zorder=3, width=0.58,
                  yerr=errs,
                  error_kw=dict(elinewidth=0.7, capsize=2.5, capthick=0.7,
                                ecolor=BLACK))

    # Highlight improvement with arrow
    ax.annotate("", xy=(1, vals[1] + errs[1] + 0.012),
                xytext=(0, vals[0] + errs[0] + 0.012),
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.0,
                                connectionstyle="arc3,rad=0.2"))
    ax.text(0.5, max(vals[0]+errs[0], vals[1]+errs[1]) + 0.030,
            "+0.013", ha="center", va="bottom", fontsize=5.5,
            color=TEAL, fontweight="bold")

    label_bars(ax, bars, vals, spread=errs, fmt="{:.3f}", dy=0.005)
    ax.set_ylim(0, 0.86)
    ax.set_yticks(np.arange(0, 0.91, 0.2))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5.5)
    ax.set_ylabel("Score")

    # Dual-meaning footnote — placed in upper-left, NOT behind bars
    ax.text(0.015, 0.97,
            "Precision@50 / Recall / Flagged-gene proportion; 5 seeds",
            transform=ax.transAxes, fontsize=5, va="top", ha="left",
            color=GRAY, style="italic",
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.85,
                      pad=1.5))
    panel_label(ax, "e")
    finalize(fig, "fig_s4", (90, 62))


def fig_s5():
    """Norman genetic perturbation: bar + per-perturbation swarm."""
    fig, ax = plt.subplots()

    mean5 = float(np.mean(S5_PER_PERT))
    se5   = float(np.std(S5_PER_PERT, ddof=1) / np.sqrt(len(S5_PER_PERT)))
    lo5, hi5 = mean5 - 1.96 * se5, mean5 + 1.96 * se5

    # Swarm: individual perturbation coverages
    swarm_strip(ax, 0, S5_PER_PERT, (min(S5_PER_PERT)-0.01,
                                     max(S5_PER_PERT)+0.01),
                color=TEAL, jitter_w=0.18, dot_ms=6, alpha=0.5)

    # Summary bar
    bars = ax.bar([0], [mean5], width=0.45, color=TEAL, edgecolor=BLACK,
                  linewidth=0.6, zorder=3, alpha=0.85,
                  yerr=[[mean5 - lo5], [hi5 - mean5]],
                  error_kw=dict(elinewidth=0.8, capsize=3, capthick=0.8,
                                ecolor=BLACK))

    ax.text(0, hi5 + 0.008, f"{mean5:.3f}", ha="center", va="bottom",
            fontsize=7, fontweight="bold", color=BLACK)
    # CI label INSIDE the bar with white text for contrast
    ax.text(0, 0.835, f"95% CI [{lo5:.3f}, {hi5:.3f}]", ha="center",
            va="center", fontsize=5.5, color=WHITE, fontweight="bold")

    nominal_line(ax, ymin=0.80, ymax=0.98)
    subtle_grid(ax)
    ax.set_ylabel("Coverage")
    ax.set_xticks([0])
    ax.set_xticklabels(["Norman genetic\n(K562, 10 perturb.)"])
    ax.set_title("Cross-modality replication", loc="left", fontsize=6.5,
                 color=GRAY, pad=4)
    panel_label(ax, "f")
    finalize(fig, "fig_s5", (82, 60))


def fig_baselines():
    """UQ method baselines: in-distribution vs unseen-drug coverage."""
    fig, axes = plt.subplots(1, 2, figsize=(155 / 25.4, 65 / 25.4))

    methods = ["Fixed\nnormal", "Bootstrap\nSE", "No shrink", "Conformal\n(ours)"]
    colors  = [GRAY, GOLD, RED, BLUE]

    s2_vals = [0.8553, 0.7905, 0.9392, 0.9511]
    s2_errs = [0.0032, 0.0058, 0.0021, 0.0030]
    s3_vals = [0.7687, 0.6747, 0.9130, 0.8658]
    s3_errs = [0.0033, 0.0060, 0.0021, 0.0022]

    for idx, (ax, vals, errs, title, ymin) in enumerate([
        (axes[0], s2_vals, s2_errs, "In-distribution", 0.70),
        (axes[1], s3_vals, s3_errs, "Unseen drug",     0.60),
    ]):
        bars = ax.bar(np.arange(4), vals, width=0.58, color=colors,
                      edgecolor=BLACK, linewidth=0.6, zorder=3,
                      yerr=errs,
                      error_kw=dict(elinewidth=0.7, capsize=2.5, capthick=0.7,
                                    ecolor=BLACK))
        label_bars(ax, bars, vals, spread=errs, fmt="{:.3f}", dy=0.004,
                   fontsize=5.5, bold_best=True, best_idx=3)

        nominal_line(ax, label=False, ymin=ymin, ymax=1.0)
        subtle_grid(ax)
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(methods, fontsize=5.5)
        ax.set_ylabel("Coverage")
        ax.set_title(title, loc="left", fontsize=6.5, color=GRAY, pad=4)

    # Subtitle placed BELOW the panel as a figure-level caption (outside bars)
    fig.text(0.5, 0.01, "sci-Plex3 held-out conditions (5 seeds)",
             ha="center", va="bottom", fontsize=5, color=GRAY, style="italic")

    panel_label(axes[0], "a", x=-0.08, y=1.05)
    panel_label(axes[1], "b", x=-0.08, y=1.05)
    finalize(fig, "fig_baselines", (155, 68))


def fig_scgpt():
    """Seed-level coverage: mean-shift vs five-seed fine-tuned scGPT."""
    fig, ax = plt.subplots(figsize=(160 / 25.4, 70 / 25.4))

    mseed_path = Path(r"D:\guo\CW1OT\results\scgpt_norman\multiseed.json")
    if not mseed_path.exists():
        raise FileNotFoundError(
            f"missing multi-seed scGPT results: {mseed_path}\n"
            "run code/run_scgpt_multiseed.sh first"
        )
    mseed = json.loads(mseed_path.read_text(encoding="utf-8"))

    # Mean-shift reference on the same ten held-out perturbations (10 cond.).
    delta_conds = [0.9050, 0.9079, 0.9070, 0.9089, 0.9128,
                   0.9138, 0.9215, 0.9070, 0.9167, 0.9041]
    delta_mean = float(np.mean(delta_conds))
    delta_ci = float(1.96 * np.std(delta_conds, ddof=1) / np.sqrt(len(delta_conds)))

    seeds = [mseed["per_seed"][str(s)]["coverage"] for s in mseed["seeds"]]
    seed_mean = mseed["coverage_mean"]
    seed_ci = mseed["coverage_ci95"]
    ens_cov = mseed["ensemble"]["coverage"]

    x = np.arange(3)
    w = 0.42

    b1 = ax.bar(0, delta_mean, width=w, color=RED,
                edgecolor=BLACK, linewidth=0.5, zorder=3)
    ax.errorbar(0, delta_mean, yerr=delta_ci, fmt="none", ecolor=BLACK,
                elinewidth=0.7, capsize=2.5, capthick=0.7, zorder=4)
    ax.text(0, delta_mean + delta_ci + 0.006, f"{delta_mean:.3f}",
            ha="center", va="bottom", fontsize=5, color=BLACK)

    # Per-seed fine-tuned scGPT points behind the mean bar.
    np.random.seed(7)
    jx = np.random.uniform(-0.16, 0.16, size=len(seeds)) + 1
    ax.scatter(jx, seeds, s=20, c=BLUE, edgecolors="white",
               linewidths=0.35, alpha=0.55, zorder=2.5)
    b2 = ax.bar(1, seed_mean, width=w, color=BLUE,
                edgecolor=BLACK, linewidth=0.5, zorder=3, alpha=0.85)
    ax.errorbar(1, seed_mean, yerr=seed_ci, fmt="none", ecolor=BLACK,
                elinewidth=0.7, capsize=2.5, capthick=0.7, zorder=4)
    ax.text(1, seed_mean + seed_ci + 0.006, f"{seed_mean:.3f} \u00b1 {seed_ci:.3f}",
            ha="center", va="bottom", fontsize=5, color=BLACK)
    ax.text(1, 0.805, f"per-seed {min(seeds):.3f}\u2013{max(seeds):.3f}",
            ha="center", va="top", fontsize=4.5, color=BLUE_DARK)

    b3 = ax.bar(2, ens_cov, width=w, color=GOLD,
                edgecolor=BLACK, linewidth=0.5, zorder=3)
    ax.text(2, ens_cov + 0.006, f"{ens_cov:.3f}",
            ha="center", va="bottom", fontsize=5, color=BLACK)

    nominal_line(ax, label=False, ymin=0.78, ymax=1.02)
    ax.text(0.985, 0.975, "nominal 0.95", transform=ax.transAxes,
            ha="right", va="top", fontsize=5, color=GRAY, style="italic")
    subtle_grid(ax)

    ax.set_xticks(x)
    ax.set_xticklabels(["Mean-shift (\u03b4)", "scGPT fine-tune\n(5 seeds)",
                        "scGPT ensemble"], fontsize=5.2)
    ax.set_ylabel("Coverage")
    finalize(fig, "fig_scgpt", (160, 78))


# ═══════════════════════════════════════════════════════════════════════════
# Overview: 2x3 + ablation table
# ═══════════════════════════════════════════════════════════════════════════

def overview():
    """Summary figure: 2x3 grid of key results + ablation table row."""
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, 135 / 25.4))
    gs = fig.add_gridspec(
        3, 3, hspace=0.55, wspace=0.32,
        left=0.058, right=0.970, top=0.89, bottom=0.10,
        height_ratios=[1, 1, 0.50],
    )
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis("off")

    a, b, c, d, e, f = axes

    # ── Panel a: Synthetic ──
    swarm_strip(a, 0, S1_SEEDS, (0.93, 0.97), color=BLUE,
                jitter_w=0.16, dot_ms=4.5, alpha=0.45)
    a.bar([0], [0.947], width=0.50, color=BLUE, edgecolor=BLACK,
          linewidth=0.5, zorder=3, alpha=0.85)
    a.errorbar(0, 0.947, yerr=[[0.009], [0.009]], fmt="none",
               ecolor=BLACK, elinewidth=0.7, capsize=2.5, capthick=0.7, zorder=4)
    a.text(0, 0.956 + 0.005, "0.947", ha="center", va="bottom",
           fontsize=6, fontweight="bold", color=BLACK)
    nominal_line(a, label=False, ymin=0.86, ymax=0.97)
    subtle_grid(a)
    a.set_xticks([0])
    a.set_xticklabels(["Synthetic"], fontsize=5.5)
    a.set_ylabel("Coverage", fontsize=6)

    # ── Panel b: Calibration ──
    xb = np.arange(3)
    naive_b = [0.9164, 0.9438, 0.9394]
    cal_b   = [0.9513, 0.9498, 0.9527]
    wb = 0.32
    b.bar(xb - wb/2, naive_b, wb, color=RED, edgecolor=BLACK,
          linewidth=0.5, zorder=3)
    b.bar(xb + wb/2, cal_b, wb, color=BLUE, edgecolor=BLACK,
          linewidth=0.5, zorder=3)
    for i, (nv, cv) in enumerate(zip(naive_b, cal_b)):
        b.text(xb[i] - wb/2, nv + 0.003, f"{nv:.3f}", ha="center",
               fontsize=4.8, color=BLACK)
        b.text(xb[i] + wb/2, cv + 0.003, f"{cv:.3f}", ha="center",
               fontsize=4.8, color=BLACK)
    nominal_line(b, label=False, ymin=0.86, ymax=0.97)
    subtle_grid(b)
    b.set_xticks(xb)
    b.set_xticklabels(CELLS, fontsize=5.5)

    # ── Panel c: Unseen drugs ──
    vals_c = [0.8534, 0.8429, 0.9094]
    c.bar(xb, vals_c, width=0.52,
          color=[BLUE_DARK, BLUE, TEAL], edgecolor=BLACK,
          linewidth=0.5, zorder=3)
    for i, v in enumerate(vals_c):
        c.text(xb[i], v + 0.004, f"{v:.3f}", ha="center",
               fontsize=4.8, color=BLACK)
    nominal_line(c, label=False, ymin=0.78, ymax=0.97)
    subtle_grid(c)
    c.set_xticks(xb)
    c.set_xticklabels(CELLS, fontsize=5.5)
    c.set_yticks([0.80, 0.85, 0.90, 0.95])

    # ── Panel d: Calibration fraction ──
    fracs_d = [25, 10, 5]
    cov_d  = [0.9522, 0.9303, 0.9275]
    sem_d  = [0.0032, 0.0056, 0.0050]
    d.fill_between(fracs_d, [c-s for c,s in zip(cov_d,sem_d)],
                   [c+s for c,s in zip(cov_d,sem_d)],
                   color=BLUE, alpha=0.10, zorder=1)
    d.errorbar(fracs_d, cov_d, yerr=sem_d, fmt="none", ecolor=BLACK,
               elinewidth=0.6, capsize=2, capthick=0.6, zorder=2)
    d.plot(fracs_d, cov_d, "-o", color=BLUE, lw=1.2, ms=4, zorder=3,
           markeredgecolor=WHITE, markeredgewidth=0.5)
    for fx, cy in zip(fracs_d, cov_d):
        d.text(fx, cy + 0.005, f"{cy:.3f}", ha="center",
               fontsize=4.8, color=BLACK)
    nominal_line(d, label=False, ymin=0.86, ymax=0.98)
    subtle_grid(d)
    d.set_xlabel("Calib. size (%)", fontsize=5.5)
    d.set_xticks(fracs_d)
    d.set_xticklabels(["25%", "10%", "5%"], fontsize=5.5)
    d.set_xlim(28, 2)
    d.invert_xaxis()
    d.set_yticks([0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98])

    # ── Panel e: DEG use case ──
    labels_e = ["Raw", "Filtered", "Recall", "Flagged"]
    vals_e  = [0.6570, 0.6699, 0.6800, 0.0955]
    errs_e  = [0.0061, 0.0091, 0.0103, 0.0026]
    e.bar(np.arange(4), vals_e, width=0.58,
          color=[GRAY, TEAL, BLUE, GOLD], edgecolor=BLACK,
          linewidth=0.5, zorder=3, yerr=errs_e,
          error_kw=dict(elinewidth=0.6, capsize=2, capthick=0.6, ecolor=BLACK))
    for i, (v, err) in enumerate(zip(vals_e, errs_e)):
        e.text(i, v + err + 0.004, f"{v:.3f}", ha="center",
               fontsize=4.8, color=BLACK)
    e.set_ylim(0, 0.84)
    e.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    e.set_xticks(np.arange(4))
    e.set_xticklabels(labels_e, fontsize=5.5)

    # ── Panel f: Norman ──
    mean_f = float(np.mean(S5_PER_PERT))
    se_f   = float(np.std(S5_PER_PERT, ddof=1) / np.sqrt(len(S5_PER_PERT)))
    lo_f, hi_f = mean_f - 1.96*se_f, mean_f + 1.96*se_f
    swarm_strip(f, 0, S5_PER_PERT, (0.88, 0.93), color=TEAL,
                jitter_w=0.16, dot_ms=4, alpha=0.40)
    f.bar([0], [mean_f], width=0.50, color=TEAL, edgecolor=BLACK,
          linewidth=0.5, zorder=3, alpha=0.85,
          yerr=[[mean_f-lo_f], [hi_f-mean_f]],
          error_kw=dict(elinewidth=0.7, capsize=2.5, capthick=0.7, ecolor=BLACK))
    f.text(0, hi_f + 0.006, f"{mean_f:.3f}", ha="center", va="bottom",
           fontsize=6, fontweight="bold", color=BLACK)
    nominal_line(f, label=False, ymin=0.78, ymax=0.98)
    subtle_grid(f)
    f.set_xticks([0])
    f.set_xticklabels(["Norman"], fontsize=5.5)

    # ── Panel g: Ablation table ──
    panel_label(ax_table, "g", x=-0.04, y=1.12, fontsize=8)

    table_rows = [
        ["Base predictor", "Coverage", "Median width", "Precision@50"],
        ["delta (mean-shift)", "0.943", "0.029", "0.777"],
        ["linear (main effects)", "0.951", "0.111", "0.427"],
        ["w1ot (optimal transport)", "0.948", "0.056", "0.717"],
    ]
    tbl = ax_table.table(
        cellText=table_rows, loc="center",
        colWidths=[0.26, 0.18, 0.19, 0.18],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6)
    tbl.scale(1.0, 1.1)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#BBBBBB")
        cell.set_linewidth(0.5)
        cell.set_text_props(color=BLACK, ha="center")
        if row == 0:
            cell.set_facecolor("#E8EAED")
            cell.set_text_props(fontweight="bold", color=BLACK)
        elif col == 0:
            cell.set_facecolor("#F5F7FA")
        # Zebra alternating on data rows
        elif row % 2 == 1:
            cell.set_facecolor("#FFFFFF")
        else:
            cell.set_facecolor("#FAFBFC")

    # Bold the best value in each metric column
    # Coverage: linear 0.951 > w1ot 0.948 > delta 0.943 → bold linear
    # Width: delta 0.029 < w1ot 0.056 < linear 0.111 → bold delta
    # Prec@50: delta 0.777 > w1ot 0.717 > linear 0.427 → bold delta
    best_cells = {(3, 1): True, (1, 2): True, (1, 3): True}  # (row, col)
    for (r, c), is_best in best_cells.items():
        cell = tbl[(r, c)]
        cell.set_text_props(fontweight="bold", color=BLACK)

    # Subtitle as figure-level text (positioned just above the table)
    fig.text(0.5, 0.255, "Base-predictor ablation (six sci-Plex3 conditions)",
             ha="center", va="bottom", fontsize=5.5, color=GRAY,
             fontstyle="italic")

    # ── Shared panel labels & nominal legend ──
    for i, (ax_i, lbl) in enumerate(zip(axes, "abcdef")):
        panel_label(ax_i, lbl, x=-0.10, y=1.12, fontsize=8)

    f.plot([], [], color=GRAY, ls="--", lw=0.8, label="Nominal 0.95")
    f.legend(loc="upper right", fontsize=5, handletextpad=0.3,
             frameon=True, facecolor='white', edgecolor=GRAY_LIGHT,
             framealpha=0.85)

    fig.text(
        0.055, 0.966,
        "Conformal intervals are well-calibrated in-distribution but "
        "under-cover for unseen perturbations",
        fontsize=6.5, va="top", ha="left", color=BLACK,
    )

    finalize(fig, "fig_overview", (WIDTH_MM, 135), use_tight=False)


# ═══════════════════════════════════════════════════════════════════════════

def main():
    apply_pub_style()
    fig_s1()
    fig_s2()
    fig_s3()
    fig_s3b()
    fig_s4()
    fig_s5()
    fig_baselines()
    fig_scgpt()
    overview()
    print(f"[OK] All figures written to {OUT}", flush=True)


if __name__ == "__main__":
    main()
