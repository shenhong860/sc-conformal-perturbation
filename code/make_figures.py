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
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---- MANDATORY editable-text settings (must precede figure creation) ------
# svg.fonttype='none'  (editable <text> nodes in SVG)
# pdf.fonttype=42      (editable TrueType text in PDF)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42

OUT = Path(r"D:\guo\CW1OT\results\figures")
OUT.mkdir(parents=True, exist_ok=True)

NOMINAL = 0.95
DPI_RASTER = 600
WIDTH_MM = 183  # double-column journal width; individual panels are sub-widths

# Low-saturation palette; blue = calibrated/primary, red = naive, teal = use-case.
BLUE = "#3775BA"
BLUE_DARK = "#0F4D92"
RED = "#B64342"
TEAL = "#42949E"
GOLD = "#C4A24E"
GRAY = "#8F8F8F"
GRAY_LIGHT = "#D8D8D8"
BLACK = "#272727"

CELLS = ["A549", "MCF7", "K562"]


def apply_pub_style(font_size=7, axes_lw=0.8):
    mpl.rcParams.update(
        {
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
        }
    )


def finalize(fig, name, figsize_mm, dpi=DPI_RASTER, use_tight=True):
    """Save one figure in SVG/PDF (vector) and PNG/TIFF (600 dpi raster)."""
    fig.set_size_inches(figsize_mm[0] / 25.4, figsize_mm[1] / 25.4)
    if use_tight:
        fig.tight_layout(pad=1.2)
    base = OUT / name
    fig.savefig(f"{base}.svg", bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    # dpi=600 raster exports (PNG preview and TIFF submission raster)
    fig.savefig(f"{base}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{base}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def panel_label(ax, label, x=-0.10, y=1.04, fontsize=8):
    ax.text(
        x, y, label, transform=ax.transAxes, fontsize=fontsize,
        fontweight="bold", ha="left", va="bottom", color=BLACK,
    )


def nominal_line(ax, label=True, ymin=0.72, ymax=1.0):
    ax.axhline(NOMINAL, color=GRAY, ls="--", lw=0.9, zorder=1)
    if label:
        ax.text(
            1.0, NOMINAL, " nominal 0.95", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=5.5, color=GRAY,
        )
    ax.set_ylim(ymin, ymax)


def label_bars(ax, bars, values, spread=None, fmt="{:.3f}", dy=0.004,
               fontsize=6):
    for i, (bar, value) in enumerate(zip(bars, values)):
        upper = bar.get_height()
        if spread is not None:
            upper = upper + spread[i]
        ax.text(
            bar.get_x() + bar.get_width() / 2, upper + dy, fmt.format(value),
            ha="center", va="bottom", fontsize=fontsize, color=BLACK,
        )


def fig_s1():
    """Synthetic: mean coverage 0.947, 95% CI [0.938, 0.956] over 5 seeds."""
    fig, ax = plt.subplots()
    mean, lo, hi = 0.947, 0.938, 0.956
    bars = ax.bar(
        ["Synthetic\n(5 seeds)"], [mean], width=0.5, color=BLUE, edgecolor=BLACK,
        linewidth=0.7, zorder=3,
    )
    ax.errorbar(
        0, mean, yerr=[[mean - lo], [hi - mean]], fmt="none", ecolor=BLACK,
        elinewidth=0.9, capsize=4, capthick=0.9, zorder=4,
    )
    label_bars(ax, bars, [mean], fmt="{:.3f}")
    nominal_line(ax, ymin=0.86, ymax=1.0)
    ax.text(
        0, hi - 0.030, "95% CI [0.938, 0.956]", ha="center", va="top",
        fontsize=6, color=BLACK,
    )
    ax.set_ylabel("Coverage")
    ax.set_xticks([0])
    ax.set_xticklabels(["Synthetic\n(5 seeds)"])
    panel_label(ax, "a")
    finalize(fig, "fig_s1", (86, 62))


def fig_s2():
    """In-distribution calibration by cell line (naive vs calibrated)."""
    fig, ax = plt.subplots()
    x = np.arange(len(CELLS))
    naive = [0.9164, 0.9438, 0.9394]
    cal = [0.9513, 0.9498, 0.9527]
    naive_sem = [0.0042, 0.0038, 0.0051]
    cal_ci = [0.0032, 0.0026, 0.0077]
    w = 0.34
    b1 = ax.bar(
        x - w / 2, naive, w, label="Naive", color=RED, edgecolor=BLACK,
        linewidth=0.7, zorder=3,
        yerr=naive_sem, error_kw=dict(elinewidth=0.7, capsize=2.5, capthick=0.7,
                                      ecolor=BLACK),
    )
    b2 = ax.bar(
        x + w / 2, cal, w, label="Calibrated", color=BLUE, edgecolor=BLACK,
        linewidth=0.7, zorder=3,
        yerr=cal_ci, error_kw=dict(elinewidth=0.7, capsize=2.5, capthick=0.7,
                                   ecolor=BLACK),
    )
    label_bars(ax, b1, naive, spread=naive_sem, fmt="{:.3f}", dy=0.004)
    label_bars(ax, b2, cal, spread=cal_ci, fmt="{:.3f}", dy=0.004)
    nominal_line(ax, ymin=0.84, ymax=0.985)
    ax.set_xticks(x)
    ax.set_xticklabels(CELLS)
    ax.set_ylabel("Coverage")
    ax.legend(loc="lower right", fontsize=6)
    ax.text(
        0.02, 0.025, "Bars: mean over drug-by-cell-line conditions; "
        "error bars: naive SE / calibrated 95% CI over 5 seeds",
        transform=ax.transAxes, fontsize=5.5, va="bottom", ha="left",
        color=BLACK,
    )
    panel_label(ax, "b")
    finalize(fig, "fig_s2", (92, 62))


def fig_s3():
    """Leave-one-drug-out: unseen perturbations under-cover."""
    fig, ax = plt.subplots()
    vals = [0.8534, 0.8429, 0.9094]
    errs = [0.0028, 0.0024, 0.0069]
    bars = ax.bar(
        CELLS, vals, width=0.55, color=[BLUE_DARK, BLUE, TEAL],
        edgecolor=BLACK, linewidth=0.7, zorder=3, yerr=errs,
        error_kw=dict(elinewidth=0.8, capsize=3, capthick=0.8, ecolor=BLACK),
    )
    label_bars(ax, bars, vals, spread=errs, fmt="{:.3f}", dy=0.004)
    nominal_line(ax, ymin=0.78, ymax=0.975)
    ax.text(
        0.02, 0.025, "Mean +/- 95% CI over five seeds",
        transform=ax.transAxes, fontsize=5.5, va="bottom", ha="left",
        color=BLACK,
    )
    ax.set_ylabel("Coverage")
    panel_label(ax, "c")
    finalize(fig, "fig_s3", (86, 62))


def fig_s3b():
    """Coverage versus calibration fraction."""
    fig, ax = plt.subplots()
    fracs = [25, 10, 5]
    cov = [0.9486, 0.9335, 0.9275]
    cell_vals = {
        25: [0.9485, 0.9497, 0.9474],
        10: [0.9454, 0.9411, 0.9100],
        5: [0.9354, 0.9506, 0.8887],
    }
    sem = [np.std(cell_vals[fx], ddof=1) / np.sqrt(3) for fx in fracs]
    ax.errorbar(
        fracs, cov, yerr=sem, fmt="none", ecolor=BLACK, elinewidth=0.8,
        capsize=3, capthick=0.8, zorder=2,
    )
    ax.plot(fracs, cov, "-o", color=BLUE, lw=1.4, ms=4.5, zorder=3)
    for fx, cy in zip(fracs, cov):
        ax.text(
            fx, cy + 0.006, f"{cy:.3f}", ha="center", va="bottom",
            fontsize=6, color=BLACK,
        )
    nominal_line(ax, ymin=0.86, ymax=0.98)
    ax.set_xlabel("Calibration fraction (%)")
    ax.set_ylabel("Coverage")
    ax.set_xticks(fracs)
    ax.set_xticklabels(["25", "10", "5"])
    ax.set_xlim(27, 3)
    ax.text(
        0.02, 0.025, "Error bars: SE across three cell lines",
        transform=ax.transAxes, fontsize=5.5, va="bottom", ha="left",
        color=BLACK,
    )
    panel_label(ax, "d", x=-0.08, y=1.09)
    finalize(fig, "fig_s3b", (86, 62))


def fig_s4():
    """DEG use case: interval filtering improves top-50 precision."""
    fig, ax = plt.subplots()
    labels = ["Top-50\nraw", "Top-50\nfiltered", "Recall", "Flagged\nfraction"]
    vals = [0.6570, 0.6699, 0.6800, 0.0955]
    errs = [0.0061, 0.0091, 0.0103, 0.0026]
    colors = [GRAY, TEAL, BLUE, GOLD]
    bars = ax.bar(
        labels, vals, color=colors, edgecolor=BLACK, linewidth=0.7, zorder=3,
        yerr=errs, error_kw=dict(elinewidth=0.8, capsize=3, capthick=0.8,
                                 ecolor=BLACK),
    )
    label_bars(ax, bars, vals, spread=errs, fmt="{:.3f}", dy=0.006)
    ax.set_ylim(0, 0.84)
    ax.text(
        0.02, 0.025, "Mean +/- 95% CI over five seeds; filtered = flagged top-50",
        transform=ax.transAxes, fontsize=5.5, va="bottom", ha="left",
        color=BLACK,
    )
    ax.set_ylabel("Value")
    panel_label(ax, "e")
    finalize(fig, "fig_s4", (92, 62))


def fig_s5():
    """Norman genetic perturbation screen: overall coverage 0.909."""
    fig, ax = plt.subplots()
    pert_cov = [
        0.909, 0.889, 0.906, 0.911, 0.919, 0.922, 0.903, 0.916, 0.907, 0.908,
    ]
    mean5 = float(np.mean(pert_cov))
    se5 = float(np.std(pert_cov, ddof=1) / np.sqrt(len(pert_cov)))
    lo5, hi5 = mean5 - 1.96 * se5, mean5 + 1.96 * se5
    bars = ax.bar(
        ["Norman\n(K562)"], [mean5], width=0.5, color=TEAL, edgecolor=BLACK,
        linewidth=0.7, zorder=3, yerr=[[mean5 - lo5], [hi5 - mean5]],
        error_kw=dict(elinewidth=0.8, capsize=3, capthick=0.8, ecolor=BLACK),
    )
    label_bars(ax, bars, [mean5], spread=[hi5 - mean5], fmt="{:.3f}", dy=0.004)
    nominal_line(ax, ymin=0.80, ymax=0.98)
    ax.text(
        0, mean5 - 0.030, f"95% CI [{lo5:.3f}, {hi5:.3f}]", ha="center",
        va="top", fontsize=6, color=BLACK,
    )
    ax.set_ylabel("Coverage")
    panel_label(ax, "f")
    finalize(fig, "fig_s5", (86, 62))


def fig_baselines():
    """UQ baseline comparison: coverage in-distribution and for unseen drugs."""
    fig, axes = plt.subplots(1, 2, figsize=(150 / 25.4, 68 / 25.4))
    methods = ["Fixed\nnormal", "Bootstrap\nSE", "No\nshrink", "Conformal\n(ours)"]
    colors = [GRAY, GOLD, RED, BLUE]
    s2 = [0.8553, 0.7905, 0.9392, 0.9511]
    s2_err = [0.0032, 0.0058, 0.0021, 0.0030]
    s3 = [0.7687, 0.6747, 0.9130, 0.8658]
    s3_err = [0.0033, 0.0060, 0.0021, 0.0022]
    for ax, vals, errs, label, ymin in (
        (axes[0], s2, s2_err, "In-distribution", 0.70),
        (axes[1], s3, s3_err, "Unseen drug", 0.60),
    ):
        bars = ax.bar(
            np.arange(4), vals, width=0.62, color=colors, edgecolor=BLACK,
            linewidth=0.7, zorder=3, yerr=errs,
            error_kw=dict(elinewidth=0.8, capsize=3, capthick=0.8, ecolor=BLACK),
        )
        label_bars(ax, bars, vals, spread=errs, fmt="{:.3f}", dy=0.005,
                   fontsize=5.5)
        nominal_line(ax, label=False, ymin=ymin, ymax=1.0)
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(methods, fontsize=6)
        ax.set_ylabel("Coverage")
        ax.set_title(label, loc="left", fontsize=7)
    panel_label(axes[0], "a", x=-0.08, y=1.05)
    panel_label(axes[1], "b", x=-0.08, y=1.05)
    finalize(fig, "fig_baselines", (150, 70))


def fig_scgpt():
    """Per-condition unseen-drug coverage: delta vs fine-tuned scGPT."""
    fig, ax = plt.subplots()
    conds = [
        "CEBPE+\nRUNX1T1", "DUSP9+\nETS2", "ETS2+\nCNN1", "LHX1+\nELMSAN1",
        "LYL1+\nIER5L", "SET+\nCEBPE", "SET+\nKLF1", "TBX3+\nTBX2",
        "UBASH3B+\nOSR2", "ZC3HAV1+\nHOXC13",
    ]
    delta = [0.9050, 0.9079, 0.9070, 0.9089, 0.9128, 0.9138, 0.9215, 0.9070,
             0.9167, 0.9041]
    scgpt = [1.0000, 0.9118, 0.9070, 0.9651, 0.9583, 0.9486, 0.9254, 0.9942,
             0.9661, 0.9060]
    x = np.arange(len(conds))
    w = 0.36
    b1 = ax.bar(x - w / 2, delta, w, label="Mean-shift (delta)", color=RED,
                edgecolor=BLACK, linewidth=0.7, zorder=3)
    b2 = ax.bar(x + w / 2, scgpt, w, label="Fine-tuned scGPT", color=BLUE,
                edgecolor=BLACK, linewidth=0.7, zorder=3)
    label_bars(ax, b1, delta, fmt="{:.2f}", dy=0.004, fontsize=5)
    label_bars(ax, b2, scgpt, fmt="{:.2f}", dy=0.004, fontsize=5)
    nominal_line(ax, label=True, ymin=0.84, ymax=1.04)
    ax.set_xticks(x)
    ax.set_xticklabels(conds, fontsize=5.5, rotation=0)
    ax.set_ylabel("Coverage")
    ax.legend(loc="lower right", fontsize=6)
    ax.text(
        0.02, 0.03,
        "Overall: delta 0.910 +/- 0.004, scGPT 0.948 +/- 0.022 "
        "(mean +/- 95% CI over 10 conditions)",
        transform=ax.transAxes, fontsize=5.5, va="bottom", ha="left",
        color=BLACK,
    )
    finalize(fig, "fig_scgpt", (150, 76))


def overview():
    """2x3 evidence panels plus base-predictor ablation table."""
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, 140 / 25.4))
    gs = fig.add_gridspec(3, 3, hspace=0.65, wspace=0.34,
                          left=0.055, right=0.975, top=0.90, bottom=0.10,
                          height_ratios=[1, 1, 0.55])
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis("off")

    a, b, c, d, e, f = axes

    # a: synthetic
    mean, lo, hi = 0.947, 0.938, 0.956
    bars = a.bar([0], [mean], width=0.55, color=BLUE, edgecolor=BLACK,
                 linewidth=0.7, zorder=3)
    a.errorbar(0, mean, yerr=[[mean - lo], [hi - mean]], fmt="none",
               ecolor=BLACK, elinewidth=0.9, capsize=4, capthick=0.9, zorder=4)
    a.text(0, hi + 0.006, "0.947", ha="center", va="bottom", fontsize=6.5)
    nominal_line(a, label=False, ymin=0.86, ymax=0.975)
    a.set_xticks([0])
    a.set_xticklabels(["Synthetic\n(5 seeds)"])
    a.set_ylabel("Coverage")

    # b: calibration
    x = np.arange(3)
    naive = [0.9164, 0.9438, 0.9394]
    cal = [0.9513, 0.9498, 0.9527]
    naive_sem = [0.0042, 0.0038, 0.0051]
    cal_ci = [0.0032, 0.0026, 0.0077]
    w = 0.34
    b1 = b.bar(x - w / 2, naive, w, label="Naive", color=RED,
               edgecolor=BLACK, linewidth=0.7, zorder=3,
               yerr=naive_sem, error_kw=dict(elinewidth=0.7, capsize=2.5,
                                             capthick=0.7, ecolor=BLACK))
    b2 = b.bar(x + w / 2, cal, w, label="Calibrated", color=BLUE,
               edgecolor=BLACK, linewidth=0.7, zorder=3,
               yerr=cal_ci, error_kw=dict(elinewidth=0.7, capsize=2.5,
                                          capthick=0.7, ecolor=BLACK))
    for bars_, vals_, spread_ in ((b1, naive, naive_sem),
                                  (b2, cal, cal_ci)):
        for bar, val, sp in zip(bars_, vals_, spread_):
            b.text(bar.get_x() + bar.get_width() / 2, val + sp + 0.004,
                   f"{val:.3f}", ha="center", va="bottom", fontsize=5.5)
    nominal_line(b, label=False, ymin=0.86, ymax=0.975)
    b.set_xticks(x)
    b.set_xticklabels(CELLS)

    # c: unseen drugs
    vals_c = [0.8534, 0.8429, 0.9094]
    errs_c = [0.0028, 0.0024, 0.0069]
    bars_c = c.bar(x, vals_c, width=0.55,
                   color=[BLUE_DARK, BLUE, TEAL], edgecolor=BLACK,
                   linewidth=0.7, zorder=3, yerr=errs_c,
                   error_kw=dict(elinewidth=0.8, capsize=3, capthick=0.8,
                                 ecolor=BLACK))
    for bar, val, err in zip(bars_c, vals_c, errs_c):
        c.text(bar.get_x() + bar.get_width() / 2, val + err + 0.005,
               f"{val:.3f}", ha="center", va="bottom", fontsize=5.5)
    nominal_line(c, label=False, ymin=0.78, ymax=0.975)
    c.set_xticks(x)
    c.set_xticklabels(CELLS)
    c.set_yticks([0.80, 0.85, 0.90, 0.95])

    # d: calibration fraction
    fracs = [25, 10, 5]
    cov = [0.9486, 0.9335, 0.9275]
    cell_vals = {
        25: [0.9485, 0.9497, 0.9474],
        10: [0.9454, 0.9411, 0.9100],
        5: [0.9354, 0.9506, 0.8887],
    }
    sem_d = [np.std(cell_vals[fx], ddof=1) / np.sqrt(3) for fx in fracs]
    d.errorbar(fracs, cov, yerr=sem_d, fmt="none", ecolor=BLACK,
               elinewidth=0.8, capsize=3, capthick=0.8, zorder=2)
    d.plot(fracs, cov, "-o", color=BLUE, lw=1.4, ms=4, zorder=3)
    for fx, cy in zip(fracs, cov):
        d.text(fx, cy + 0.007, f"{cy:.3f}", ha="center", va="bottom",
               fontsize=5.5)
    nominal_line(d, label=False, ymin=0.86, ymax=0.985)
    d.set_xlabel("Calibration fraction (%)")
    d.set_xticks(fracs)
    d.set_xticklabels(["25", "10", "5"])
    d.set_xlim(27, 3)
    d.set_yticks([0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98])

    # e: DEG use case
    labels_e = ["Top-50\nraw", "Top-50\nfiltered", "Recall", "Flagged\nfraction"]
    vals_e = [0.6570, 0.6699, 0.6800, 0.0955]
    errs_e = [0.0061, 0.0091, 0.0103, 0.0026]
    bars_e = e.bar(np.arange(4), vals_e, width=0.62,
                   color=[GRAY, TEAL, BLUE, GOLD], edgecolor=BLACK,
                   linewidth=0.7, zorder=3, yerr=errs_e,
                   error_kw=dict(elinewidth=0.8, capsize=3, capthick=0.8,
                                 ecolor=BLACK))
    for bar, val, err in zip(bars_e, vals_e, errs_e):
        e.text(bar.get_x() + bar.get_width() / 2, val + err + 0.008,
               f"{val:.3f}", ha="center", va="bottom", fontsize=5.5)
    e.set_ylim(0, 0.84)
    e.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    e.set_xticks(np.arange(4))
    e.set_xticklabels(["Top-50\nraw", "Top-50\nfiltered", "Recall",
                       "Flagged\nfraction"])

    # f: Norman
    pert_cov = [
        0.909, 0.889, 0.906, 0.911, 0.919, 0.922, 0.903, 0.916, 0.907, 0.908,
    ]
    mean_f = float(np.mean(pert_cov))
    se_f = float(np.std(pert_cov, ddof=1) / np.sqrt(len(pert_cov)))
    lo_f, hi_f = mean_f - 1.96 * se_f, mean_f + 1.96 * se_f
    bars_f = f.bar([0], [mean_f], width=0.55, color=TEAL, edgecolor=BLACK,
                   linewidth=0.7, zorder=3,
                   yerr=[[mean_f - lo_f], [hi_f - mean_f]],
                   error_kw=dict(elinewidth=0.8, capsize=3, capthick=0.8,
                                 ecolor=BLACK))
    f.text(0, hi_f + 0.008, "0.909", ha="center", va="bottom", fontsize=6.5)
    nominal_line(f, label=False, ymin=0.78, ymax=0.98)
    f.set_xticks([0])
    f.set_xticklabels(["Norman\n(K562)"])

    # Ablation table panel (g): delta vs w1ot base predictor.
    panel_label(ax_table, "g", x=-0.045, y=1.10, fontsize=8)
    table_rows = [
        ["", "Coverage", "Median width", "Precision@50"],
        ["delta", "0.943", "0.029", "0.777"],
        ["w1ot", "0.948", "0.056", "0.717"],
    ]
    tbl = ax_table.table(
        cellText=table_rows, loc="center", colWidths=[0.18, 0.2, 0.2, 0.2],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6.5)
    tbl.scale(1, 1.5)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(BLACK)
        cell.set_linewidth(0.6)
        cell.set_text_props(color=BLACK, ha="center")
        if row == 0:
            cell.set_facecolor(GRAY_LIGHT)
        elif col == 0:
            cell.set_facecolor("#EEF2F7")
    ax_table.text(
        0.5, 0.98, "Base-predictor ablation (six sci-Plex3 conditions)",
        transform=ax_table.transAxes, ha="center", va="top", fontsize=6.5,
        color=BLACK,
    )

    for i, (ax, label) in enumerate(zip(axes, "abcdef")):
        panel_label(ax, label, x=-0.10, y=1.13, fontsize=8)
        if i == 0:
            ax.set_ylabel("Coverage")
        if i == 1:
            ax.set_ylabel("Coverage")

    # Shared nominal legend line in panel f.
    f.plot([], [], color=GRAY, ls="--", lw=0.9, label="Nominal 0.95")
    f.legend(loc="lower right", fontsize=6)
    fig.text(
        0.055, 0.845, "Conformal intervals are calibrated in-distribution "
        "but under-cover for unseen perturbations and small calibration sets",
        fontsize=7, va="top", ha="left", color=BLACK,
    )

    finalize(fig, "fig_overview", (WIDTH_MM, 140), use_tight=False)


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
    print(f"figures written to {OUT}", flush=True)


if __name__ == "__main__":
    main()
