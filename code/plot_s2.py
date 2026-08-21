"""Generate fig_s2 for the UQ-CP manuscript."""

import numpy as np
import matplotlib.pyplot as plt
from plot_style import *

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

    finalize(fig, "fig_s2", (88, 60))


if __name__ == "__main__":
    apply_pub_style()
    fig_s2()
