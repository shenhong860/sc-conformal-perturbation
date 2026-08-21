"""Generate fig_s3 for the UQ-CP manuscript."""

import numpy as np
import matplotlib.pyplot as plt
from plot_style import *

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

    # Footnote in lower-LEFT, away from the nominal label
    ax.text(0.02, 0.04, "Mean \u00b1 95% CI (5 seeds)",
            transform=ax.transAxes, fontsize=5.5, va="bottom", ha="left",
            color=BLACK, style="italic",
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8,
                      pad=1.2))
    ax.set_ylabel("Coverage")
    finalize(fig, "fig_s3", (82, 60))


if __name__ == "__main__":
    apply_pub_style()
    fig_s3()
