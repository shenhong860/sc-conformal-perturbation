"""Generate fig_s4 for the UQ-CP manuscript."""

import numpy as np
import matplotlib.pyplot as plt
from plot_style import *

def fig_s4():
    """DEG prioritization use case: interval filtering."""
    fig, ax = plt.subplots()
    labels = ["Top-50\nraw", "Top-50\nfiltered", "Recall", "Flagged\nfraction"]
    vals  = [0.6570, 0.6699, 0.6800, 0.0955]
    errs  = [0.0086, 0.0128, 0.0146, 0.0036]
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
            transform=ax.transAxes, fontsize=5.5, va="top", ha="left",
            color=BLACK, style="italic",
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.85,
                      pad=1.5))
    finalize(fig, "fig_s4", (90, 62))


if __name__ == "__main__":
    apply_pub_style()
    fig_s4()
