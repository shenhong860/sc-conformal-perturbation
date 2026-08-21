"""Generate fig_s1 for the UQ-CP manuscript."""

import numpy as np
import matplotlib.pyplot as plt
from plot_style import *

def fig_s1():
    """Synthetic validation: bar + per-seed swarm strip."""
    fig, ax = plt.subplots()
    mean, lo, hi = 0.947, 0.936, 0.958

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
            fontsize=6.5, fontweight="bold", color=BLACK)
    ax.text(0, 0.905, f"95% CI [{lo:.3f}, {hi:.3f}]", ha="center",
            va="center", fontsize=5.5, color=WHITE, fontweight="bold")

    nominal_line(ax, ymin=0.86, ymax=1.0)
    subtle_grid(ax)
    ax.set_ylabel("Coverage")
    ax.set_xticks([0])
    ax.set_xticklabels(["Synthetic\n(5 seeds)"])
    ax.set_title("Method validation on synthetic data", loc="left", fontsize=6.5,
                 color=BLACK, pad=4)
    finalize(fig, "fig_s1", (80, 60))


if __name__ == "__main__":
    apply_pub_style()
    fig_s1()
