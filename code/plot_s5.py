"""Generate fig_s5 for the UQ-CP manuscript."""

import numpy as np
import matplotlib.pyplot as plt
from plot_style import *

from scipy import stats

def fig_s5():
    """Norman genetic perturbation: bar + per-perturbation swarm."""
    fig, ax = plt.subplots()

    mean5 = float(np.mean(S5_PER_PERT))
    se5   = float(np.std(S5_PER_PERT, ddof=1) / np.sqrt(len(S5_PER_PERT)))
    t9    = float(stats.t.ppf(0.975, len(S5_PER_PERT) - 1))
    lo5, hi5 = mean5 - t9 * se5, mean5 + t9 * se5

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
    finalize(fig, "fig_s5", (82, 60))


if __name__ == "__main__":
    apply_pub_style()
    fig_s5()
