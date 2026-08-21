"""Generate fig_s3b for the UQ-CP manuscript."""

import numpy as np
import matplotlib.pyplot as plt
from plot_style import *

def fig_s3b():
    """Coverage vs calibration-fraction sensitivity."""
    fig, ax = plt.subplots()
    fracs = [25, 10, 5]
    cov  = [0.9522, 0.9303, 0.9275]
    sem  = [0.0045, 0.0079, 0.0071]

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
    finalize(fig, "fig_s3b", (82, 60))


if __name__ == "__main__":
    apply_pub_style()
    fig_s3b()
