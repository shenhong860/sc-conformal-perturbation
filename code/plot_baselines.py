"""Generate fig_baselines for the UQ-CP manuscript."""

import numpy as np
import matplotlib.pyplot as plt
from plot_style import *

def fig_baselines():
    """UQ method baselines: in-distribution vs unseen-drug coverage."""
    fig, axes = plt.subplots(1, 2, figsize=(155 / 25.4, 65 / 25.4))

    methods = ["Fixed\nnormal", "Bootstrap\nSE", "No shrink", "Conformal\n(ours)"]
    colors  = [GRAY, GOLD, RED, BLUE]

    s2_vals = [0.8553, 0.7905, 0.9392, 0.9511]
    s2_errs = [0.0045, 0.0082, 0.0030, 0.0043]
    s3_vals = [0.7687, 0.6747, 0.9130, 0.8658]
    s3_errs = [0.0046, 0.0085, 0.0030, 0.0032]

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

    panel_label(axes[0], "A", x=-0.08, y=1.05)
    panel_label(axes[1], "B", x=-0.08, y=1.05)
    finalize(fig, "fig_baselines", (155, 68))


if __name__ == "__main__":
    apply_pub_style()
    fig_baselines()
