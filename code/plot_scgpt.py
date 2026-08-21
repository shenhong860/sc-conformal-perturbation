"""Generate fig_scgpt for the UQ-CP manuscript."""

import numpy as np
import matplotlib.pyplot as plt
from plot_style import *

from scipy import stats
import json
from pathlib import Path

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
    delta_ci = float(stats.t.ppf(0.975, len(delta_conds) - 1)
                     * np.std(delta_conds, ddof=1) / np.sqrt(len(delta_conds)))

    seeds = [mseed["per_seed"][str(s)]["coverage"] for s in mseed["seeds"]]
    seed_mean = mseed["coverage_mean"]
    seed_ci = float(stats.t.ppf(0.975, len(seeds) - 1)
                    * np.std(seeds, ddof=1) / np.sqrt(len(seeds)))
    ens_cov = mseed["ensemble"]["coverage"]

    x = np.arange(3)
    w = 0.42

    b1 = ax.bar(0, delta_mean, width=w, color=RED,
                edgecolor=BLACK, linewidth=0.5, zorder=3)
    ax.errorbar(0, delta_mean, yerr=delta_ci, fmt="none", ecolor=BLACK,
                elinewidth=0.7, capsize=2.5, capthick=0.7, zorder=4)
    ax.text(0, delta_mean + delta_ci + 0.006, f"{delta_mean:.3f}",
            ha="center", va="bottom", fontsize=5.5, color=BLACK)

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
            ha="center", va="bottom", fontsize=5.5, color=BLACK)
    ax.text(1, 0.805, f"per-seed {min(seeds):.3f}\u2013{max(seeds):.3f}",
            ha="center", va="top", fontsize=5.5, color=BLUE_DARK)

    b3 = ax.bar(2, ens_cov, width=w, color=GOLD,
                edgecolor=BLACK, linewidth=0.5, zorder=3)
    ax.text(2, ens_cov + 0.006, f"{ens_cov:.3f}",
            ha="center", va="bottom", fontsize=5.5, color=BLACK)

    nominal_line(ax, label=False, ymin=0.78, ymax=1.02)
    ax.text(0.985, 0.975, "nominal 0.95", transform=ax.transAxes,
            ha="right", va="top", fontsize=5.5, color=BLACK, style="italic")
    subtle_grid(ax)

    ax.set_xticks(x)
    ax.set_xticklabels(["Mean-shift (\u03b4)", "scGPT fine-tune\n(5 seeds)",
                        "scGPT ensemble"], fontsize=5.5)
    ax.set_ylabel("Coverage")
    finalize(fig, "fig_scgpt", (160, 78))


# ═══════════════════════════════════════════════════════════════════════════
# Overview: 2x3 + ablation table


if __name__ == "__main__":
    apply_pub_style()
    fig_scgpt()
