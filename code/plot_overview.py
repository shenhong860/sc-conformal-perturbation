"""Generate overview for the UQ-CP manuscript."""

import numpy as np
import matplotlib.pyplot as plt
from plot_style import *

from scipy import stats

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
    a.errorbar(0, 0.947, yerr=[[0.011], [0.011]], fmt="none",
               ecolor=BLACK, elinewidth=0.7, capsize=2.5, capthick=0.7, zorder=4)
    a.text(0, 0.958 + 0.005, "0.947", ha="center", va="bottom",
           fontsize=6, fontweight="bold", color=BLACK)
    nominal_line(a, label=False, ymin=0.86, ymax=0.97)
    subtle_grid(a)
    a.set_xticks([0])
    a.set_xticklabels(["Synthetic"], fontsize=5.5)
    a.set_ylabel("Coverage", fontsize=6.5)

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
        b.text(xb[i] - wb/2, nv + 0.006, f"{nv:.3f}", ha="center",
               fontsize=5.5, color=BLACK)
        b.text(xb[i] + wb/2, cv + 0.012, f"{cv:.3f}", ha="center",
               fontsize=5.5, color=BLACK)
    nominal_line(b, label=False, ymin=0.86, ymax=0.99)
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
               fontsize=5.5, color=BLACK)
    nominal_line(c, label=False, ymin=0.78, ymax=0.97)
    subtle_grid(c)
    c.set_xticks(xb)
    c.set_xticklabels(CELLS, fontsize=5.5)
    c.set_yticks([0.80, 0.85, 0.90, 0.95])

    # ── Panel d: Calibration fraction ──
    fracs_d = [25, 10, 5]
    cov_d  = [0.9522, 0.9303, 0.9275]
    sem_d  = [0.0045, 0.0079, 0.0071]
    d.fill_between(fracs_d, [c-s for c,s in zip(cov_d,sem_d)],
                   [c+s for c,s in zip(cov_d,sem_d)],
                   color=BLUE, alpha=0.10, zorder=1)
    d.errorbar(fracs_d, cov_d, yerr=sem_d, fmt="none", ecolor=BLACK,
               elinewidth=0.6, capsize=2, capthick=0.6, zorder=2)
    d.plot(fracs_d, cov_d, "-o", color=BLUE, lw=1.2, ms=4, zorder=3,
           markeredgecolor=WHITE, markeredgewidth=0.5)
    for fx, cy in zip(fracs_d, cov_d):
        d.text(fx, cy + 0.005, f"{cy:.3f}", ha="center",
               fontsize=5.5, color=BLACK)
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
    errs_e  = [0.0086, 0.0128, 0.0146, 0.0036]
    e.bar(np.arange(4), vals_e, width=0.58,
          color=[GRAY, TEAL, BLUE, GOLD], edgecolor=BLACK,
          linewidth=0.5, zorder=3, yerr=errs_e,
          error_kw=dict(elinewidth=0.6, capsize=2, capthick=0.6, ecolor=BLACK))
    for i, (v, err) in enumerate(zip(vals_e, errs_e)):
        e.text(i, v + err + 0.004, f"{v:.3f}", ha="center",
               fontsize=5.5, color=BLACK)
    e.set_ylim(0, 0.84)
    e.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    e.set_xticks(np.arange(4))
    e.set_xticklabels(labels_e, fontsize=5.5)

    # ── Panel f: Norman ──
    mean_f = float(np.mean(S5_PER_PERT))
    se_f   = float(np.std(S5_PER_PERT, ddof=1) / np.sqrt(len(S5_PER_PERT)))
    t_f    = float(stats.t.ppf(0.975, len(S5_PER_PERT) - 1))
    lo_f, hi_f = mean_f - t_f*se_f, mean_f + t_f*se_f
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
    panel_label(ax_table, "G", x=-0.04, y=1.12, fontsize=8)

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
             ha="center", va="bottom", fontsize=5.5, color=BLACK,
             fontstyle="italic")

    # ── Shared panel labels & nominal legend ──
    for i, (ax_i, lbl) in enumerate(zip(axes, "ABCDEF")):
        panel_label(ax_i, lbl, x=-0.10, y=1.12, fontsize=8)

    f.plot([], [], color=GRAY, ls="--", lw=0.8, label="Nominal 0.95")
    f.legend(loc="upper right", fontsize=5.5, handletextpad=0.3,
             frameon=True, facecolor='white', edgecolor=GRAY_LIGHT,
             framealpha=0.85)

    fig.text(
        0.055, 0.966,
        "Conformal intervals are well-calibrated in-distribution but "
        "under-cover for unseen perturbations",
        fontsize=6.5, va="top", ha="left", color=BLACK,
    )

    finalize(fig, "fig_overview", (WIDTH_MM, 135), use_tight=False)


if __name__ == "__main__":
    apply_pub_style()
    overview()
