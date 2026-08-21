"""Regenerate all UQ-CP publication figures.

Individual figures can be regenerated with the plot_*.py scripts; this
entry point runs the full set and writes SVG/PDF/PNG/TIFF into the
figures output directory.
"""
from plot_style import OUT, apply_pub_style
from plot_s1 import fig_s1
from plot_s2 import fig_s2
from plot_s3 import fig_s3
from plot_s3b import fig_s3b
from plot_s4 import fig_s4
from plot_s5 import fig_s5
from plot_baselines import fig_baselines
from plot_scgpt import fig_scgpt
from plot_overview import overview


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
    print(f"[OK] All figures written to {OUT}", flush=True)


if __name__ == "__main__":
    main()

