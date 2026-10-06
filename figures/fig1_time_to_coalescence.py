#!/usr/bin/env python3
"""Fig. 1: time to coalescence as a function of chirp mass and GW frequency, Eq. (3).

    python figures/fig1_time_to_coalescence.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import _setup  # noqa: F401  (adds src/ to sys.path)
from pbh_viterbi.paths import PLOTS_DIR

G = 6.674e-11  # m^3 kg^-1 s^-2
C = 3e8  # m s^-1
M_SUN = 1.989e30  # kg
YEAR = 3.156e7  # s

FONT_SIZE = 22
CONTOUR_COLOR = "#2d4a3e"
# Contours at 1 hour, 1 day and 1 year, with the (f, Mc) positions of their labels.
CONTOURS_YR = {(1 / 24) / 365.25: "1 hr", 1 / 365.25: "1 d", 1.0: "1 yr"}
LABEL_POSITIONS = [(300, 1.5e-3), (270, 1e-3), (175, 6e-5)]


def time_to_coalescence_yr(f_gw, mchirp):
    """Eq. (3): (5/256) (G Mc / c^3)^(-5/3) (pi f)^(-8/3), in years."""
    return (5 / 256) * (G * mchirp * M_SUN / C ** 3) ** (-5 / 3) * (np.pi * f_gw) ** (-8 / 3) / YEAR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PLOTS_DIR / "fig1_time_to_coalescence.png")
    args = parser.parse_args()

    f_gw = np.logspace(1, 3, 500)
    mchirp = np.logspace(-6, -1, 500)
    t_coal = time_to_coalescence_yr(*np.meshgrid(f_gw, mchirp))

    plt.rcParams.update({"font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE,
                         "xtick.labelsize": FONT_SIZE, "ytick.labelsize": FONT_SIZE})
    fig, ax = plt.subplots(figsize=(11.6, 6.5))
    mesh = ax.pcolormesh(f_gw, mchirp, t_coal, norm=LogNorm(vmin=1e-7, vmax=1e5), cmap="GnBu", shading="auto")
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("Time to coalescence [yr]", labelpad=10, fontsize=FONT_SIZE)
    cbar.ax.tick_params(labelsize=FONT_SIZE)
    cbar.set_ticks([1e-7, 1e-5, 1e-3, 1e-1, 1e1, 1e3, 1e5])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$f_{\rm gw}$ [Hz]", fontsize=FONT_SIZE)
    ax.set_ylabel(r"$\mathcal{M}\ [M_\odot]$", fontsize=FONT_SIZE)

    contours = ax.contour(f_gw, mchirp, t_coal, levels=list(CONTOURS_YR), colors=CONTOUR_COLOR,
                          linewidths=1.5, linestyles="--")
    ax.clabel(contours, fmt=CONTOURS_YR, fontsize=FONT_SIZE, inline=True, inline_spacing=8, manual=LABEL_POSITIONS)

    plt.tight_layout(pad=0.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=400, bbox_inches="tight")
    print(f"Figure saved to {args.output}")


if __name__ == "__main__":
    main()
