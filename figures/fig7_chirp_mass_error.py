#!/usr/bin/env python3
"""Fig. 7: relative error of the chirp-mass estimate, delta = (M_hat - M_true) / M_true, Eq. (26).

Central panel: median delta per injected (Mc, dL) over noise realisations,
with the lower bound of the distance reach (Fig. 6) overlaid. Side panels:
median over the other axis of the per-cell medians, with a band of the
median per-cell standard deviation.

    python figures/fig7_chirp_mass_error.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator
from _paper_data import CAMPAIGN_CSV, THRESHOLD_JSON, load_campaign
from fig6_distance_reach import reach_kpc

from pbh_viterbi.paths import PLOTS_DIR
from pbh_viterbi.sensitivity import pchip_curve

MCHIRP_RANGE = (2.8e-4, 1e-1)
LINTHRESH = 0.1  # |delta| below 10% shown on a linear colour scale
LINSCALE = 0.7


def minor_log_locator():
    return LogLocator(subs=np.arange(2, 10) * 0.1, numticks=20)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign-csv", type=Path, default=CAMPAIGN_CSV)
    parser.add_argument("--threshold", type=Path, default=THRESHOLD_JSON)
    parser.add_argument("--output", type=Path, default=PLOTS_DIR / "fig7_chirp_mass_error.png")
    args = parser.parse_args()

    df = load_campaign(args.campaign_csv, args.threshold)
    df["delta"] = (df["mass"] - df["mchirp"]) / df["mchirp"]
    stats = df.groupby(["mchirp", "distance"])["delta"].agg(median="median", std="std").reset_index()

    in_range = (stats["mchirp"] >= MCHIRP_RANGE[0]) & (stats["mchirp"] <= MCHIRP_RANGE[1])
    mchirp = np.sort(stats.loc[in_range, "mchirp"].unique())
    distance = np.sort(stats["distance"].unique())
    grid = stats.pivot(index="distance", columns="mchirp", values="median")[mchirp].values
    stats = stats[in_range]
    right_med = stats.groupby("distance")["median"].median().reindex(distance).values
    right_std = stats.groupby("distance")["std"].median().reindex(distance).values
    bottom_med = stats.groupby("mchirp")["median"].median().reindex(mchirp).values
    bottom_std = stats.groupby("mchirp")["std"].median().reindex(mchirp).values
    distance_kpc = 1e3 * distance

    vlim = max(np.nanpercentile(np.abs(grid), 95), 0.05)
    norm = mcolors.SymLogNorm(linthresh=LINTHRESH, linscale=LINSCALE, vmin=-vlim, vmax=vlim, base=10)

    plt.rcParams.update({"font.size": 14, "axes.labelsize": 17, "xtick.labelsize": 15, "ytick.labelsize": 15})
    fig = plt.figure(figsize=(7.1, 5.0), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[4, 1], hspace=0.05, wspace=0.05,
                           left=0.09, right=0.88, top=0.90, bottom=0.10)
    ax_main, ax_right, ax_bot = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0])
    ax_cb = fig.add_axes([0.90, 0.30, 0.018, 0.60])

    image = ax_main.pcolormesh(mchirp, distance_kpc, grid, cmap="RdBu_r", norm=norm, shading="nearest")
    ax_main.set_xscale("log")
    ax_main.set_yscale("log")
    ax_main.set_ylabel(r"$d_L\ (\mathrm{kpc})$")
    ax_main.set_xlim(*MCHIRP_RANGE)
    ax_main.set_ylim(bottom=1e-1)
    ax_main.tick_params(labelbottom=False)
    ax_main.xaxis.set_minor_locator(minor_log_locator())
    ax_main.yaxis.set_minor_locator(minor_log_locator())
    cbar = fig.colorbar(image, cax=ax_cb)
    cbar.set_label(r"$\delta$", labelpad=4)
    cbar.ax.tick_params(labelsize=14, direction="in")

    ax_right.fill_betweenx(distance_kpc, right_med - right_std, right_med + right_std, alpha=0.25, color="steelblue")
    ax_right.plot(right_med, distance_kpc, color="steelblue", lw=1.1)
    ax_right.axvline(0, color="k", lw=0.6, ls="--")
    ax_right.set_yscale("log")
    ax_right.tick_params(labelleft=False, labelbottom=False, top=True, right=True)
    ax_right.set_xlim(np.nanmin(right_med - right_std) * 1.1, np.nanmax(right_med + right_std) * 1.1)
    ax_right.set_xlabel(r"$\langle\delta\rangle_{\mathcal{M}}$", fontsize=16, labelpad=5)
    ax_right.xaxis.set_label_position("top")
    ax_right.tick_params(labeltop=True, labelsize=14)
    ax_right.yaxis.set_minor_locator(minor_log_locator())
    ax_right.xaxis.set_major_locator(plt.FixedLocator([-1, 0, 1]))

    ax_bot.fill_between(mchirp, bottom_med - bottom_std, bottom_med + bottom_std, alpha=0.25, color="indianred")
    ax_bot.plot(mchirp, bottom_med, color="indianred", lw=1.1)
    ax_bot.axhline(0, color="k", lw=0.6, ls="--")
    ax_bot.set_xscale("log")
    ax_bot.tick_params(labelleft=True, labelsize=14, axis="x")
    ax_bot.set_ylabel(r"$\langle\delta\rangle_{d_L}$", labelpad=3)
    ax_bot.set_xlabel(r"$\mathcal{M} \ (M_\odot)$", fontsize=16)
    ax_bot.set_ylim(np.nanmin(bottom_med - bottom_std) * 1.1, np.nanmax(bottom_med + bottom_std) * 1.1)
    ax_bot.xaxis.set_minor_locator(minor_log_locator())
    ax_bot.yaxis.set_major_locator(plt.FixedLocator([0, 10]))
    ax_bot.tick_params(axis="y", labelsize=13)
    ax_bot.set_xlim(2.6e-4, 1e-1)
    ax_right.set_ylim(ax_main.get_ylim())

    reach = reach_kpc(args.campaign_csv, args.threshold)
    curve_mass, curve_low = pchip_curve(reach["mchirp"].to_numpy(), reach["d95_low"].to_numpy())
    ax_main.plot(curve_mass, curve_low, color="black", lw=1.2, ls="--", label=r"Distance reach $(d_{L,95\%})$")
    ax_main.legend(loc="upper left", framealpha=0.7, fontsize=11)
    fig.add_subplot(gs[1, 1]).axis("off")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=500, bbox_inches="tight")
    print(f"Figure saved to {args.output}")


if __name__ == "__main__":
    main()
