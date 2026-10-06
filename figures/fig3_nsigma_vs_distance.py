#!/usr/bin/env python3
"""Fig. 3: nsigma versus luminosity distance, measured and analytical.

Points: mean nsigma (and standard error over noise realisations) of the
injected search for three chirp masses. Lines: large-SNR approximation
Eq. (19), (2 N_SFT + rho_opt^2 - mu) / sigma, with rho_opt^2 proportional to
1/d_L and normalised to the measured value at d_L = 0.1 kpc.

The paper figure uses the April 2026 injection campaign (60 packs, 20 noise
realisations per signal):

    python figures/fig3_nsigma_vs_distance.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from _paper_data import BACKGROUND_CSV, CAMPAIGN_60_CSV

from pbh_viterbi.config import PACK_DURATION
from pbh_viterbi.paths import PLOTS_DIR

REFERENCE_DISTANCE_MPC = 0.0001
X_LIMITS_KPC = (0.08, 220)
ANALYTIC_MIN_NSIGMA = -0.101
MAX_PACKS = 28
# Chirp mass (Msun) -> Tsft (s) closest to its optimal value, Eq. (14).
TSFT_BY_MASS = {
    0.0044721359549995: 18,
    0.0015874010519681986: 35,
    0.000563453822769568: 88,
}


def mass_label(mass):
    exponent = int(np.floor(np.log10(mass)))
    return rf"{int(mass / 10 ** exponent)}\times 10^{{{exponent}}}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign-csv", type=Path, default=CAMPAIGN_60_CSV)
    parser.add_argument("--background", type=Path, default=BACKGROUND_CSV)
    parser.add_argument("--output", type=Path, default=PLOTS_DIR / "fig3_nsigma_vs_distance.png")
    args = parser.parse_args()

    df = pd.read_csv(args.campaign_csv)
    background = pd.read_csv(args.background).set_index("tsft")

    fig, ax = plt.subplots(figsize=(10.5, 7.0), constrained_layout=True)
    handles = []
    for mass, tsft in TSFT_BY_MASS.items():
        rows = df[np.isclose(df["mchirp"], mass, rtol=1e-12, atol=1e-15)]
        rows = rows[rows["pack"].isin(sorted(rows["pack"].dropna().unique())[:MAX_PACKS])]
        grid = rows.pivot_table(index="distance", columns="pack", values="nsigma", aggfunc="mean").sort_index()

        distances = grid.index.to_numpy(dtype=float)
        values = grid.to_numpy(dtype=float)
        mean = np.nanmean(values, axis=1)
        n_real = np.sum(~np.isnan(values), axis=1)
        sem = np.where(n_real > 1, np.nanstd(values, axis=1, ddof=1) / np.sqrt(n_real), 0.0)

        # Eq. (19): track power at the reference distance, scaled as rho_opt^2 ~ 1/d_L.
        mu, sigma = background.loc[tsft, "mean_total_power"], background.loc[tsft, "std_total_power"]
        n_sft = PACK_DURATION / tsft
        reference = np.isclose(distances, REFERENCE_DISTANCE_MPC, rtol=1e-12, atol=1e-15)
        if not np.any(reference):
            raise ValueError(f"No injections at dL = {REFERENCE_DISTANCE_MPC} Mpc")
        power_reference = sigma * mean[reference][0] + mu
        d_smooth = np.logspace(np.log10(distances.min()), np.log10(X_LIMITS_KPC[1] / 1e3), 2000)
        analytic = (2 * n_sft + power_reference * (REFERENCE_DISTANCE_MPC / d_smooth) - mu) / sigma
        keep = analytic >= ANALYTIC_MIN_NSIGMA

        points = ax.errorbar(distances * 1e3, mean, yerr=sem, fmt="o", markersize=7, elinewidth=1.3, capsize=3,
                             zorder=3)
        color = points.lines[0].get_color()
        for capline in points.lines[1]:
            capline.set_alpha(0.35)
        for barline in points.lines[2]:
            barline.set_alpha(0.45)
        ax.plot(d_smooth[keep] * 1e3, analytic[keep], linestyle="-", linewidth=2.0, color=color, zorder=2)
        handles.append(Line2D([0], [0], marker="o", linestyle="none", markersize=7, markerfacecolor=color,
                              markeredgecolor=color, label=rf"$\mathcal{{M}}={mass_label(mass)}\ M_\odot$"))

    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=0.1, linscale=0.5)
    ax.set_xlim(*X_LIMITS_KPC)
    ax.set_ylim(-0.5, 10 ** 4)
    ax.set_xlabel(r"$d_L\ [{\rm kpc}]$", fontsize=23)
    ax.set_ylabel(r"$n_\sigma$", fontsize=23)
    ax.tick_params(axis="both", which="major", labelsize=23, width=1.0, length=6)
    ax.tick_params(axis="both", which="minor", width=0.8, length=3)
    ax.grid(True, which="both", alpha=0.25)
    formula = Line2D([0], [0], color="black", linewidth=2.0, label=(
        r"$\frac{2N_{\rm SFT}+(\rho^{\rm opt}_{\rm tot})^2"
        r"-\mu(\rho^{\rm opt}_{\rm tot}=0)}"
        r"{\sigma(\rho^{\rm opt}_{\rm tot}=0)}$"))
    ax.legend(handles=handles + [formula], loc="upper right", fontsize=19, frameon=True, framealpha=0.85,
              borderpad=0.45, labelspacing=0.35, handletextpad=0.55, handlelength=1.8, borderaxespad=0.5)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f"Figure saved to {args.output}")


if __name__ == "__main__":
    main()
