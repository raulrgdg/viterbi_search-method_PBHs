#!/usr/bin/env python3
"""Fig. 6: distance reach d_L,95% of the injected search versus chirp mass.

Injections recovered by the FAR threshold give, per chirp mass, the cumulative
recovered fraction versus distance; d_L,95% and its 1-sigma Wilson bounds
follow from pbh_viterbi.sensitivity. Curves are PCHIP interpolations in
log-log space. The blue region is below the lower (pessimistic) bound; the
grey band spans the Wilson interval. The per-mass values are also written to CSV.

    python figures/fig6_distance_reach.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from _paper_data import CAMPAIGN_CSV, THRESHOLD_JSON, load_campaign

from pbh_viterbi.paths import PLOTS_DIR, SEARCH_RESULTS_DIR
from pbh_viterbi.sensitivity import distance_reach, pchip_curve

FILL_COLOR = "#92befb"
BAND_COLOR = "#8c8c8c"


def reach_kpc(campaign_csv, threshold):
    df = load_campaign(campaign_csv, threshold)
    reach = distance_reach(df["mchirp"], df["distance"], df["recovered"])
    reach[["d95", "d95_low", "d95_high"]] *= 1e3  # Mpc -> kpc
    return reach


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign-csv", type=Path, default=CAMPAIGN_CSV)
    parser.add_argument("--threshold", type=Path, default=THRESHOLD_JSON)
    parser.add_argument("--output", type=Path, default=PLOTS_DIR / "fig6_distance_reach.png")
    parser.add_argument("--table", type=Path, default=SEARCH_RESULTS_DIR / "distance_reach.csv")
    args = parser.parse_args()

    reach = reach_kpc(args.campaign_csv, args.threshold)
    mass, low = pchip_curve(reach["mchirp"].to_numpy(), reach["d95_low"].to_numpy())
    _, high = pchip_curve(reach["mchirp"].to_numpy(), reach["d95_high"].to_numpy())

    plt.rcParams.update({"font.size": 16, "axes.labelsize": 19, "xtick.labelsize": 17, "ytick.labelsize": 17})
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.fill_between(mass, low, 1e-4, color=FILL_COLOR, alpha=0.25)
    ax.fill_between(mass, low, high, color=BAND_COLOR, alpha=0.25, zorder=1.5)
    for curve in (low, high):
        ax.plot(mass, curve, color=BAND_COLOR, linestyle="--", linewidth=1.3, zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\mathcal{M}\,(M_\odot)$")
    ax.set_ylabel(r"$d_{L,\,95\%}$ (kpc)")
    ax.set_xlim(2.8e-4, 1e-1)
    ax.set_ylim(1e-1, 2e2)
    ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext())
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
    ax.tick_params(which="both", direction="in", top=True, right=True)
    # Same late rcParams update as the published figure (it changes the tight_layout margins).
    plt.rcParams.update({"font.size": 16, "axes.labelsize": 18, "xtick.labelsize": 14, "ytick.labelsize": 14,
                         "axes.titlesize": 16})
    plt.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=400)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    reach.rename(columns={"d95": "d95_kpc", "d95_low": "d95_low_kpc", "d95_high": "d95_high_kpc"}).to_csv(
        args.table, index=False)
    print(f"Figure saved to {args.output}\nTable saved to {args.table}")
    print(reach.to_string(index=False, float_format=lambda v: f"{v:.6g}"))


if __name__ == "__main__":
    main()
