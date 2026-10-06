#!/usr/bin/env python3
"""Fig. 5: triggers of the noise and injected searches in the (NMSE, nsigma) plane.

Injected triggers are coloured by the optimal SNR of the signal (median over
sky locations, paper_data/snr/injection_snr_sky_median.csv); noise triggers
are black crosses; the line is the FAR threshold
(paper_data/calibration/far_threshold.json, analysis/compute_threshold.py).

    python figures/fig5_trigger_plane.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from scipy.spatial import KDTree
from _paper_data import CAMPAIGN_CSV, NOISE_CSV, SNR_CSV, THRESHOLD_JSON

from pbh_viterbi.paths import PLOTS_DIR
from pbh_viterbi.threshold import load_threshold, threshold_curve

X_LIMITS = (6e-8, 10)
Y_LIMITS = (-5, 1e4)
CURVE_MAX_LOG_NMSE = 1.0


def valid_triggers(path):
    df = pd.read_csv(path)
    for column in ("nmse", "nsigma"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["nmse", "nsigma"])
    return df[df["nmse"] > 0]


def attach_snr(signal, snr_path):
    """Optimal SNR of each injection, matched to the SNR grid in log(Mc), log(dL)."""
    grid = pd.read_csv(snr_path).groupby(["mchirp", "distance"], as_index=False)["optimal_snr"].mean()
    tree = KDTree(np.column_stack([np.log10(grid["mchirp"]), np.log10(grid["distance"])]))
    _, idx = tree.query(np.column_stack([np.log10(signal["mchirp"]), np.log10(signal["distance"])]))
    return grid["optimal_snr"].to_numpy()[idx]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--noise-csv", type=Path, default=NOISE_CSV)
    parser.add_argument("--signal-csv", type=Path, default=CAMPAIGN_CSV)
    parser.add_argument("--snr-csv", type=Path, default=SNR_CSV)
    parser.add_argument("--threshold", type=Path, default=THRESHOLD_JSON)
    parser.add_argument("--output", type=Path, default=PLOTS_DIR / "fig5_trigger_plane.png")
    args = parser.parse_args()

    noise = valid_triggers(args.noise_csv)
    signal = valid_triggers(args.signal_csv).dropna(subset=["mchirp", "distance", "pack"])
    # Drawing order of the published figure (overlapping semi-transparent points).
    signal = signal.sort_values(["mchirp", "distance", "pack"]).reset_index(drop=True)
    snr = attach_snr(signal, args.snr_csv)

    plt.rcParams.update({"font.size": 14, "axes.labelsize": 19, "xtick.labelsize": 17, "ytick.labelsize": 17})
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    signal_points = ax.scatter(signal["nmse"], signal["nsigma"], c=snr, cmap="YlOrRd",
                               norm=LogNorm(vmin=float(np.nanmin(snr)), vmax=float(np.nanmax(snr))),
                               s=12, alpha=0.70, label="Signal")
    noise_points = ax.scatter(noise["nmse"], noise["nsigma"], s=24, marker="x", color="black", linewidths=1.2,
                              label="Noise")

    # The curve is drawn from the smallest fitted NMSE (<= 10) up to NMSE = 10.
    log_nmse = np.log10(np.concatenate([signal["nmse"], noise["nmse"]]))
    x_line = np.linspace(log_nmse[log_nmse <= CURVE_MAX_LOG_NMSE].min(), CURVE_MAX_LOG_NMSE, 600)
    threshold_line, = ax.plot(10 ** x_line, threshold_curve(load_threshold(args.threshold), 10 ** x_line),
                              color="black", linewidth=1.5, label="Threshold\n" + r"$\mathrm{FAR} \,  3\%$")

    ax.set_xlabel(r"$\mathrm{NMSE}$")
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylabel(r"$n_{\sigma}$")
    ax.set_xlim(*X_LIMITS)
    ax.set_ylim(*Y_LIMITS)
    ax.grid(True, alpha=0.3)
    ax.legend(handles=[signal_points, noise_points, threshold_line], loc="upper right", frameon=True)
    fig.colorbar(signal_points, ax=ax, pad=0.02).set_label("SNR")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Figure saved to {args.output}")


if __name__ == "__main__":
    main()
