#!/usr/bin/env python3
"""Fig. 8: average noise PSD of the O3b H1 data compared with aLIGOO3LowT1800545.

Top: average PSD (analysis/average_noise_psd.py) and the analytic O3 curve,
with the search band shaded. Bottom: their ratio.

The published figure evaluated the analytic curve on a grid starting at 0 Hz
while the average PSD starts at 10 Hz, which shifts the analytic curve by
10 Hz. This script aligns both by default; --published-alignment reproduces
the published figure.

    python figures/fig8_average_psd.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from _paper_data import PSD_CSV

from pbh_viterbi.config import FMAX, FMIN
from pbh_viterbi.paths import PLOTS_DIR
from pbh_viterbi.snr import load_psd_csv

ANALYTIC_FLOW = 10
CURVE_FHIGH = 160
PLOT_RANGE = (15, 200)


def analytic_psd(freqs, published_alignment):
    import pycbc.psd

    delta_f = float(freqs[1] - freqs[0])
    if published_alignment:
        # Analytic values at k * delta_f (from 0 Hz), drawn at freqs[k] (from 10 Hz).
        return np.asarray(pycbc.psd.aLIGOO3LowT1800545(len(freqs), delta_f, ANALYTIC_FLOW), dtype=float)
    n = int(round(freqs[-1] / delta_f)) + 1
    curve = pycbc.psd.aLIGOO3LowT1800545(n, delta_f, ANALYTIC_FLOW)
    return np.interp(freqs, np.asarray(curve.sample_frequencies), np.asarray(curve))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--psd", type=Path, default=PSD_CSV)
    parser.add_argument("--published-alignment", action="store_true")
    parser.add_argument("--output", type=Path, default=PLOTS_DIR / "fig8_average_psd.png")
    args = parser.parse_args()

    freqs, real = load_psd_csv(args.psd)
    analytic = analytic_psd(freqs, args.published_alignment)
    keep = ((freqs <= CURVE_FHIGH) & np.isfinite(real) & np.isfinite(analytic) & (real > 0) & (analytic > 0))

    fig, (psd_ax, ratio_ax) = plt.subplots(2, 1, figsize=(7.4, 6.2), sharex=True,
                                           gridspec_kw={"height_ratios": [3, 1]}, constrained_layout=True)
    psd_ax.loglog(freqs[keep], real[keep], linewidth=0.95, color="#155e75", label="Real H1 O3 PSD")
    psd_ax.loglog(freqs[keep], analytic[keep], linewidth=1.2, color="#b45309", label="aLIGO O3 analytic PSD")
    psd_ax.axvspan(FMIN, FMAX, color="#64748b", alpha=0.14, label="Search band")
    psd_ax.set_ylabel("PSD [1/Hz]")
    psd_ax.set_xlim(*PLOT_RANGE)
    psd_ax.legend(frameon=False, loc="best")
    psd_ax.grid(True, which="both", alpha=0.22)

    ratio_ax.loglog(freqs[keep], real[keep] / analytic[keep], linewidth=0.95, color="#374151")
    ratio_ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.6)
    ratio_ax.axvspan(FMIN, FMAX, color="#64748b", alpha=0.14)
    ratio_ax.set_xlabel("Frequency [Hz]")
    ratio_ax.set_ylabel("Ratio")
    ratio_ax.grid(True, which="both", alpha=0.22)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    print(f"Figure saved to {args.output}")


if __name__ == "__main__":
    main()
