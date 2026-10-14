#!/usr/bin/env python3
"""Compare the paper PSD (Appendix A, Welch) with lalpulsar_ComputePSD.

Run compute_psd_lalpulsar.sh first. Prints the ratio of the PSDs in the search
band and the change of the inspiral SNR, rho^2 ~ int f^{-7/3} / S(f) df, and
plots both PSDs and their ratio.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pbh_viterbi.config import FMAX, FMIN  # noqa: E402

OUT = HERE / "output"


def snr2(freqs, psd):
    band = (freqs >= FMIN) & (freqs <= FMAX)
    return np.trapezoid(freqs[band] ** (-7 / 3) / psd[band], freqs[band])


def main():
    welch_f, welch = np.loadtxt(ROOT / "paper_data/snr/average_noise_psd_O3b_H1.csv", delimiter=",", skiprows=1).T
    lal_f, lal = np.loadtxt(OUT / "computepsd_T512.txt", comments="%").T[:2]

    ratio = lal / np.interp(lal_f, welch_f, welch)
    band = (lal_f >= FMIN) & (lal_f <= FMAX)
    lo, med, hi = np.percentile(ratio[band], [16, 50, 84])
    rho = np.sqrt(snr2(lal_f, lal) / snr2(welch_f, welch))
    print(f"Band {FMIN}-{FMAX} Hz: S_ComputePSD/S_Welch median {med:.3f} (16-84%: {lo:.3f}-{hi:.3f}), "
          f"rho_ComputePSD/rho_Welch = {rho:.3f}")

    fig, (ax, rax) = plt.subplots(2, 1, figsize=(7.4, 6.2), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]}, constrained_layout=True)
    ax.loglog(welch_f, welch, lw=0.9, color="#155e75", label="Welch (paper PSD)")
    ax.loglog(lal_f, lal, lw=0.9, color="#b45309", label="lalpulsar_ComputePSD")
    rax.semilogx(lal_f, ratio, lw=0.6, color="#374151")
    rax.axhline(1, color="black", lw=0.8)
    rax.text(0.02, 0.9, f"Search band: median ratio {med:.2f}, " + r"$\rho_{\rm CPSD}/\rho_{\rm Welch}$" + f" = {rho:.3f}",
             transform=rax.transAxes, va="top", fontsize=8,
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))
    for a in (ax, rax):
        a.axvspan(FMIN, FMAX, color="#64748b", alpha=0.14)
        a.grid(True, which="both", alpha=0.22)
    ax.set_xlim(15, 245)
    ax.set_ylim(1e-47, 1e-41)
    ax.set_ylabel("PSD [1/Hz]")
    ax.legend(frameon=False)
    rax.set_ylim(0.5, 1.5)
    rax.set_ylabel("ComputePSD / Welch")
    rax.set_xlabel("Frequency [Hz]")
    fig.savefig(OUT / "psd_crosscheck.png", dpi=200)
    print(f"Figure saved to {OUT / 'psd_crosscheck.png'}")


if __name__ == "__main__":
    main()
