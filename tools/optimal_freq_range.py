#!/usr/bin/env python3
"""Optimal search band [f0, f*] for a detector noise curve.

Maximises Eq. (2.30) of arXiv:2401.02314 (Eq. 13 of the paper),

    F(f0, f*) = f0^(2/3) f*^(-11/24) sqrt( int_{f0}^{f*} df / (f^(7/3) S_n(f)) ),

by brute force on a 0.1 Hz grid of (f0, f*), using a cumulative integral.
For the O3 noise curve (aLIGOAdVO3LowT1800545) the optimum is f0 = 61.1 Hz,
f* = 126.8 Hz, F = 7.77e21 Hz^(1/24), as in Fig. 1 of arXiv:2401.02314; for
the A+ design (O5), f0 ≈ 57.4 Hz, f* ≈ 134 Hz (flat maximum; the last digit
depends on how the tabulated curve is interpolated).

The noise curve must cover the whole band well: a PSD measured on data
resampled to 512 Hz is suppressed by the anti-aliasing filter above ~150 Hz
and pushes the optimum to the edge of the data.

    python tools/optimal_freq_range.py                    # O3
    python tools/optimal_freq_range.py --psd aplus        # A+ design (O5)
    python tools/optimal_freq_range.py --psd-file psd.txt # two columns: f (Hz), PSD (1/Hz)
"""

import argparse
from pathlib import Path

import numpy as np

APLUS_FILE = Path(__file__).resolve().parent / "data" / "AplusDesign.txt"  # columns: f (Hz), ASD (1/sqrt(Hz))


def load_psd(name, psd_file=None, fmin=15.0, fmax=600.0, delta_f=0.05):
    """Frequencies and PSD on a uniform grid of step delta_f between fmin and fmax."""
    if psd_file:
        csv = str(psd_file).endswith(".csv")
        data = np.loadtxt(psd_file, delimiter="," if csv else None, skiprows=1 if csv else 0)
        f, psd = data[:, 0], data[:, 1]
    elif name == "aplus":
        data = np.loadtxt(APLUS_FILE)
        f, psd = data[:, 0], data[:, 1] ** 2
    else:
        import pycbc.psd

        series = pycbc.psd.aLIGOAdVO3LowT1800545(int(fmax / delta_f) + 1, delta_f, 10.0)
        f, psd = np.asarray(series.sample_frequencies), np.asarray(series)
    good = psd > 0
    grid = np.arange(max(fmin, f[good].min()), min(fmax, f[good].max()), delta_f)
    return grid, np.exp(np.interp(grid, f[good], np.log(psd[good])))


def merit_grid(f, psd, step=0.1):
    """F(f0, f*) on a grid of step ``step`` Hz: returns (frequencies, F[f0 index, f* index])."""
    g = 1.0 / (f ** (7 / 3) * psd)
    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(f))])
    idx = np.arange(0, len(f), max(1, int(round(step / (f[1] - f[0])))))
    integral = cumulative[idx][None, :] - cumulative[idx][:, None]
    fk = f[idx]
    merit = fk[:, None] ** (2 / 3) * fk[None, :] ** (-11 / 24) * np.sqrt(np.clip(integral, 0, None))
    return fk, merit


def optimal_band(f, psd, step=0.1):
    """(f0, f*, F_max) maximising F."""
    fk, merit = merit_grid(f, psd, step)
    i, j = np.unravel_index(np.argmax(merit), merit.shape)
    return fk[i], fk[j], merit[i, j]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--psd", choices=("o3", "aplus"), default="o3")
    parser.add_argument("--psd-file", type=Path, help="Two-column PSD file (f, PSD); overrides --psd.")
    parser.add_argument("--fmin", type=float, default=15.0, help="Lowest frequency considered (Hz).")
    parser.add_argument("--fmax", type=float, default=600.0, help="Highest frequency considered (Hz).")
    parser.add_argument("--plot", type=Path, help="Save the map of F / F_max to this file.")
    args = parser.parse_args()

    f, psd = load_psd(args.psd, args.psd_file, args.fmin, args.fmax)
    f0, f1, best = optimal_band(f, psd)
    print(f"Optimal band: f0 = {f0:.1f} Hz, f* = {f1:.1f} Hz (F = {best:.3e} Hz^(1/24))")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fk, merit = merit_grid(f, psd, step=1.0)
        z = merit.T / best
        z[z <= 0] = np.nan
        fig, ax = plt.subplots(figsize=(6.5, 5))
        image = ax.pcolormesh(fk, fk, z, cmap="viridis", shading="auto", vmin=0, vmax=1, rasterized=True)
        fig.colorbar(image, ax=ax, label=r"$F(f_0, f_*) / F_{\max}$")
        ax.plot(f0, f1, "k^", ms=7, label=f"optimum ({f0:.1f}, {f1:.1f}) Hz")
        ax.set(xscale="log", yscale="log", xlabel=r"$f_0$ [Hz]", ylabel=r"$f_*$ [Hz]")
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=300)
        print(f"Plot saved to {args.plot}")


if __name__ == "__main__":
    main()
