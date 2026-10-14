#!/usr/bin/env python3
"""Maximum SFT duration that Viterbi can track in the remapped (t, f^-8/3) map.

In x = f^-8/3 an inspiral moves at constant speed
K = (256/5) pi^(8/3) (G Mc/c^3)^(5/3). The band [f0, f1] (width B = f1 - f0)
is remapped onto B*Tsft points equally spaced in x, and the Viterbi
transitions allow at most one bin per step, so the signal is tracked if
K * Tsft <= dx, i.e.

    T_lim = sqrt((f0^-8/3 - f1^-8/3) / (K B)) = sqrt(dtau_band / B),

with dtau_band the time the signal takes to cross the band. For 61.1-126.8 Hz,
T_lim ~= 17.7 s (Mc / 1e-2 Msun)^(-5/6). The selected Tsft is the largest
grid value <= T_lim (checked with 100 injections, see
sft-gen-fast/fast-test/README.md). For comparison the plot also shows the
paper Eq. (14), one linear frequency bin per SFT at f1.

    python tools/opt_tsft_remapped/plot_tsft_lim.py
    python tools/opt_tsft_remapped/plot_tsft_lim.py --fmin 61.1 --fmax 126.8 --mchirp-min 2e-4 --mchirp-max 0.1
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

T_SUN = 4.92549094830932e-6  # G Msun / c^3 (s)
TSFT_GRID = (2, 3, 4, 5, 7, 10, 13, 18, 25, 35, 47, 63, 88)  # paper SFT durations (s)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "results" / "plots" / "tsft_lim_vs_mchirp.png"


def tsft_lim(mchirp, fmin, fmax, max_bins_per_step=1):
    """Largest Tsft (s) for which the signal moves <= max_bins_per_step remapped bins per SFT."""
    k = 256.0 / 5.0 * np.pi ** (8.0 / 3.0) * (T_SUN * np.asarray(mchirp)) ** (5.0 / 3.0)
    return np.sqrt(max_bins_per_step * (fmin ** (-8.0 / 3.0) - fmax ** (-8.0 / 3.0)) / (k * (fmax - fmin)))


def tsft_eq14(mchirp, fmax):
    """Paper Eq. (14): one linear frequency bin per SFT at the upper cutoff fmax."""
    return (5.0 * np.pi / 96.0) ** 0.5 * (T_SUN * np.asarray(mchirp)) ** (-5.0 / 6.0) * (np.pi * fmax) ** (-11.0 / 6.0)


def selected_from_grid(t_lim, grid=TSFT_GRID):
    """Largest grid Tsft <= t_lim (the shortest one if none fits)."""
    grid = np.asarray(sorted(grid))
    idx = np.searchsorted(grid, t_lim, side="right") - 1
    return grid[np.clip(idx, 0, len(grid) - 1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fmin", type=float, default=61.1, help="Lower edge of the band (Hz).")
    parser.add_argument("--fmax", type=float, default=126.8, help="Upper edge of the band (Hz).")
    parser.add_argument("--mchirp-min", type=float, default=2e-4, help="Msun")
    parser.add_argument("--mchirp-max", type=float, default=0.1, help="Msun")
    parser.add_argument("--max-bins-per-step", type=int, default=1, help="Largest Viterbi jump (bins per step).")
    parser.add_argument("--measured", type=Path, help="Optional CSV with columns mchirp, tsft_selected "
                                                      "(e.g. fast-test/results/tsft_formula.csv) to overlay.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    mchirp = np.logspace(np.log10(args.mchirp_min), np.log10(args.mchirp_max), 500)
    t_lim = tsft_lim(mchirp, args.fmin, args.fmax, args.max_bins_per_step)
    ratio = float(tsft_lim(1e-2, args.fmin, args.fmax, args.max_bins_per_step) / tsft_eq14(1e-2, args.fmax))
    coefficient = float(tsft_lim(1e-2, args.fmin, args.fmax, args.max_bins_per_step))

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.loglog(mchirp, t_lim, color="C0", lw=2,
              label=rf"$T_{{\rm SFT}}^{{\rm lim}}$ (remapped map, $\pm${args.max_bins_per_step} bin)")
    ax.loglog(mchirp, tsft_eq14(mchirp, args.fmax), color="C1", ls="--", lw=1.5,
              label=r"$T_{\rm SFT}^{\rm opt}$, Eq. (14)")
    for t in TSFT_GRID:
        ax.axhline(t, color="0.85", lw=0.6, zorder=0)
    if args.measured:
        import pandas as pd

        df = pd.read_csv(args.measured)
        ax.scatter(df.mchirp, df.tsft_selected, s=18, color="k", zorder=3, label="selected in injections")
    ax.set_xlabel(r"Chirp mass $\mathcal{M}$ [$M_\odot$]")
    ax.set_ylabel(r"$T_{\rm SFT}$ [s]")
    ax.set_title(f"Band {args.fmin:g}–{args.fmax:g} Hz", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, which="major", alpha=0.3)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=400)

    print(f"T_lim = {coefficient:.2f} s (Mc/1e-2 Msun)^(-5/6); ratio to Eq. (14) = {ratio:.3f}")
    for mc in (2e-4, 1e-3, 4.472e-3, 1e-2, 3.55e-2, 7.08e-2, 0.1):
        if args.mchirp_min <= mc <= args.mchirp_max:
            t = float(tsft_lim(mc, args.fmin, args.fmax, args.max_bins_per_step))
            print(f"  Mc = {mc:.3g} Msun: T_lim = {t:7.2f} s, grid Tsft = {selected_from_grid(t)} s")
    print(f"Plot written to {args.output}")


if __name__ == "__main__":
    main()
