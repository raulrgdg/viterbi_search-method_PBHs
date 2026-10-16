#!/usr/bin/env python3
"""nsigma / max nsigma vs Tsft for the dense Tsft list, compared with sqrt(Tsft / T_ref).

For each chirp mass and distance the curve is the median over the packs of
nsigma(Tsft) / max_Tsft nsigma. T_ref is the largest Tsft of the list <= T_lim. Undetected signals
(median max nsigma < 5) are left out.
Outputs results/injections.csv, results/summary.csv and results/nsigma_vs_tsft.png.

    python analyse.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from plot_tsft_lim import tsft_lim  # noqa: E402

RESULTS = HERE / "results"
FMIN, FMAX = 61.1, 126.8
MIN_DETECTED = 5  # curves of undetected signals (median max nsigma below this) are noise; not shown


def sci(value):
    exponent = int(np.floor(np.log10(value)))
    return rf"{value / 10**exponent:.1f}\times10^{{{exponent}}}"


def sci_label(distance):
    """Legend label for a distance given in Mpc, shown in kpc."""
    return rf"$d_L = {distance * 1e3:.2g}$ kpc"


def main():
    df = pd.concat([pd.read_csv(p) for p in sorted(RESULTS.glob("injections.job*.csv"))], ignore_index=True)
    df = df.sort_values(["i_mass", "i_dist", "pack"])
    df.to_csv(RESULTS / "injections.csv", index=False)
    cols = [c for c in df.columns if c.startswith("nsigma_tsft")]
    ts = np.array([float(c.removeprefix("nsigma_tsft")) for c in cols])
    order = np.argsort(ts)
    ts, cols = ts[order], [cols[k] for k in order]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    rows = []
    for ax, (_, g) in zip(axes, df.groupby("i_mass")):
        mc = g.mchirp.iloc[0]
        t_lim = float(tsft_lim(mc, FMIN, FMAX))
        t_ref = ts[ts <= t_lim].max()
        curves = []
        for _, h in g.groupby("i_dist"):
            values = h[cols].to_numpy()
            norm = np.median(values / values.max(axis=1, keepdims=True), axis=0)
            nmax = np.median(values.max(axis=1))
            if nmax >= MIN_DETECTED:
                curves.append((h.distance.iloc[0], nmax, norm))
        # Plot the nearest, middle and farthest detected distances.
        shown = sorted({0, len(curves) // 2, len(curves) - 1})
        for color, k in zip(plt.cm.viridis(np.linspace(0, 0.85, len(shown))), shown):
            ax.plot(ts, curves[k][2], "o-", color=color, ms=2.5, lw=1, label=sci_label(curves[k][0]))
        for distance, nmax, norm in curves:
            near = (ts >= t_ref / 3) & (ts <= t_ref)
            dev = norm[near] - np.sqrt(ts[near] / t_ref)
            rows.append({"mchirp": mc, "t_lim": t_lim, "t_ref": t_ref, "distance": distance,
                         "nsigma_max_median": nmax, "argmax_tsft": ts[np.argmax(norm)],
                         "max_abs_dev_near": np.abs(dev).max(), "mean_dev_near": dev.mean()})
        grid = np.geomspace(ts.min(), t_ref, 200)
        ax.plot(grid, np.sqrt(grid / t_ref), "k--", lw=1.6, label=r"$\sqrt{T_{\rm SFT}/T_{\rm ref}}$")
        ax.axvline(t_lim, color="r", lw=0.9, ls=":", label=r"$T_{\rm SFT}^{\rm lim}$")
        ax.set_xscale("log")
        ax.set_title(rf"$\mathcal{{M}} = {sci(mc)}\,M_\odot$", fontsize=11)
        ax.set_xlabel(r"$T_{\rm SFT}$ [s]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel(r"$n_\sigma / \max_{T_{\rm SFT}} n_\sigma$")
    axes[0].set_ylim(-0.05, 1.08)
    fig.tight_layout()
    fig.savefig(RESULTS / "nsigma_vs_tsft.png", dpi=300)

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
