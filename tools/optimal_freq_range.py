#!/usr/bin/env python3
"""Optimal search band [f0, f*] for a detector noise curve, Eq. (13).

Maximises F(f0, f*) = f0^(2/3) / f*^(11/24) * sqrt( int_{f0}^{f*} df / (f^(7/3) S_n(f)) )
and plots F / F_max over the (f0, f*) plane. For the O3 PSD the optimum is
f0 = 61.1 Hz, f* = 126.8 Hz; for the A+ design (O5) 57.5 Hz and 134.4 Hz.

    python tools/optimal_freq_range.py --psd o3
    python tools/optimal_freq_range.py --psd aplus
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution, minimize

APLUS_FILE = Path(__file__).resolve().parent / "data" / "AplusDesign.txt"  # columns: f (Hz), ASD (1/sqrt(Hz))


def load_psd(name):
    if name == "aplus":
        data = np.loadtxt(APLUS_FILE)
        return data[:, 0], data[:, 1] ** 2
    import pycbc.psd

    fhigh, delta_f = 4096.0, 0.25
    psd = pycbc.psd.aLIGOAdVO3LowT1800545(int(2 * fhigh / delta_f), delta_f, 25.0)
    return np.asarray(psd.sample_frequencies), np.asarray(psd)


def figure_of_merit(freqs, psd, flow, fstar):
    mask = (freqs >= flow) & (freqs <= fstar) & (psd > 0)
    if mask.sum() < 2 or fstar <= flow:
        return 0.0
    integral = np.trapezoid(1.0 / (psd[mask] * freqs[mask] ** (7 / 3)), x=freqs[mask])
    if not np.isfinite(integral) or integral <= 0:
        return 0.0
    return flow ** (2 / 3) / fstar ** (11 / 24) * np.sqrt(integral)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--psd", choices=("o3", "aplus"), default="o3")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    freqs, psd = load_psd(args.psd)
    merit = lambda x: -figure_of_merit(freqs, psd, x[0], x[1])  # noqa: E731

    flow_vals = np.linspace(10, 400, 300)
    fstar_vals = np.linspace(30, 600, 300)
    bounds = [(flow_vals[0], flow_vals[-1]), (fstar_vals[0], fstar_vals[-1])]
    # Global search with differential evolution, refined with L-BFGS-B.
    de = differential_evolution(merit, bounds=bounds, seed=0, tol=1e-8, maxiter=2000, popsize=35, polish=False)
    res = minimize(merit, x0=de.x, bounds=bounds, method="L-BFGS-B",
                   options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-10})
    fl_opt, fs_opt = res.x
    f_opt = figure_of_merit(freqs, psd, fl_opt, fs_opt)
    print(f"Optimum: f0 = {fl_opt:.1f} Hz, f* = {fs_opt:.1f} Hz, F = {f_opt:.4g}")

    grid = np.array([[figure_of_merit(freqs, psd, fl, fs) for fl in flow_vals] for fs in fstar_vals])
    grid[grid == 0] = np.nan
    fig, ax = plt.subplots(figsize=(9, 6))
    image = ax.pcolormesh(flow_vals, fstar_vals, grid / f_opt, cmap="viridis", shading="auto", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, label=r"$F(f_0,\,f_*)\;/\;F_\mathrm{max}$")
    ax.plot(fl_opt, fs_opt, "^", color="black", ms=7, label=f"optimum ({fl_opt:.1f}, {fs_opt:.1f}) Hz")
    ax.set_xlabel(r"$f_0$ [Hz]")
    ax.set_ylabel(r"$f_*$ [Hz]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    output = args.output or Path(f"optimal_freq_range_{args.psd}.png")
    fig.savefig(output, dpi=300)
    print(f"Plot saved to {output}")


if __name__ == "__main__":
    main()
