#!/usr/bin/env python3
"""Sensitivity and cost of Tsft lists built with nsigma / max nsigma >= 0.95 and >= 0.90.

For chirp masses log-uniform in [2e-4, 0.1] Msun, each mass uses the largest
Tsft of the list <= T_lim, which gives nsigma / max = sqrt(Tsft / T_lim)
(nsigma_derivation.tex). Reported: worst and mean nsigma fraction, and the
corresponding reach in distance with nsigma ~ d^-gamma, gamma = 0.87 measured
on injections (the map uses the |SFT| amplitude, not the power). Cost per
injection = fixed part (frame reading, injection) + per-Tsft part, both
measured with the 45-Tsft run (injection_check_dense: 75 s per injection).
The paper list (13 Tsft) is shown for reference.

    python tools/opt_tsft_remapped/compare_tsft_lists.py
"""

import numpy as np

from plot_tsft_lim import TSFT_GRID, tsft_lim
from tsft_list_remapped import coverage, optimal_list

FMIN, FMAX, MCHIRP_RANGE = 61.1, 126.8, (2e-4, 0.1)
GAMMA = 0.87  # nsigma ~ d^-gamma (measured)
FIXED_S, PER_TSFT_S = 8.0, (75.0 - 8.0) / 45  # s per injection: fixed part, per Tsft (45-Tsft run)
N_INJECTIONS, N_NOISE = 18000, 108  # paper campaign: 600 signals x 30 noise realisations; noise chunks
S_MAX = 1.03


def tsft_list(threshold):
    lo, hi = float(tsft_lim(MCHIRP_RANGE[1], FMIN, FMAX)), float(tsft_lim(MCHIRP_RANGE[0], FMIN, FMAX))
    candidates = np.unique(np.round(np.geomspace(0.8 * lo, hi, 2000) * 8) / 8)  # multiples of 1/8 s
    mchirps = np.geomspace(*MCHIRP_RANGE, 600)
    return optimal_list(mchirps, candidates, FMIN, FMAX, threshold, S_MAX)


def main():
    mchirps = np.geomspace(*MCHIRP_RANGE, 4000)
    lists = {"0.95": tsft_list(0.95), "0.90": tsft_list(0.90), "paper (13)": np.array(TSFT_GRID, float)}
    print(f"{'list':>11} {'N_Tsft':>6} {'worst ns':>8} {'mean ns':>8} {'worst d':>8} {'mean d':>7} "
          f"{'s/inj':>6} {'core-h':>7} {'h on 8 cores':>12} {'h on 24':>8}")
    for name, tsfts in lists.items():
        q = coverage(tsfts, mchirps, FMIN, FMAX, S_MAX)
        per_chunk = FIXED_S + PER_TSFT_S * len(tsfts)
        core_h = (N_INJECTIONS + 2 * N_NOISE) * per_chunk / 3600  # injections + noise search + background
        print(f"{name:>11} {len(tsfts):6d} {q.min():8.3f} {q.mean():8.3f} {q.min() ** (1 / GAMMA):8.3f} "
              f"{np.mean(q ** (1 / GAMMA)):7.3f} {per_chunk:6.0f} {core_h:7.0f} {core_h / 8:12.0f} {core_h / 24:8.0f}")
    for name in ("0.95", "0.90"):
        print(f"\nList {name} ({len(lists[name])} Tsft, s): " + ", ".join(f"{t:g}" for t in lists[name]))


if __name__ == "__main__":
    main()
