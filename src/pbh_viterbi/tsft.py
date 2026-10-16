"""SFT durations for the remapped (t, f^-8/3) search (docs/tsft_optimum.md).

In x = f^-8/3 an inspiral moves at constant speed K = (256/5) pi^(8/3) (G Mc/c^3)^(5/3).
The band [f0, f1] is remapped onto B * Tsft points equally spaced in x
(B = f1 - f0), and the Viterbi transitions allow at most one bin per step, so
the signal is tracked only if K * Tsft <= dx, i.e.

    Tsft <= T_lim = sqrt((f0^-8/3 - f1^-8/3) / (K B)) ~= 17.7 s (Mc / 1e-2 Msun)^(-5/6)

for 61.1-126.8 Hz. Below T_lim, nsigma / max nsigma ~= sqrt(Tsft / T_lim);
above it nsigma drops to the noise level. The best Tsft of a list for a chirp
mass is therefore the largest one <= T_lim.
"""

import numpy as np

from pbh_viterbi.config import FMAX, FMIN

T_SUN = 4.92549094830932e-6  # G Msun / c^3 (s)


def tsft_lim(mchirp, fmin=FMIN, fmax=FMAX, max_bins_per_step=1):
    """Largest Tsft (s) for which the signal moves <= max_bins_per_step remapped bins per SFT."""
    k = 256.0 / 5.0 * np.pi ** (8.0 / 3.0) * (T_SUN * np.asarray(mchirp, dtype=float)) ** (5.0 / 3.0)
    return np.sqrt(max_bins_per_step * (fmin ** (-8.0 / 3.0) - fmax ** (-8.0 / 3.0)) / (k * (fmax - fmin)))


def nsigma_fraction(mchirp, tsft, fmin=FMIN, fmax=FMAX, s_max=1.03):
    """Predicted nsigma / max nsigma: sqrt(Tsft / T_lim) while the drift (Tsft/T_lim)^2 <= s_max, else 0.

    ``s_max`` = 1.03 is the largest drift (bins per step) at which Viterbi was
    still seen to follow the signal in injections.
    """
    t_lim = tsft_lim(mchirp, fmin, fmax)
    ratio = np.asarray(tsft, dtype=float) / t_lim
    return np.where(ratio ** 2 <= s_max, np.sqrt(np.minimum(ratio, 1.0)), 0.0)


def tsft_coverage(tsft_values, mchirps, fmin=FMIN, fmax=FMAX, s_max=1.03):
    """Best predicted nsigma / max nsigma available for each chirp mass with a list of Tsft."""
    mchirps = np.atleast_1d(mchirps)
    fractions = nsigma_fraction(mchirps[:, None], np.asarray(tsft_values, dtype=float)[None, :], fmin, fmax, s_max)
    return fractions.max(axis=1)


def best_tsft(mchirp, tsft_values, fmin=FMIN, fmax=FMAX):
    """Largest Tsft of a list not exceeding T_lim(Mc) (the shortest one if none does)."""
    values = np.sort(np.asarray(tsft_values, dtype=float))
    below = values[values <= tsft_lim(mchirp, fmin, fmax)]
    return float(below.max()) if below.size else float(values[0])
