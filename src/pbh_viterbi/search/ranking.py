"""[UNDER TEST] Ranking statistics and generalisable thresholds in the (NMSE, nsigma) plane.

Part of the threshold review (studies/ranking_statistic). Nothing here is used
by the default pipeline; analysis/compute_threshold.py (the paper threshold) is
unchanged.

Problems of the paper threshold that this module addresses:

* it is a free polynomial in log10(NMSE): outside the range of the noise
  triggers it extrapolates (e.g. nsigma > -82 at NMSE = 1e-12) and it may
  decrease with NMSE;
* it is fitted and its FAR measured on the same triggers (optimistic);
* a small NMSE alone is not evidence of a signal: Viterbi tracks of noise
  that jump one bin per step over a few steps are straight lines, fitted
  with NMSE down to 1e-9 (blocks of 2-24 steps), see block_steps.

Tools:

* block_steps, drift_bins: length (SFT steps) and frequency drift (remapped
  bins) of the isolated block, from the per-Tsft table;
* linear_stat: R_a = nsigma - a log10(NMSE), a one-parameter family that spans
  the ranking space from nsigma alone (a = 0) to NMSE alone (a -> inf);
* MonotoneThreshold: nsigma >= g(log10 NMSE), g piecewise linear and
  non-decreasing by construction, flat outside the knot range (no
  extrapolation beyond the noise data);
* pack_folds, clopper_pearson: cross-validation over packs and the
  confidence interval of a FAR k/N.
"""

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import beta

from pbh_viterbi.config import FBAND, FMAX, FMIN
from pbh_viterbi.search.fitting import slope_of_mass


def block_steps(block_start_s, block_end_s, tsft):
    """Number of SFT steps of the isolated block."""
    return (np.asarray(block_end_s, float) - np.asarray(block_start_s, float)) / np.asarray(tsft, float)


def drift_bins(mass, block_start_s, block_end_s, tsft):
    """Frequency drift of the fitted inspiral across the block, in remapped (f^-8/3) bins."""
    tsft = np.asarray(tsft, float)
    bin_width = (FMIN ** (-8 / 3) - FMAX ** (-8 / 3)) / (FBAND * tsft)
    duration = np.asarray(block_end_s, float) - np.asarray(block_start_s, float)
    return np.abs(slope_of_mass(np.asarray(mass, float))) * duration / bin_width


def linear_stat(nsigma, nmse, a):
    """R_a = nsigma - a log10(NMSE); -inf where there is no valid NMSE."""
    nsigma, nmse = np.asarray(nsigma, float), np.asarray(nmse, float)
    out = np.full(nsigma.shape, -np.inf)
    ok = np.isfinite(nsigma) & np.isfinite(nmse) & (nmse > 0)
    out[ok] = nsigma[ok] - a * np.log10(nmse[ok])
    return out


def threshold_for_far(noise_stat, max_false):
    """Smallest value t with at most ``max_false`` noise statistics strictly above it."""
    values = np.sort(np.asarray(noise_stat, float))[::-1]
    return values[max_false] if max_false < len(values) else -np.inf


def clopper_pearson(k, n, cl=0.68):
    """Two-sided Clopper-Pearson interval of a binomial proportion k/n (default 68% CL)."""
    alpha = 1 - cl
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def pack_folds(packs, n_folds, seed=0):
    """Split pack ids into ``n_folds`` folds (a dict pack -> fold), shuffled reproducibly."""
    packs = np.array(sorted(set(int(p) for p in packs)))
    order = np.random.default_rng(seed).permutation(len(packs))
    return {int(packs[i]): int(k % n_folds) for k, i in enumerate(order)}


class MonotoneThreshold:
    """nsigma threshold g(x), x = log10(NMSE): piecewise linear, non-decreasing, flat outside the knots.

    g(x_0) = v and g(x_{j+1}) = g(x_j) + d_j with d_j >= 0, so the curve can only
    rise with NMSE and is held constant below the first and above the last knot.
    """

    def __init__(self, knots, values=None):
        self.knots = np.asarray(knots, float)
        self.values = None if values is None else np.asarray(values, float)

    @staticmethod
    def _values(params):
        return params[0] + np.concatenate([[0.0], np.cumsum(np.abs(params[1:]))])

    def curve(self, nmse):
        x = np.log10(np.asarray(nmse, float))
        return np.interp(x, self.knots, self.values)  # np.interp is flat outside the knots

    def passes(self, nsigma, nmse):
        nsigma, nmse = np.asarray(nsigma, float), np.asarray(nmse, float)
        ok = np.isfinite(nsigma) & np.isfinite(nmse) & (nmse > 0)
        out = np.zeros(nsigma.shape, bool)
        out[ok] = nsigma[ok] >= self.curve(nmse[ok])
        return out

    def fit(self, noise_nsigma, noise_nmse, signal_nsigma, signal_nmse, max_false, seed=0):
        """Maximise the signals above g with at most ``max_false`` noise triggers above it.

        A small tie-breaker keeps the curve as high as possible between the noise
        triggers (it prefers the most conservative of the equally good curves).
        """
        nn, nm = np.asarray(noise_nsigma, float), np.asarray(noise_nmse, float)
        sn, sm = np.asarray(signal_nsigma, float), np.asarray(signal_nmse, float)
        n_ok = np.isfinite(nn) & np.isfinite(nm) & (nm > 0)
        s_ok = np.isfinite(sn) & np.isfinite(sm) & (sm > 0)
        xn, yn = np.log10(nm[n_ok]), nn[n_ok]
        xs, ys = np.log10(sm[s_ok]), sn[s_ok]
        span = max(np.ptp(np.concatenate([yn, ys])), 1.0)
        lo = float(np.min(np.concatenate([yn, ys])))

        def objective(params):
            values = self._values(params)
            false = np.count_nonzero(yn >= np.interp(xn, self.knots, values))
            true = np.count_nonzero(ys >= np.interp(xs, self.knots, values))
            return -true + 1e6 * max(0, false - max_false) - 1e-3 * np.mean(values) / span

        bounds = [(lo - 1.0, lo + span)] + [(0.0, span / 2)] * (len(self.knots) - 1)
        result = differential_evolution(objective, bounds, seed=seed, maxiter=600, popsize=30, tol=1e-8,
                                        polish=False)
        self.values = self._values(result.x)
        return self
