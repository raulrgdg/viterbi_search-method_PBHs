"""[UNDER TEST] Ranking statistics and thresholds of the threshold review (search/ranking.py)."""

import numpy as np
import pytest

from pbh_viterbi.search.ranking import (
    MonotoneThreshold,
    block_steps,
    clopper_pearson,
    drift_bins,
    linear_stat,
    pack_folds,
    threshold_for_far,
)
from pbh_viterbi.tsft import tsft_lim


def test_monotone_threshold_is_non_decreasing_and_flat_outside():
    rng = np.random.default_rng(0)
    noise_x, noise_y = rng.uniform(-3, 1, 200), rng.normal(0, 1, 200)
    sig_x, sig_y = rng.uniform(-8, 0, 400), rng.normal(2, 2, 400)
    th = MonotoneThreshold([-3, -2, -1, 0, 1]).fit(noise_y, 10.0 ** noise_x, sig_y, 10.0 ** sig_x, max_false=4)
    grid = 10.0 ** np.linspace(-12, 4, 400)
    curve = th.curve(grid)
    assert np.all(np.diff(curve) >= -1e-12)
    assert th.curve(1e-12) == pytest.approx(th.curve(1e-3)) and th.curve(1e4) == pytest.approx(th.curve(10.0))
    assert np.sum(th.passes(noise_y, 10.0 ** noise_x)) <= 4


def test_linear_stat_and_threshold():
    assert linear_stat([1.0], [1e-3], 2.0)[0] == pytest.approx(7.0)
    assert linear_stat([1.0], [np.nan], 2.0)[0] == -np.inf
    assert threshold_for_far(np.arange(10.0), 2) == 7.0


def test_block_features():
    assert block_steps(0.0, 320.0, 160.0) == 2.0
    # A track at the tracking limit moves one remapped bin per SFT step.
    m = 1e-2
    t = float(tsft_lim(m))
    assert drift_bins(m, 0.0, 10 * t, t) == pytest.approx(10.0, rel=1e-6)


def test_clopper_pearson_and_folds():
    lo, hi = clopper_pearson(0, 141)
    assert lo == 0.0 and 0 < hi < 0.02
    folds = pack_folds(range(1, 142), 3)
    assert sorted(np.bincount(list(folds.values()))) == [47, 47, 47]


def test_quantised_nmse_formula():
    from pbh_viterbi.search.ranking import quantised_nmse

    m, t = 1e-2, float(tsft_lim(1e-2))
    # A block of 50 steps at the tracking limit drifts D = 50 bins: NMSE_eff = NMSE + 1/(4 * 50^2).
    assert quantised_nmse(1e-6, m, 0.0, 50 * t, t) == pytest.approx(1e-6 + 0.25 / 50 ** 2)
    assert np.isinf(quantised_nmse(1e-6, m, 0.0, 0.0, t))  # no drift, no chirp information


@pytest.mark.parametrize("drift", [10, 24, 100, 300])
def test_quantisation_limit_matches_the_fit(drift):
    """Fitting straight lines rounded to integer bins gives NMSE ~ 1/(4 D^2) (the coefficient is not tuned)."""
    from pbh_viterbi.config import FBAND, FMAX, FMIN, NMSE_QUANT_COEFF
    from pbh_viterbi.search.fitting import fit_slope_windows, mass_grid

    rng = np.random.default_rng(drift)
    tsft = 10.0
    bin_width = (FMIN ** (-8 / 3) - FMAX ** (-8 / 3)) / (FBAND * tsft)
    nmse = []
    for _ in range(30):
        n_steps = int(drift / rng.uniform(0.3, 0.95)) + 1  # non-integer slope, 0.3-0.95 bin per step
        line = np.linspace(0, drift, n_steps) + rng.uniform(0, 1)
        track = -np.floor(line + 0.5) * bin_width  # the Viterbi track takes integer bins
        nmse.append(fit_slope_windows(track, tsft, 1, mass_grid())[4][0])
    ratio = np.median(nmse) / (NMSE_QUANT_COEFF / drift ** 2)
    assert 0.7 < ratio < 1.5
