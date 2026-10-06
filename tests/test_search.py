"""Regression tests of the candidate search.

Reference values were produced with the code used for the paper results
(identical output verified against it); any change here changes the science.
"""

import numpy as np
import pytest
from synthetic import synthetic_products

from pbh_viterbi.paths import DEFAULT_NOISE_BACKGROUND
from pbh_viterbi.search.candidates import STATUS_BELOW_THRESHOLD, STATUS_CANDIDATE, search_candidate
from pbh_viterbi.search.fitting import fit_slope_windows, mass_grid, slope_of_mass
from pbh_viterbi.search.power import load_noise_background

CASES = [
    (dict(amplitude=20.0), STATUS_CANDIDATE, 2.9043097502328036e-07, 27.193242043903346, 0.009998848716163801),
    (dict(mchirp=3e-3, amplitude=1.0, seed=1), STATUS_CANDIDATE, 0.00010530084622096395, 0.4879053245381729,
     0.0029958759471652523),
    (dict(duration_frac=0.0, seed=2), STATUS_BELOW_THRESHOLD, 2.1646712834880844, -0.864807374149339,
     0.0035361235156470532),
    (dict(mchirp=5e-2, duration_frac=0.05, amplitude=40, seed=3, tsft=88), STATUS_CANDIDATE, 2.1942729771918586e-07,
     4.782863753587656, 0.05000172062612015),
]


@pytest.fixture(scope="module")
def background():
    return load_noise_background(DEFAULT_NOISE_BACKGROUND)


@pytest.mark.parametrize("kwargs,status,nmse,nsigma,mass", CASES)
def test_search_regression(background, kwargs, status, nmse, nsigma, mass):
    result = search_candidate([synthetic_products(**kwargs)], *background)
    assert result["status"] == status
    assert result["nmse"] == pytest.approx(nmse, rel=1e-12)
    assert result["nsigma"] == pytest.approx(nsigma, rel=1e-12)
    assert result["mass"] == pytest.approx(mass, rel=1e-12)


def test_best_tsft_is_selected(background):
    products = [synthetic_products(amplitude=0.0, seed=4), synthetic_products(amplitude=20.0)]
    assert search_candidate(products, *background)["tsft"] == 63


def test_fit_recovers_chirp_mass():
    tsft, mchirp = 10, 4e-3
    track = slope_of_mass(mchirp) * np.arange(500) * tsft
    _, _, _, best_mass, best_nmse = fit_slope_windows(track, tsft, 1, mass_grid())
    assert best_mass[0] == pytest.approx(mchirp, rel=1e-3)
    assert best_nmse[0] < 1e-6
