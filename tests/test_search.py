"""Regression tests of the candidate search.

Reference values were produced with the code used for the paper results
(identical output verified against it) and updated in v1.1, where the
frequency grid of the remapped maps starts at the first SFT bin instead of the
nominal FMIN (same statuses, chirp masses within 0.1%). Any change here
changes the science.
"""

import numpy as np
import pytest
from synthetic import synthetic_products

from pbh_viterbi.paths import PAPER_NOISE_BACKGROUND
from pbh_viterbi.search.candidates import STATUS_BELOW_THRESHOLD, STATUS_CANDIDATE, search_candidate
from pbh_viterbi.search.fitting import fit_slope_windows, mass_grid, slope_of_mass
from pbh_viterbi.search.power import load_noise_background

CASES = [
    (dict(amplitude=20.0), STATUS_CANDIDATE, 5.585267255785663e-06, 27.346808529379746, 0.010008062699408052),
    (dict(mchirp=3e-3, amplitude=1.0, seed=1), STATUS_CANDIDATE, 0.00011540057392794976, 0.4043111400612557,
     0.002994496543979514),
    (dict(duration_frac=0.0, seed=2), STATUS_BELOW_THRESHOLD, 2.164671498239518, -0.864807374149339,
     0.0035361235156470532),
    (dict(mchirp=5e-2, duration_frac=0.05, amplitude=40, seed=3, tsft=88), STATUS_CANDIDATE, 2.2539760381352335e-07,
     4.798959220368315, 0.05000172062612015),
]


@pytest.fixture(scope="module")
def background():
    return load_noise_background(PAPER_NOISE_BACKGROUND)


@pytest.mark.parametrize("kwargs,status,nmse,nsigma,mass", CASES)
def test_search_regression(background, kwargs, status, nmse, nsigma, mass):
    """v1.0/v1.1 search (max-nsigma selection) is unchanged."""
    result = search_candidate([synthetic_products(**kwargs)], *background, method="max_nsigma")
    assert result["status"] == status
    assert result["nmse"] == pytest.approx(nmse, rel=1e-12)
    assert result["nsigma"] == pytest.approx(nsigma, rel=1e-12)
    assert result["mass"] == pytest.approx(mass, rel=1e-12)


def test_best_tsft_is_selected(background):
    products = [synthetic_products(amplitude=0.0, seed=4), synthetic_products(amplitude=20.0)]
    assert search_candidate(products, *background, method="max_nsigma")["tsft"] == 63


@pytest.mark.parametrize("kwargs,status,nmse,nsigma,mass", CASES)
def test_hybrid_matches_isolation_on_a_single_tsft(background, kwargs, status, nmse, nsigma, mass):
    """With one Tsft the hybrid selection isolates the same candidate as v1.1."""
    from pbh_viterbi.config import RANKING_NMSE_WEIGHT

    result = search_candidate([synthetic_products(**kwargs)], *background, method="hybrid")
    assert result["selection"] == "hybrid" and len(result["per_tsft"]) == 1
    assert result["nsigma"] == pytest.approx(nsigma, rel=1e-12)
    assert result["nmse"] == pytest.approx(nmse, rel=1e-12) and result["mass"] == pytest.approx(mass, rel=1e-12)
    assert result["ranking_stat"] == pytest.approx(nsigma - RANKING_NMSE_WEIGHT * np.log10(result["nmse_eff"]))
    assert result["status"] in ("candidate_found", "rejected_ranking_threshold")


def test_hybrid_prefers_the_well_fitted_tsft(background):
    """A loud but unfitted map loses against a weaker clean inspiral."""
    products = [synthetic_products(amplitude=0.0, seed=4), synthetic_products(amplitude=20.0, tsft=88)]
    result = search_candidate(products, *background, method="hybrid")
    assert result["tsft"] == 88 and result["selection"] == "hybrid" and len(result["per_tsft"]) == 2


def test_fit_recovers_chirp_mass():
    tsft, mchirp = 10, 4e-3
    track = slope_of_mass(mchirp) * np.arange(500) * tsft
    _, _, _, best_mass, best_nmse = fit_slope_windows(track, tsft, 1, mass_grid())
    assert best_mass[0] == pytest.approx(mchirp, rel=1e-3)
    assert best_nmse[0] < 1e-6
