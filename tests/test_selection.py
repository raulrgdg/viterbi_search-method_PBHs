"""[UNDER TEST] Tsft-selection rules of the Tsft-selection study (search/selection.py)."""

import numpy as np
import pandas as pd
import pytest

from pbh_viterbi.config import TSFT_LISTS
from pbh_viterbi.search.candidates import per_tsft_candidates
from pbh_viterbi.search.selection import RULES, apply_rule, empirical_pvalues, prepare, threshold_at_far

SHORT, LONG = 2.625, 293.25  # both in the 95 list; 2.625 also close to the paper's 2-3 s


def chunk(pack, signal_id, injected, rows):
    """Per-Tsft rows of one chunk: rows = [(tsft, nsigma, iso_status, nmse, mass), ...]."""
    return [dict(pack=pack, signal_id=signal_id, injected=injected, tsft=t, nsigma=z, iso_status=s, nmse=e, mass=m)
            for t, z, s, e, m in rows]


@pytest.fixture
def tables():
    rng = np.random.default_rng(0)
    noise = []
    for pack in range(1, 41):
        noise += chunk(pack, None, 0, [(SHORT, rng.normal(), "ok", 0.5, 1e-3),
                                       (LONG, rng.normal(), "failed_third_fit_check", None, None)])
    # Pack 25 case: a heavy signal, weak at the short Tsft but well fitted; a noise
    # fluctuation wins nsigma at the long Tsft, where the isolation fails.
    signal = chunk(25, 590, 1, [(SHORT, 2.0, "ok", 1e-10, 0.1), (LONG, 2.4, "failed_third_fit_check", None, None)])
    return prepare(pd.DataFrame(noise)), prepare(pd.DataFrame(signal))


def test_max_nsigma_picks_noise_tsft_and_rejects(tables):
    _noise, signal = tables
    sel = apply_rule("B_max_nsigma_95", signal)
    assert sel.loc[0, "tsft"] == LONG and sel.loc[0, "stat"] == -np.inf


@pytest.mark.parametrize("rule", ["D_isolated_95", "D_topk_nmse_95", "E_mass_consistent_95"])
def test_isolation_rules_recover_short_tsft(tables, rule):
    _noise, signal = tables
    sel = apply_rule(rule, signal)
    assert sel.loc[0, "tsft"] == SHORT and sel.loc[0, "stat"] == pytest.approx(2.0)


def test_mass_consistency_drops_untrackable_tsft():
    # Isolated at the long Tsft with a fitted mass whose T_lim (2.6 s) is far shorter.
    df = prepare(pd.DataFrame(chunk(1, 0, 1, [(SHORT, 1.0, "ok", 1e-3, 0.1), (LONG, 3.0, "ok", 1e-3, 0.1)])))
    assert apply_rule("D_isolated_95", df).loc[0, "tsft"] == LONG
    assert apply_rule("E_mass_consistent_95", df).loc[0, "tsft"] == SHORT


def test_pvalues_leave_own_pack_out(tables):
    noise, _signal = tables
    row = noise[(noise["pack"] == 3) & (noise["tsft"] == SHORT)]
    others = noise[(noise["pack"] != 3) & (noise["tsft"] == SHORT)]["nsigma"].to_numpy()
    expected = (np.sum(others >= row["nsigma"].iloc[0]) + 1) / (len(others) + 1)
    assert empirical_pvalues(row, noise)[0] == pytest.approx(expected)


def test_tail_pvalue_below_empirical_floor(tables):
    noise, signal = tables
    loud = signal.assign(nsigma=10.0)
    p_emp = empirical_pvalues(loud, noise)
    p_tail = empirical_pvalues(loud, noise, tail_quantile=0.9)
    # 40 noise packs, pack 25 (the chunk's own) left out: empirical floor 1/40.
    assert np.all(p_emp == pytest.approx(1 / 40)) and np.all(p_tail < p_emp)


def test_tail_rule_keeps_very_loud_signal(tables):
    noise, _signal = tables
    loud = prepare(pd.DataFrame(chunk(25, 1, 1, [(SHORT, 5000.0, "ok", 1e-9, 0.1), (LONG, 3.0, "ok", 1.0, 1e-4)])))
    sel = apply_rule("C_min_p_tail_95", loud, noise)
    assert sel.loc[0, "tsft"] == SHORT and np.isfinite(sel.loc[0, "stat"])


def test_every_rule_returns_one_row_per_chunk(tables):
    noise, signal = tables
    both = pd.concat([noise, signal], ignore_index=True)
    for rule in RULES:
        sel = apply_rule(rule, both, noise)
        assert len(sel) == 41 and set(sel["rule"]) == {rule}


def test_threshold_at_far():
    noise = np.arange(100.0)
    threshold, reached, n = threshold_at_far(noise, 0.03)
    assert (threshold, reached, n) == (96.0, 3, 100)
    assert threshold_at_far(noise, 0.0)[1] == 0


def test_per_tsft_candidates_matches_search(tables):
    from synthetic import synthetic_products

    from pbh_viterbi.paths import PAPER_NOISE_BACKGROUND
    from pbh_viterbi.search.candidates import search_candidate
    from pbh_viterbi.search.power import load_noise_background

    means, stds = load_noise_background(PAPER_NOISE_BACKGROUND)
    products = [synthetic_products(amplitude=0.0, seed=4), synthetic_products(amplitude=20.0, tsft=88)]
    rows = per_tsft_candidates(products, means, stds)
    best = search_candidate(products, means, stds)
    chosen = next(r for r in rows if r["tsft"] == best["tsft"])
    assert len(rows) == len(products) and chosen["nsigma"] == pytest.approx(best["nsigma"])
    assert chosen["iso_status"] == "ok" and chosen["nmse"] == pytest.approx(best["nmse"])
    assert set(TSFT_LISTS["union"]) >= set(TSFT_LISTS["95"]) | set(TSFT_LISTS["paper"])
