"""Search configuration and campaign layout match the paper (Table I, Sec. III)."""

import numpy as np

from pbh_viterbi.campaign import assignment_for_pack, packs_for_cluster
from pbh_viterbi.config import DISTANCE_GRID, MCHIRP_GRID, SAMPLE_RATE, TSFT_LISTS, t_to_merger_for_mchirp
from pbh_viterbi.jobs import split_for_job
from pbh_viterbi.o3.packs import ALL_PACKS, PACK_WINDOWS, PAPER_PACKS, parse_pack_list


def test_population_grid():
    assert len(MCHIRP_GRID) == 20 and len(DISTANCE_GRID) == 30
    assert np.isclose(MCHIRP_GRID.min(), 2e-4) and np.isclose(MCHIRP_GRID.max(), 1e-1)
    assert np.isclose(DISTANCE_GRID.min(), 1e-4) and np.isclose(DISTANCE_GRID.max(), 0.145)


def test_tsft_lists():
    assert len(TSFT_LISTS["paper"]) == 13 and min(TSFT_LISTS["paper"]) == 2 and max(TSFT_LISTS["paper"]) == 88
    assert len(TSFT_LISTS["95"]) == 45 and len(TSFT_LISTS["90"]) == 24
    for values in TSFT_LISTS.values():
        assert list(values) == sorted(set(values))
        assert all(float(t * SAMPLE_RATE).is_integer() for t in values)  # whole number of samples per SFT


def test_tsft_lists_cover_the_mass_range():
    """Every chirp mass keeps nsigma / max >= 0.947 (0.897) with the largest Tsft <= T_lim.

    The lists were optimised for 0.95 (0.90) on a coarser mass grid and rounded to 1/8 s.
    """
    from pbh_viterbi.tsft import tsft_coverage

    mchirps = np.geomspace(2e-4, 0.1, 2000)
    assert tsft_coverage(TSFT_LISTS["95"], mchirps).min() >= 0.947
    assert tsft_coverage(TSFT_LISTS["90"], mchirps).min() >= 0.896


def test_packs():
    assert PAPER_PACKS == tuple(range(1, 109)) and ALL_PACKS == tuple(range(1, 142))
    assert sum(PACK_WINDOWS[p][1] - PACK_WINDOWS[p][0] for p in PAPER_PACKS) / 3600 > 980
    windows = sorted(PACK_WINDOWS.values())
    assert all(end - start == 32768 for start, end in windows)
    assert all(windows[i][1] <= windows[i + 1][0] for i in range(len(windows) - 1))  # no overlaps


def test_paper_campaign_packs():
    from pbh_viterbi.campaign import campaign_packs

    packs = campaign_packs("paper")
    assert len(packs) == 90 and len(campaign_packs("all")) == 108
    assert sorted({assignment_for_pack(p).cluster for p in packs}) == ["HPC1", "HPC2", "HPC3"]


def test_campaign_partition():
    clusters = {c: packs_for_cluster(c) for c in ("HPC1", "HPC2", "HPC3")}
    assert sorted(sum(clusters.values(), [])) == list(PAPER_PACKS)
    assert all(len(packs) == 36 for packs in clusters.values())
    assert clusters["HPC1"][:12] == list(range(1, 13))
    slices = {(a.signal_start, a.signal_end) for a in map(assignment_for_pack, PAPER_PACKS)}
    assert slices == {(0, 200), (200, 400), (400, 600)}


def test_t_to_merger():
    assert t_to_merger_for_mchirp(1e-4) == 1.3939520546672462e07
    assert t_to_merger_for_mchirp(1e-2) == 32780


def test_split_for_job_covers_every_target_once():
    targets = list(range(200))
    for n_jobs in (1, 7, 200, 250):
        chunks = [split_for_job(targets, n_jobs, j) for j in range(n_jobs)]
        assert sum(chunks, []) == targets


def test_parse_pack_list():
    assert parse_pack_list("1-3,7") == [1, 2, 3, 7]
    assert parse_pack_list("all") == list(ALL_PACKS)
