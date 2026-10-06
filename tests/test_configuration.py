"""Search configuration and campaign layout match the paper (Table I, Sec. III)."""

import numpy as np

from pbh_viterbi.campaign import assignment_for_pack, packs_for_cluster
from pbh_viterbi.config import DISTANCE_GRID, MCHIRP_GRID, TSFT_VALUES, t_to_merger_for_mchirp
from pbh_viterbi.jobs import split_for_job
from pbh_viterbi.o3.packs import ALL_PACKS, PACK_WINDOWS, parse_pack_list


def test_population_grid():
    assert len(MCHIRP_GRID) == 20 and len(DISTANCE_GRID) == 30
    assert np.isclose(MCHIRP_GRID.min(), 2e-4) and np.isclose(MCHIRP_GRID.max(), 1e-1)
    assert np.isclose(DISTANCE_GRID.min(), 1e-4) and np.isclose(DISTANCE_GRID.max(), 0.145)


def test_tsft_values():
    assert len(TSFT_VALUES) == 13 and min(TSFT_VALUES) == 2 and max(TSFT_VALUES) == 88


def test_packs_cover_about_1000_hours():
    assert ALL_PACKS == tuple(range(1, 109))
    assert sum(end - start for start, end in PACK_WINDOWS.values()) / 3600 > 980


def test_campaign_partition():
    clusters = {c: packs_for_cluster(c) for c in ("HPC1", "HPC2", "HPC3")}
    assert sorted(sum(clusters.values(), [])) == list(ALL_PACKS)
    assert all(len(packs) == 36 for packs in clusters.values())
    assert clusters["HPC1"][:12] == list(range(1, 13))
    slices = {(a.signal_start, a.signal_end) for a in map(assignment_for_pack, ALL_PACKS)}
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
