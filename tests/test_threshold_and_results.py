"""Paper threshold and result-file handling."""

import csv

import pandas as pd

from pbh_viterbi import results
from pbh_viterbi.paths import PAPER_DATA_DIR
from pbh_viterbi.threshold import is_candidate, load_threshold


SEARCH = PAPER_DATA_DIR / "search_results"


def test_paper_threshold_far():
    threshold = load_threshold(PAPER_DATA_DIR / "calibration" / "far_threshold.json")
    noise = pd.read_csv(SEARCH / "noise_search_O3b_H1_108packs.csv")
    signal = pd.read_csv(SEARCH / "injected_search_O3b_H1_90packs.csv")
    assert is_candidate(threshold, noise.nmse, noise.nsigma).sum() == 4  # 4 / 108 = 3.7%
    assert is_candidate(threshold, signal.nmse, signal.nsigma).sum() == 9940


def test_campaign_files():
    campaign = pd.read_csv(SEARCH / "injected_search_O3b_H1_90packs.csv")
    assert campaign.pack.nunique() == 90 and len(campaign) == 18004
    assert set(campaign.cluster) == {"HPC1", "HPC2", "HPC3"}
    assert pd.read_csv(SEARCH / "noise_search_O3b_H1_108packs.csv").pack.nunique() == 108


def test_distance_reach_example():
    """Worked example of the Wilson-interval note: Mc = 0.0178 Msun."""
    from pbh_viterbi.sensitivity import distance_reach

    threshold = load_threshold(PAPER_DATA_DIR / "calibration" / "far_threshold.json")
    signal = pd.read_csv(SEARCH / "injected_search_O3b_H1_90packs.csv")
    recovered = is_candidate(threshold, signal.nmse, signal.nsigma)
    reach = distance_reach(signal.mchirp, signal.distance, recovered).set_index("mchirp")
    row = reach.loc[reach.index[(reach.index - 0.0177945131298806).__abs__().argmin()]]
    assert round(row.d95 * 1e3, 2) == 107.71
    assert round(row.d95_low * 1e3, 2) == 100.0
    assert round(row.d95_high * 1e3, 2) == 107.71


def test_shards_are_merged(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "SHARDS_DIR", tmp_path / "shards")
    monkeypatch.setattr(results, "SEARCH_RESULTS_DIR", tmp_path / "search")
    search = {"candidate": True, "nmse": 1e-3, "nsigma": 5.0, "mass": 1e-2, "tsft": 9, "status": "candidate_found"}
    name = results.csv_name(injected=True, pack=1)
    for job in range(3):
        row = results.result_row(1, search, injected=True, mchirp=1e-2, distance=0.01 * (job + 1))
        results.append_rows([row], results.shard_path(name, 3, job))
        results.finish_shard(name, 3, job)

    merged = list(csv.DictReader((tmp_path / "search" / name).open()))
    assert [row["distance"] for row in merged] == ["0.01", "0.02", "0.03"]
    assert not list((tmp_path / "shards").glob("*.csv"))
