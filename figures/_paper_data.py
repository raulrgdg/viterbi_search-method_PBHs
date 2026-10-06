"""Loading the paper data products used by the result figures (Figs. 3, 5-7)."""

import pandas as pd

import _setup  # noqa: F401  (adds src/ to sys.path)
from pbh_viterbi.paths import PAPER_DATA_DIR
from pbh_viterbi.threshold import is_candidate, load_threshold

NOISE_CSV = PAPER_DATA_DIR / "search_results" / "noise_search_O3b_H1_108packs.csv"
CAMPAIGN_CSV = PAPER_DATA_DIR / "search_results" / "injected_search_O3b_H1_90packs.csv"
CAMPAIGN_60_CSV = PAPER_DATA_DIR / "search_results" / "injected_search_O3b_H1_60packs_april2026.csv"
THRESHOLD_JSON = PAPER_DATA_DIR / "calibration" / "far_threshold.json"
BACKGROUND_CSV = PAPER_DATA_DIR / "calibration" / "noise_power_background.csv"
SNR_CSV = PAPER_DATA_DIR / "snr" / "injection_snr_sky_median.csv"
PSD_CSV = PAPER_DATA_DIR / "snr" / "average_noise_psd_O3b_H1.csv"


def load_campaign(path=CAMPAIGN_CSV, threshold_path=THRESHOLD_JSON):
    """Injected-search triggers with a ``recovered`` column from the FAR threshold."""
    df = pd.read_csv(path)
    df["nmse"] = pd.to_numeric(df["nmse"], errors="coerce")
    df["recovered"] = is_candidate(load_threshold(threshold_path), df["nmse"], df["nsigma"])
    return df
