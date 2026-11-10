"""Filesystem layout of the repository.

Every location can be redirected with an environment variable, which is useful
on clusters where data live on a different filesystem than the code:

    PBH_VITERBI_ROOT     repository root (default: inferred from this file)
    PBH_VITERBI_O3_DIR   downloaded O3 strain, one O3b-packN folder per pack
    PBH_VITERBI_RESULTS  search outputs, logs and plots
"""

import os
from pathlib import Path

from pbh_viterbi.config import DEFAULT_TSFT_LIST, DETECTORS, MAP_NORMALISATION, SFT_HIGHPASS

PROJECT_ROOT = Path(os.environ.get("PBH_VITERBI_ROOT", Path(__file__).resolve().parents[2]))

DATA_DIR = PROJECT_ROOT / "data"
O3_DIR = Path(os.environ.get("PBH_VITERBI_O3_DIR", DATA_DIR / "o3"))
PAPER_DATA_DIR = PROJECT_ROOT / "paper_data"

RESULTS_DIR = Path(os.environ.get("PBH_VITERBI_RESULTS", PROJECT_ROOT / "results"))
SEARCH_RESULTS_DIR = RESULTS_DIR / "search"
SHARDS_DIR = RESULTS_DIR / "tmp" / "shards"
PLOTS_DIR = RESULTS_DIR / "plots"
LOGS_DIR = RESULTS_DIR / "logs"

# nsigma background: mean and std of the Viterbi track power in noise, per Tsft,
# one file per Tsft list and map configuration (band FMIN-FMAX). The files without
# suffix are the v1.2 maps (H1, per-segment high-pass, per-bin normalisation).
CALIBRATION_DIR = PROJECT_ROOT / "calibration"
# Background of the paper (13 Tsft, 61.1-127.2 Hz, v1.0 maps); kept for the regression tests.
PAPER_NOISE_BACKGROUND = PAPER_DATA_DIR / "calibration" / "noise_power_background.csv"


def config_suffix(detectors=DETECTORS, sft_highpass=SFT_HIGHPASS, normalisation=MAP_NORMALISATION):
    """File-name suffix of a map configuration.

    The v1.2 maps (H1, per-segment high-pass, per-bin normalisation) have no suffix, so the
    files of v1.1-v1.2 keep their names; other cases add ``_H1L1``, ``_chunkhp`` and
    ``_sftnorm``, e.g. ``_chunkhp_sftnorm`` for the v1.3 default.
    """
    suffix = "" if tuple(detectors) == ("H1",) else "_" + "".join(detectors)
    suffix += "" if sft_highpass == "segment" else f"_{sft_highpass}hp"
    suffix += "" if normalisation == "bin" else "_sftnorm"
    return suffix


def noise_background_path(tsft_list, detectors=DETECTORS, sft_highpass=SFT_HIGHPASS, normalisation=MAP_NORMALISATION):
    """nsigma background file of a Tsft list ("95", "90" or "paper") and map configuration."""
    return CALIBRATION_DIR / f"noise_background_tsft{tsft_list}{config_suffix(detectors, sft_highpass, normalisation)}.csv"


def noise_track_power_path(detectors=DETECTORS, sft_highpass=SFT_HIGHPASS, normalisation=MAP_NORMALISATION):
    """Per-pack noise track power table of a map configuration (analysis/noise_background_stats.py)."""
    return CALIBRATION_DIR / f"noise_track_power{config_suffix(detectors, sft_highpass, normalisation)}.csv"


DEFAULT_NOISE_BACKGROUND = noise_background_path(DEFAULT_TSFT_LIST)

MAKE_SFTS_SCRIPT = Path(__file__).resolve().parent / "sft" / "make_sfts.sh"


def ensure_dir(path):
    """Create ``path`` (and parents) if needed and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def o3_pack_dir(pack, o3_dir=None):
    """Return the folder holding the downloaded frames of one O3 pack."""
    return Path(o3_dir or O3_DIR) / f"O3b-pack{pack}"
