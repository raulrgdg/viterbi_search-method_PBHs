"""Filesystem layout of the repository.

Every location can be redirected with an environment variable, which is useful
on clusters where data live on a different filesystem than the code:

    PBH_VITERBI_ROOT     repository root (default: inferred from this file)
    PBH_VITERBI_O3_DIR   downloaded O3 strain, one O3b-packN folder per pack
    PBH_VITERBI_RESULTS  search outputs, logs and plots
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PBH_VITERBI_ROOT", Path(__file__).resolve().parents[2]))

DATA_DIR = PROJECT_ROOT / "data"
O3_DIR = Path(os.environ.get("PBH_VITERBI_O3_DIR", DATA_DIR / "o3"))
PAPER_DATA_DIR = PROJECT_ROOT / "paper_data"

RESULTS_DIR = Path(os.environ.get("PBH_VITERBI_RESULTS", PROJECT_ROOT / "results"))
SEARCH_RESULTS_DIR = RESULTS_DIR / "search"
SHARDS_DIR = RESULTS_DIR / "tmp" / "shards"
PLOTS_DIR = RESULTS_DIR / "plots"
LOGS_DIR = RESULTS_DIR / "logs"

# nsigma background: mean and std of the Viterbi track power in noise, per Tsft.
DEFAULT_NOISE_BACKGROUND = PAPER_DATA_DIR / "calibration" / "noise_power_background.csv"

MAKE_SFTS_SCRIPT = Path(__file__).resolve().parent / "sft" / "make_sfts.sh"


def ensure_dir(path):
    """Create ``path`` (and parents) if needed and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def o3_pack_dir(pack, o3_dir=None):
    """Return the folder holding the downloaded frames of one O3 pack."""
    return Path(o3_dir or O3_DIR) / f"O3b-pack{pack}"
