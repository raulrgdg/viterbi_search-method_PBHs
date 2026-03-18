from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
BIN_DIR = PROJECT_ROOT / "bin"
CONDOR_DIR = PROJECT_ROOT / "condor"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DATA_DIR = OUTPUTS_DIR / "data"
OUTPUTS_REPORTS_DIR = OUTPUTS_DIR / "reports"
OUTPUTS_PLOTS_DIR = OUTPUTS_DIR / "plots"
LOGS_DIR = OUTPUTS_DIR / "logs"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
