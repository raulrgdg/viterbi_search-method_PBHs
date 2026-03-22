from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
BIN_DIR = PROJECT_ROOT / "bin"
CONDOR_DIR = PROJECT_ROOT / "condor"
INPUTS_DIR = PROJECT_ROOT / "inputs"
INPUTS_DATA_DIR = INPUTS_DIR / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_REPORTS_DIR = OUTPUTS_DIR / "reports"
OUTPUTS_PLOTS_DIR = OUTPUTS_DIR / "plots"
LOGS_DIR = OUTPUTS_DIR / "logs"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_pack_data_dir(pack: int) -> Path:
    """Return the per-pack data directory used as search input."""
    return ensure_dir(INPUTS_DATA_DIR / f"pack-{pack}")


def get_pack_product_dir(pack: int, product_name: str, data_kind: str) -> Path:
    """Return a per-pack directory for a specific generated product."""
    return ensure_dir(get_pack_data_dir(pack) / product_name / data_kind)
