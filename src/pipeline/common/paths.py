from pathlib import Path


COMMON_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = COMMON_DIR.parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SCRIPTS_UTILS_DIR = SCRIPTS_DIR / "utils"
WORKFLOWS_DIR = PROJECT_ROOT / "workflows"
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
STUDIES_DIR = PROJECT_ROOT / "studies"
STRONG_SCALING_DIR = STUDIES_DIR / "strong_scaling"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_RAW_O3_DIR = DATA_RAW_DIR / "o3"
DATA_INTERIM_DIR = DATA_DIR / "interim"
DATA_PRODUCTS_DIR = DATA_DIR / "products"
RESULTS_REPORTS_DIR = RESULTS_DIR / "reports"
RESULTS_PLOTS_DIR = RESULTS_DIR / "plots"
RESULTS_LOGS_DIR = RESULTS_DIR / "logs"
RESULTS_TMP_DIR = RESULTS_DIR / "tmp"

# Compatibility aliases mapped to the new canonical layout.
BIN_DIR = SCRIPTS_DIR
CONDOR_DIR = PROJECT_ROOT / "condor"
INPUTS_DIR = DATA_DIR
INPUTS_O3_DATA_DIR = DATA_RAW_O3_DIR
INPUTS_DATA_DIR = DATA_PRODUCTS_DIR
OUTPUTS_DIR = RESULTS_DIR
OUTPUTS_REPORTS_DIR = RESULTS_REPORTS_DIR
OUTPUTS_PLOTS_DIR = RESULTS_PLOTS_DIR
LOGS_DIR = RESULTS_LOGS_DIR


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_pack_data_dir(pack: int) -> Path:
    """Return the per-pack data directory used as search input."""
    return ensure_dir(INPUTS_DATA_DIR / f"pack-{pack}")


def get_pack_product_dir(pack: int, product_name: str, data_kind: str) -> Path:
    """Return a per-pack directory for a specific generated product."""
    return ensure_dir(get_pack_data_dir(pack) / product_name / data_kind)
