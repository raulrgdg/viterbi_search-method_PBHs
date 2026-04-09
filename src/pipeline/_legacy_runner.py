from __future__ import annotations

import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LEGACY_SRC_DIR = PROJECT_ROOT / "src"


def run_legacy_script(filename: str) -> None:
    """Execute one legacy src/ script as the current __main__ entrypoint."""
    legacy_src = str(LEGACY_SRC_DIR)
    if legacy_src not in sys.path:
        sys.path.insert(0, legacy_src)
    runpy.run_path(str(LEGACY_SRC_DIR / filename), run_name="__main__")
