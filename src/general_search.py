import csv
import random

import numpy as np

from pipeline_paths import OUTPUTS_REPORTS_DIR, ensure_dir
from power_metric_prueba import search_candidates

# ----------------------------- User configuration -----------------------------
PACK_RANGE = range(1, 109)
MCHIRP_GRID = [1e-04, 5e-04, 8e-04, 1e-03, 2e-03, 3e-03, 4e-03, 5e-03, 6e-03, 7e-03, 8e-03, 9e-03, 1e-02, 2e-02, 3e-02, 4e-02, 5e-02, 6e-02, 7e-02, 8e-02, 9e-02, 1e-01]
DISTANCE_GRID = np.concatenate([np.array([0.001]), np.arange(0.005, 0.155, 0.005)])
SIGNAL_PACK = 3
TSFT_LIST = [2, 3, 4, 5, 6, 9, 12, 15, 21, 29, 39, 54, 74, 101, 132, 181, 248, 340]

N_RANDOM_NOISE_TARGETS = 10
N_RANDOM_SIGNAL_TARGETS = 10
RANDOM_SEED = 42
# ---------------------------------------------------------------------------

REPORTS_DIR = ensure_dir(OUTPUTS_REPORTS_DIR)


def _resolve_result(result):
    """Normalize the variable-length search result tuple."""
    if result[0] is None:
        if len(result) == 2:
            res, opt_nsigma = result
            return res, None, opt_nsigma, None
        res, nmse_eval, opt_nsigma, extended_optimal_block = result
        return res, nmse_eval, opt_nsigma, extended_optimal_block

    res, nmse_eval, opt_nsigma, extended_optimal_block = result
    return res, nmse_eval, opt_nsigma, extended_optimal_block


def _random_noise_packs(rng):
    """Pick random noise packs from the configured range."""
    packs = list(PACK_RANGE)
    n_targets = min(N_RANDOM_NOISE_TARGETS, len(packs))
    return rng.sample(packs, n_targets)


def _random_signal_targets(rng):
    """Pick random ``(mchirp, distance)`` targets from the configured grids."""
    signal_targets = [(mchirp, distance) for mchirp in MCHIRP_GRID for distance in DISTANCE_GRID]
    n_targets = min(N_RANDOM_SIGNAL_TARGETS, len(signal_targets))
    return rng.sample(signal_targets, n_targets)


def _append_noise_rows(rows, rng):
    """Run the search pipeline on random noise targets."""
    for pack in _random_noise_packs(rng):
        result = search_candidates(
            mchirp=None,
            distance=None,
            tsft_list=TSFT_LIST,
            pack=pack,
            noise=True,
        )
        res, nmse_eval, opt_nsigma, extended_optimal_block = _resolve_result(result)
        final_mass = ""
        if extended_optimal_block is not None:
            final_mass_value = extended_optimal_block.get("final_mass")
            if final_mass_value is not None:
                final_mass = float(final_mass_value)

        rows.append(
            {
                "candidate": bool(res),
                "nmse": nmse_eval if nmse_eval is not None else "",
                "nsigma": opt_nsigma if opt_nsigma is not None else "",
                "mass": final_mass,
            }
        )


def _append_signal_rows(rows, rng):
    """Run the search pipeline on random signal targets."""
    for mchirp, distance in _random_signal_targets(rng):
        result = search_candidates(
            mchirp=mchirp,
            distance=distance,
            tsft_list=TSFT_LIST,
            pack=SIGNAL_PACK,
            noise=False,
        )
        res, nmse_eval, opt_nsigma, extended_optimal_block = _resolve_result(result)
        final_mass = ""
        if extended_optimal_block is not None:
            final_mass_value = extended_optimal_block.get("final_mass")
            if final_mass_value is not None:
                final_mass = float(final_mass_value)

        rows.append(
            {
                "candidate": bool(res),
                "nmse": nmse_eval if nmse_eval is not None else "",
                "nsigma": opt_nsigma if opt_nsigma is not None else "",
                "mass": final_mass,
            }
        )


def main():
    """Run the same search pipeline over random noise and signal targets."""
    rng = random.Random(RANDOM_SEED)
    rows = []

    _append_noise_rows(rows, rng)
    _append_signal_rows(rows, rng)

    output_csv = REPORTS_DIR / "general_search_results.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate", "nmse", "nsigma", "mass"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Targets procesados: {len(rows)}")
    print(f"CSV guardado en: {output_csv}")


if __name__ == "__main__":
    main()
