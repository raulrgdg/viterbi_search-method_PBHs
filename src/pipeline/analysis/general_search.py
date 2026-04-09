import csv
import random

import numpy as np

from pipeline.common.paths import OUTPUTS_REPORTS_DIR, ensure_dir
from pipeline.search_metrics import search_candidates

PACK_RANGE = range(1, 109)
MCHIRP_GRID = [1e-03, 2e-03, 3e-03, 4e-03, 5e-03, 6e-03, 7e-03, 8e-03, 9e-03, 1e-02, 2e-02, 3e-02, 4e-02, 5e-02, 6e-02, 7e-02, 8e-02, 9e-02, 1e-01]
DISTANCE_GRID = np.concatenate([np.array([0.001]), np.arange(0.005, 0.155, 0.005)])
SIGNAL_PACK = 3
TSFT_LIST = [2, 3, 4, 5, 6, 9, 12, 15, 21, 29, 39, 54, 74, 101, 132, 181, 248, 340]
N_RANDOM_NOISE_TARGETS = 10
N_RANDOM_SIGNAL_TARGETS = 10
RANDOM_SEED = 42

REPORTS_DIR = ensure_dir(OUTPUTS_REPORTS_DIR)


def _resolve_result(result):
    if result[0] is None:
        if len(result) == 2:
            res, opt_nsigma = result
            return res, None, opt_nsigma, None
        res, nmse_eval, opt_nsigma, extended_optimal_block = result
        return res, nmse_eval, opt_nsigma, extended_optimal_block
    res, nmse_eval, opt_nsigma, extended_optimal_block = result
    return res, nmse_eval, opt_nsigma, extended_optimal_block


def _random_noise_packs(rng):
    packs = list(PACK_RANGE)
    return rng.sample(packs, min(N_RANDOM_NOISE_TARGETS, len(packs)))


def _random_signal_targets(rng):
    signal_targets = [(mchirp, distance) for mchirp in MCHIRP_GRID for distance in DISTANCE_GRID]
    return rng.sample(signal_targets, min(N_RANDOM_SIGNAL_TARGETS, len(signal_targets)))


def _append_noise_rows(rows, rng):
    for pack in _random_noise_packs(rng):
        result = search_candidates(mchirp=None, distance=None, tsft_list=TSFT_LIST, pack=pack, noise=True)
        res, nmse_eval, opt_nsigma, extended_optimal_block = _resolve_result(result)
        final_mass = ""
        if extended_optimal_block is not None and extended_optimal_block.get("final_mass") is not None:
            final_mass = float(extended_optimal_block["final_mass"])
        rows.append({"candidate": bool(res), "nmse": nmse_eval if nmse_eval is not None else "", "nsigma": opt_nsigma if opt_nsigma is not None else "", "mass": final_mass, "injected": False})


def _append_signal_rows(rows, rng):
    for mchirp, distance in _random_signal_targets(rng):
        result = search_candidates(mchirp=mchirp, distance=distance, tsft_list=TSFT_LIST, pack=SIGNAL_PACK, noise=False)
        res, nmse_eval, opt_nsigma, extended_optimal_block = _resolve_result(result)
        final_mass = ""
        if extended_optimal_block is not None and extended_optimal_block.get("final_mass") is not None:
            final_mass = float(extended_optimal_block["final_mass"])
        rows.append({"candidate": bool(res), "nmse": nmse_eval if nmse_eval is not None else "", "nsigma": opt_nsigma if opt_nsigma is not None else "", "mass": final_mass, "injected": True})


def main():
    rng = random.Random(RANDOM_SEED)
    rows = []
    _append_noise_rows(rows, rng)
    _append_signal_rows(rows, rng)

    output_csv = REPORTS_DIR / "general_search_results.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate", "nmse", "nsigma", "mass", "injected"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Targets procesados: {len(rows)}")
    print(f"CSV guardado en: {output_csv}")


if __name__ == "__main__":
    main()
