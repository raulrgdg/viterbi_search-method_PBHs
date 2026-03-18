import csv
from pathlib import Path

import numpy as np

from pipeline_paths import OUTPUTS_REPORTS_DIR, ensure_dir
from power_metric_prueba import search_candidates

MCHIRP_GRID = [1e-04, 5e-04, 8e-04, 1e-03, 2e-03, 3e-03, 4e-03, 5e-03, 6e-03, 7e-03, 8e-03, 9e-03, 1e-02, 2e-02, 3e-02, 4e-02, 5e-02, 6e-02, 7e-02, 8e-02, 9e-02, 1e-01]
DISTANCE_GRID = np.concatenate([np.array([0.001]), np.arange(0.005, 0.155, 0.005)])
TSFT_LIST = [2, 3, 4, 5, 6, 9, 12, 15, 21, 29, 39, 54, 74, 101, 132, 181, 248, 340]
PACKS_LIST = list(range(1, 109))
NOISE_MODE = True
SIGNAL_PACK = 3
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


def _run_noise_search():
    """Run the search over all noise packs and collect CSV rows."""
    rows = []
    skipped = 0

    for pack in PACKS_LIST:
        result = search_candidates(
            mchirp=None,
            distance=None,
            tsft_list=TSFT_LIST,
            pack=pack,
            noise=True,
        )
        res, nmse_eval, opt_nsigma, _extended_optimal_block = _resolve_result(result)
        candidate = bool(res)
        if not candidate:
            print(f"[SKIP] pack={pack}, noise")
            skipped += 1

        rows.append(
            {
                "candidate": candidate,
                "final_nmse": nmse_eval if nmse_eval is not None else "",
                "opt_nsigma": opt_nsigma if opt_nsigma is not None else "",
            }
        )

    return rows, skipped


def _run_signal_search():
    """Run the search over the injected signal grid and collect CSV rows."""
    rows = []
    skipped = 0

    for mchirp in MCHIRP_GRID:
        for distance in DISTANCE_GRID:
            result = search_candidates(
                mchirp,
                distance,
                tsft_list=TSFT_LIST,
                pack=SIGNAL_PACK,
                noise=False,
            )
            res, nmse_eval, opt_nsigma, extended_optimal_block = _resolve_result(result)
            candidate = bool(res)
            if not candidate:
                print(f"[SKIP] mchirp={mchirp:.0e}, distance={distance}")
                skipped += 1

            final_mass = ""
            final_slope = ""
            if extended_optimal_block is not None:
                final_mass_value = extended_optimal_block.get("final_mass")
                final_slope_value = extended_optimal_block.get("final_slope")
                if final_mass_value is not None:
                    final_mass = float(final_mass_value)
                if final_slope_value is not None:
                    final_slope = float(final_slope_value)

            rows.append(
                {
                    "real_mchirp": mchirp,
                    "real_distance": distance,
                    "candidate": candidate,
                    "final_nmse": nmse_eval if nmse_eval is not None else "",
                    "opt_nsigma": opt_nsigma if opt_nsigma is not None else "",
                    "final_mass": final_mass,
                    "final_slope": final_slope,
                }
            )

    return rows, skipped


def main():
    """Run the candidate search workflow and write the summary CSV."""
    if NOISE_MODE:
        output_csv = REPORTS_DIR / "search_results_noise_final-big-search-flag_perc-90-new-recorte-total-window-v4.csv"
        fieldnames = ["candidate", "final_nmse", "opt_nsigma"]
        rows, skipped = _run_noise_search()
    else:
        output_csv = REPORTS_DIR / "search_results_signal_final-big-search-flag_perc-90-new-recorte-total-window-v4.csv"
        fieldnames = [
            "real_mchirp",
            "real_distance",
            "candidate",
            "final_nmse",
            "opt_nsigma",
            "final_mass",
            "final_slope",
        ]
        rows, skipped = _run_signal_search()

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Total skipped candidates: {skipped}")
    print(f"CSV guardado en: {output_csv}")


if __name__ == "__main__":
    main()
