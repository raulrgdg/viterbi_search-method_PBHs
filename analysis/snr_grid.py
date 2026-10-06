#!/usr/bin/env python3
"""Optimal SNR of every signal of the injected population (Appendix A).

For each (chirp mass, distance) of the population, the waveform of the
32768 s chunk is projected onto H1 with a random sky location and the optimal
SNR, Eqs. (A3)-(A5), is computed in the search band with the average O3b PSD.

    python analysis/snr_grid.py
    python analysis/snr_grid.py --psd paper_data/average_noise_psd.csv --output results/search/snr_grid.csv
"""

import argparse
import csv
import logging
from pathlib import Path

import numpy as np

import _setup  # noqa: F401  (adds src/ to sys.path)
from pbh_viterbi.config import DISTANCE_GRID, MCHIRP_GRID, RA_RANGE, DEC_RANGE, POL_RANGE
from pbh_viterbi.paths import PAPER_DATA_DIR, SEARCH_RESULTS_DIR
from pbh_viterbi.snr import frame_polarizations, load_psd_csv, optimal_snr
from pbh_viterbi.waveform.injection import equal_component_masses

log = logging.getLogger("snr_grid")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--psd", type=Path, default=PAPER_DATA_DIR / "average_noise_psd.csv")
    parser.add_argument("--seed", type=int, default=1234, help="Seed of the sky draws.")
    parser.add_argument("--output", type=Path, default=SEARCH_RESULTS_DIR / "snr_grid.csv")
    args = parser.parse_args()

    psd_freqs, psd_values = load_psd_csv(args.psd)
    rng = np.random.default_rng(args.seed)
    rows = []
    for mchirp in MCHIRP_GRID:
        m1, m2 = equal_component_masses(mchirp)
        for distance in DISTANCE_GRID:
            ra, dec, pol = rng.uniform(*RA_RANGE), rng.uniform(*DEC_RANGE), rng.uniform(*POL_RANGE)
            # The SNR does not depend on the absolute start time; use t_start = 0.
            snr = optimal_snr(frame_polarizations(m1, m2, distance, 0), ra, dec, pol, psd_freqs, psd_values)
            rows.append({"mchirp": float(mchirp), "distance": float(distance), "ra": ra, "dec": dec, "pol": pol,
                         "optimal_snr": snr})
            log.info("mchirp=%.4g Msun, dL=%.4g Mpc: SNR=%.4g", mchirp, distance, snr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    log.info("SNR grid written to %s", args.output)


if __name__ == "__main__":
    main()
