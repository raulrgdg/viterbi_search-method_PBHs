#!/usr/bin/env python3
"""Step 2: Mc = 0.085 Msun injections with 13 and 14 Tsft on the same signals.

The 30 packs where the paper injected Mc = 0.085 Msun (HPC3: 25-36, 61-66,
68-72, 97-103), each with the 30 distances of the grid: 900 injections. Sky
and polarisation from pbh_viterbi.waveform.injection.sample_sky (the paper
sky is not stored). Injection in memory (checked in step 0), band
61.1-126.8 Hz.

For every injection the maps of the 14 Tsft are built once and the search is
run twice: with the 13 paper Tsft and with Tsft = 1 s added. nsigma uses the
paper background for the 13 Tsft and the step 1 background for Tsft = 1 s.

Run as 8 parallel jobs (--n-jobs 8 --job-id k); job k writes
results/injections_14tsft.job<k>.csv (one row per injection).
"""

import argparse
import time

import numpy as np
from common import (BACKGROUND_14, PAPER_BACKGROUND, RESULTS, TSFT_13, TSFT_14, append_csv, concat, done_keys,
                    inject_in_memory, load_background, products, read_pack, track_powers)

from pbh_viterbi.config import DEFAULT_SKY_SEED, DISTANCE_GRID, MCHIRP_GRID, t_to_merger_for_mchirp
from pbh_viterbi.search.candidates import search_candidate
from pbh_viterbi.waveform.injection import build_signal_grid, sample_sky

MCHIRP = 0.085
PACKS = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 61, 62, 63, 64, 65, 66, 68, 69, 70, 71, 72,
         97, 98, 99, 100, 101, 102, 103]
RESULT_KEYS = ("candidate", "nmse", "nsigma", "mass", "tsft", "status")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--job-id", type=int, default=0)
    args = parser.parse_args()
    from pycbc.conversions import mchirp_from_mass1_mass2

    paper_mean, paper_std = load_background(PAPER_BACKGROUND)
    new_mean, new_std = load_background(BACKGROUND_14)
    mean14, std14 = {**paper_mean, 1: new_mean[1]}, {**paper_std, 1: new_std[1]}

    grid = build_signal_grid()
    i_mc = int(np.argmin(np.abs(MCHIRP_GRID - MCHIRP)))
    output = RESULTS / f"injections_14tsft.job{args.job_id}.csv"
    done = done_keys(output, ["pack", "signal_id"])

    for pack in PACKS[args.job_id::args.n_jobs]:
        raw, t_start = read_pack(pack)
        for i_d in range(len(DISTANCE_GRID)):
            signal_id = i_mc * len(DISTANCE_GRID) + i_d
            if (str(pack), str(signal_id)) in done:
                continue
            tic = time.perf_counter()
            m1, m2, distance = grid[signal_id]
            mchirp = mchirp_from_mass1_mass2(m1, m2)
            sky = sample_sky(DEFAULT_SKY_SEED, pack, signal_id)
            strain, fs = concat(inject_in_memory(raw, t_start, m1, m2, distance, t_to_merger_for_mchirp(mchirp),
                                                 sky["ra"], sky["dec"], sky["pol"]))
            prods = products(strain, fs, TSFT_14)
            res13 = search_candidate([p for p in prods if p["tsft"] in TSFT_13], paper_mean, paper_std)
            res14 = search_candidate(prods, mean14, std14)
            powers = track_powers(prods)

            row = {"pack": pack, "signal_id": signal_id, "mchirp": mchirp, "distance": distance, **sky}
            for name, res in (("13", res13), ("14", res14)):
                row.update({f"{k}_{name}": res[k] for k in RESULT_KEYS})
            row["same_result"] = all(row[f"{k}_13"] == row[f"{k}_14"] for k in RESULT_KEYS)
            row.update({f"nsigma_tsft{t}": (powers[t] - mean14[t]) / std14[t] for t in TSFT_14})
            row["time_s"] = round(time.perf_counter() - tic, 1)
            append_csv([row], output)
            print(f"pack {pack} dL={distance:.4g}: 13 -> {res13['status']} tsft={res13['tsft']} "
                  f"nsigma={res13['nsigma']:.3f} nmse={res13['nmse']} | 14 -> {res14['status']} "
                  f"tsft={res14['tsft']} nsigma={res14['nsigma']:.3f} nmse={res14['nmse']} ({row['time_s']} s)",
                  flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
