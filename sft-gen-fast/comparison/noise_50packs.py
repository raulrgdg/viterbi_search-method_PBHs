#!/usr/bin/env python3
"""Test 1: noise search with fast_sft on 50 random packs vs the paper results.

The reference is paper_data/search_results/noise_search_O3b_H1_108packs.csv,
produced with lalpulsar_MakeSFTs (band 61.1-127.2 Hz). Each pack is appended to
results/noise_50packs.csv as soon as it finishes, so the run can be resumed.

    python noise_50packs.py
"""

import argparse
import csv
import time

import numpy as np
from common import O3_DIR, RESULTS, append_csv, available_packs, random_packs, same_value, sft_matrix, \
    products_from_sfts

from pbh_viterbi.config import FMIN, TSFT_VALUES
from pbh_viterbi.o3.frames import read_pack_frames
from pbh_viterbi.o3.packs import pack_window
from pbh_viterbi.paths import DEFAULT_NOISE_BACKGROUND, PAPER_DATA_DIR, o3_pack_dir
from pbh_viterbi.results import result_row
from pbh_viterbi.search.candidates import search_candidate
from pbh_viterbi.search.power import load_noise_background

REFERENCE = PAPER_DATA_DIR / "search_results" / "noise_search_O3b_H1_108packs.csv"
COMPARED = ("candidate", "nmse", "nsigma", "mass", "status")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-packs", type=int, default=50)
    parser.add_argument("--fmax", type=float, default=127.2, help="Band of the paper noise search.")
    parser.add_argument("--output", default=str(RESULTS / "noise_50packs.csv"))
    args = parser.parse_args()

    with REFERENCE.open() as handle:
        reference = {int(r["pack"]): r for r in csv.DictReader(handle)}
    packs = random_packs(args.n_packs, set(available_packs()) & set(reference), stream=1)
    print(f"Packs ({len(packs)}): {packs}", flush=True)

    done = set()
    try:
        with open(args.output) as handle:
            done = {int(r["pack"]) for r in csv.DictReader(handle)}
    except FileNotFoundError:
        pass

    background = load_noise_background(DEFAULT_NOISE_BACKGROUND)
    for pack in packs:
        if pack in done:
            continue
        t0 = time.perf_counter()
        strain_segments = read_pack_frames(o3_pack_dir(pack, O3_DIR), pack_window(pack)[0])
        strain = np.concatenate([np.asarray(s, dtype=np.float64) for s in strain_segments])
        fs = 1.0 / strain_segments[0].delta_t
        products = [products_from_sfts(sft_matrix(strain, fs, tsft, FMIN, args.fmax - FMIN), tsft, FMIN)
                    for tsft in TSFT_VALUES]
        fast = result_row(pack, search_candidate(products, *background), injected=False)
        ref = reference[pack]
        row = {"pack": pack}
        for key in COMPARED:
            row[f"{key}_makesfts"] = ref[key]
            row[f"{key}_fast"] = fast[key]
        row["tsft_fast"] = fast["tsft"]
        row["identical"] = all(same_value(ref[k], fast[k]) for k in COMPARED)
        row["time_fast_s"] = round(time.perf_counter() - t0, 1)
        append_csv([row], args.output)
        print(", ".join(f"{k}={v}" for k, v in row.items()), flush=True)

    with open(args.output) as handle:
        rows = list(csv.DictReader(handle))
    n_same = sum(r["identical"] == "True" for r in rows)
    print(f"DONE: {n_same}/{len(rows)} packs identical to the paper noise search", flush=True)


if __name__ == "__main__":
    main()
