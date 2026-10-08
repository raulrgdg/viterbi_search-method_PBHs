#!/usr/bin/env python3
"""Noise search with in-memory SFT generation (fast_sft) instead of lalpulsar_MakeSFTs.

Everything after the SFTs (remapping, Viterbi, candidate isolation, output
format) is the pbh_viterbi pipeline unchanged.

    python run_noise_search_fast.py --packs 4,37,49 --fmax 127.2
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from fast_sft import sft_matrix  # noqa: E402

from pbh_viterbi.config import FMAX, FMIN, TSFT_VALUES  # noqa: E402
from pbh_viterbi.o3.frames import read_pack_frames  # noqa: E402
from pbh_viterbi.o3.packs import pack_window, parse_pack_list  # noqa: E402
from pbh_viterbi.paths import DEFAULT_NOISE_BACKGROUND, O3_DIR, o3_pack_dir  # noqa: E402
from pbh_viterbi.results import FIELDNAMES, result_row, write_rows  # noqa: E402
from pbh_viterbi.search.candidates import search_candidate  # noqa: E402
from pbh_viterbi.search.power import load_noise_background  # noqa: E402
from pbh_viterbi.sft.maps import build_remap_geometry, normalized_power, remap_to_fm83, viterbi_track  # noqa: E402


def tsft_products_fast(strain, sample_rate, tsft_values, fmin, fmax, timing):
    products = []
    for tsft in tsft_values:
        t0 = time.perf_counter()
        sfts = sft_matrix(strain, sample_rate, tsft, fmin, fmax - fmin)
        t1 = time.perf_counter()
        geometry = build_remap_geometry(tsft, fmin, sfts.shape[0])
        power = remap_to_fm83(normalized_power(sfts), geometry["x_inc"], geometry["x_new"])
        track = viterbi_track(power)
        t2 = time.perf_counter()
        timing["sft"] += t1 - t0
        timing["remap_viterbi"] += t2 - t1
        products.append({"tsft": tsft, "track_index": track, "track_freq": geometry["x_new"][track],
                         "power": np.asarray(power, dtype=float)})
    return products


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--packs", required=True)
    parser.add_argument("--fmin", type=float, default=FMIN)
    parser.add_argument("--fmax", type=float, default=FMAX)
    parser.add_argument("--tsft", type=int, nargs="+", default=list(TSFT_VALUES))
    parser.add_argument("--o3-dir", type=Path, default=O3_DIR)
    parser.add_argument("--noise-background", type=Path, default=DEFAULT_NOISE_BACKGROUND)
    parser.add_argument("--output", type=Path, default=HERE / "results" / "search_results_noise_fast.csv")
    parser.add_argument("--timing", type=Path, default=None, help="CSV with the time of each stage per pack.")
    args = parser.parse_args()

    background = load_noise_background(args.noise_background)
    viterbi_track(np.ones((20, 10)))  # import and compile soapcw before timing
    rows, timings = [], []
    for pack in parse_pack_list(args.packs):
        timing = {"pack": pack, "read": 0.0, "sft": 0.0, "remap_viterbi": 0.0, "candidate": 0.0}
        t0 = time.perf_counter()
        segments = read_pack_frames(o3_pack_dir(pack, args.o3_dir), pack_window(pack)[0])
        strain = np.concatenate([np.asarray(s, dtype=np.float64) for s in segments])
        timing["read"] = time.perf_counter() - t0

        products = tsft_products_fast(strain, 1.0 / segments[0].delta_t, args.tsft, args.fmin, args.fmax, timing)
        t0 = time.perf_counter()
        result = search_candidate(products, *background)
        timing["candidate"] = time.perf_counter() - t0
        timing["total"] = sum(v for k, v in timing.items() if k != "pack")
        rows.append(result_row(pack, result, injected=False))
        timings.append(timing)
        print(f"pack {pack}: nsigma={result['nsigma']:.6f} nmse={result['nmse']} mass={result['mass']} "
              f"tsft={result['tsft']} {result['status']} | " +
              ", ".join(f"{k} {v:.1f} s" for k, v in timing.items() if k != "pack"), flush=True)

    write_rows(rows, args.output, FIELDNAMES)
    if args.timing:
        with args.timing.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(timings[0]))
            writer.writeheader()
            writer.writerows(timings)
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
