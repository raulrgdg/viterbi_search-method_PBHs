#!/usr/bin/env python3
"""Step 1: noise track power of the 108 packs for the 14 Tsft (1 s added).

Run as 8 parallel jobs (--n-jobs 8 --job-id k); job k appends
(pack, tsft, total_power) to results/noise_track_power_14tsft.job<k>.csv. Then
`--stats` computes the mean/std per Tsft (results/noise_background_14tsft.csv)
and compares the 13 paper Tsft with paper_data/calibration/noise_power_background.csv.

Band 61.1-126.8 Hz, as in the injected search.
"""

import argparse
import csv
import statistics
from collections import defaultdict

from common import (BACKGROUND_14, PAPER_BACKGROUND, RESULTS, TSFT_14, append_csv, concat, done_keys, load_background,
                    products, read_pack, track_powers)

POWER_GLOB = "noise_track_power_14tsft.job*.csv"
PACKS = list(range(1, 109))


def run(n_jobs, job_id):
    output = RESULTS / f"noise_track_power_14tsft.job{job_id}.csv"
    done = {int(k[0]) for k in done_keys(output, ["pack"])}
    for pack in PACKS[job_id::n_jobs]:
        if pack in done:
            continue
        segments, _ = read_pack(pack)
        strain, fs = concat(segments)
        powers = track_powers(products(strain, fs, TSFT_14))
        append_csv([{"pack": pack, "tsft": t, "total_power": p} for t, p in powers.items()], output)
        print(f"pack {pack}: " + ", ".join(f"{t}:{p:.1f}" for t, p in powers.items()), flush=True)


def stats():
    values = defaultdict(list)
    for path in sorted(RESULTS.glob(POWER_GLOB)):
        with path.open() as handle:
            for row in csv.DictReader(handle):
                values[int(row["tsft"])].append(float(row["total_power"]))
    paper_mean, paper_std = load_background(PAPER_BACKGROUND)
    rows = []
    for tsft in sorted(values):
        mean, std = statistics.mean(values[tsft]), statistics.stdev(values[tsft])
        rows.append({"tsft": tsft, "mean_total_power": mean, "std_total_power": std, "n_packs": len(values[tsft]),
                     "paper_mean": paper_mean.get(tsft, ""), "paper_std": paper_std.get(tsft, ""),
                     "rel_diff_mean": (mean / paper_mean[tsft] - 1) if tsft in paper_mean else "",
                     "rel_diff_std": (std / paper_std[tsft] - 1) if tsft in paper_std else ""})
    BACKGROUND_14.unlink(missing_ok=True)
    append_csv(rows, BACKGROUND_14)
    for r in rows:
        print(r)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--job-id", type=int, default=0)
    parser.add_argument("--stats", action="store_true", help="Only compute the per-Tsft mean/std.")
    args = parser.parse_args()
    stats() if args.stats else run(args.n_jobs, args.job_id)


if __name__ == "__main__":
    main()
