#!/usr/bin/env python3
"""Mean and standard deviation of the noise Viterbi track power per Tsft.

These numbers normalise the nsigma statistic, Eq. (17). The input is the
output of the background mode of the noise search:

    python -m pbh_viterbi.workflows.noise_search --mode background ...
    python analysis/noise_background_stats.py results/search/noise_track_power.csv

The default output is the file the searches read by default.
"""

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import _setup  # noqa: F401  (adds src/ to sys.path)
from pbh_viterbi.paths import DEFAULT_NOISE_BACKGROUND


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_csv", type=Path, help="CSV with columns pack, tsft, total_power.")
    parser.add_argument("--output", type=Path, default=DEFAULT_NOISE_BACKGROUND)
    args = parser.parse_args()

    values = defaultdict(list)
    with args.input_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            values[row["tsft"]].append(float(row["total_power"]))

    rows = [
        {
            "tsft": tsft,
            "mean_total_power": statistics.mean(powers),
            "std_total_power": statistics.stdev(powers) if len(powers) > 1 else math.nan,
        }
        for tsft, powers in sorted(values.items(), key=lambda item: float(item[0]))
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["tsft", "mean_total_power", "std_total_power"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Background statistics for {len(rows)} Tsft values written to {args.output}")


if __name__ == "__main__":
    main()
