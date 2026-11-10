#!/usr/bin/env python3
"""Mean and standard deviation of the noise Viterbi track power per Tsft.

These numbers normalise the nsigma statistic, Eq. (17). They come from the
per-pack table calibration/noise_track_power<suffix>.csv (columns pack, tsft,
total_power), produced by the background mode of the noise search:

    python -m pbh_viterbi.workflows.noise_search --mode background --packs all --tsft <Tsft...>

For every Tsft list whose values are all in the table, this script writes
calibration/noise_background_tsft<list><suffix>.csv, the file the searches read by
default for that list. The suffix names the map configuration and is set by
--detectors, --sft-highpass and --map-normalisation, which must be those of the
background run (default: the default maps of config.py; the v1.2 maps,
--sft-highpass segment --map-normalisation bin, have no suffix).

    # recompute the per-list files from the stored table
    python analysis/noise_background_stats.py
    # add new noise packs: run the background mode on them, then merge and update
    python analysis/noise_background_stats.py results/search/noise_track_power.csv --add

With --add the new rows are merged into the table (a pack already present is
replaced) and the table is saved. Without --add, input CSVs are used together
with the table only for this computation.
"""

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import _setup  # noqa: F401  (adds src/ to sys.path)
from pbh_viterbi.config import (DETECTORS, MAP_NORMALISATION, MAP_NORMALISATION_MODES, SFT_HIGHPASS,
                                SFT_HIGHPASS_MODES, SUPPORTED_DETECTORS, TSFT_LISTS)
from pbh_viterbi.paths import noise_background_path, noise_track_power_path
from pbh_viterbi.search.power import tsft_key

FIELDS = ["pack", "tsft", "total_power"]


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [(int(r["pack"]), tsft_key(r["tsft"]), float(r["total_power"])) for r in csv.DictReader(handle)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_csv", type=Path, nargs="*", help="New CSV(s) with columns pack, tsft, total_power.")
    parser.add_argument("--add", action="store_true", help="Merge the inputs into the stored table and save it.")
    parser.add_argument("--detectors", nargs="+", choices=SUPPORTED_DETECTORS, default=list(DETECTORS))
    parser.add_argument("--sft-highpass", choices=SFT_HIGHPASS_MODES, default=SFT_HIGHPASS)
    parser.add_argument("--map-normalisation", choices=MAP_NORMALISATION_MODES, default=MAP_NORMALISATION)
    parser.add_argument("--table", type=Path, default=None,
                        help="Per-pack table (default: calibration/noise_track_power<suffix>.csv).")
    parser.add_argument("--tsft-list", choices=sorted(TSFT_LISTS) + ["all"], default="all",
                        help="List(s) to write (default: every list fully covered by the table).")
    parser.add_argument("--output", type=Path, default=None, help="Output file (only with a single --tsft-list).")
    args = parser.parse_args()
    config = (args.detectors, args.sft_highpass, args.map_normalisation)
    if args.table is None:
        args.table = noise_track_power_path(*config)
    print(f"Map configuration {'+'.join(args.detectors)}, {args.sft_highpass} high-pass, "
          f"{args.map_normalisation} normalisation; table {args.table}")

    power = {}  # (pack, tsft) -> total power; later rows replace earlier ones
    for path in ([args.table] if args.table.exists() else []) + list(args.input_csv):
        for pack, tsft, value in read_rows(path):
            power[(pack, tsft)] = value
    if not power:
        raise SystemExit(f"No noise track power found (table {args.table} missing and no input given).")

    if args.add:
        args.table.parent.mkdir(parents=True, exist_ok=True)
        with args.table.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(FIELDS)
            writer.writerows([pack, f"{tsft:g}", value] for (pack, tsft), value in sorted(power.items()))
        print(f"Table {args.table}: {len({p for p, _ in power})} packs, {len({t for _, t in power})} Tsft")

    values = defaultdict(list)
    for (_, tsft), value in power.items():
        values[tsft].append(value)

    lists = sorted(TSFT_LISTS) if args.tsft_list == "all" else [args.tsft_list]
    for name in lists:
        wanted = [tsft_key(t) for t in TSFT_LISTS[name]]
        missing = [t for t in wanted if t not in values]
        if missing:
            if args.tsft_list != "all":
                raise SystemExit(f"No noise track power for Tsft {missing} of list {name}.")
            print(f"List {name}: skipped, {len(missing)} Tsft missing")
            continue
        rows = [{"tsft": t, "mean_total_power": statistics.mean(values[t]),
                 "std_total_power": statistics.stdev(values[t]) if len(values[t]) > 1 else math.nan,
                 "n_packs": len(values[t])} for t in wanted]
        output = args.output if args.output and args.tsft_list != "all" else noise_background_path(name, *config)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"List {name}: {len(rows)} Tsft, {sorted({r['n_packs'] for r in rows})} packs per Tsft -> {output}")


if __name__ == "__main__":
    main()
