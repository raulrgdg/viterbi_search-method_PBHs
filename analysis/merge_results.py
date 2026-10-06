#!/usr/bin/env python3
"""Merge all CSV files in a folder into one CSV with a single header.

Typical use: gather the per-pack injected-search results produced on the
different clusters and merge them into one campaign file, e.g.

    python analysis/merge_results.py results/search results/search/injected_campaign.csv \
        --pattern "search_results_injected_pack-*.csv" --drop-duplicates pack,mchirp,distance

``--drop-duplicates`` keeps the last row of every key, which removes the
duplicates left by jobs that were re-run after a failure.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge CSV files from a folder into one output CSV."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Folder containing the CSV files to merge.",
    )
    parser.add_argument(
        "output_csv",
        type=Path,
        help="Path for the merged output CSV.",
    )
    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="Glob pattern for input files. Default: *.csv",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for CSV files recursively below input_dir.",
    )
    parser.add_argument(
        "--allow-different-headers",
        action="store_true",
        help="Do not fail when a later CSV has a different header.",
    )
    parser.add_argument(
        "--drop-duplicates",
        default=None,
        help="Comma-separated key columns; keep only the last row of every key.",
    )
    parser.add_argument(
        "--sort-by",
        default=None,
        help="Column name used to sort the merged rows, for example: mass.",
    )
    return parser.parse_args()


def find_input_files(
    input_dir: Path, output_csv: Path, pattern: str, recursive: bool
) -> list[Path]:
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input folder does not exist: {input_dir}")

    output_csv = output_csv.resolve()
    globber = input_dir.rglob if recursive else input_dir.glob
    files = sorted(path for path in globber(pattern) if path.is_file())
    return [path for path in files if path.resolve() != output_csv]


def merge_csvs(
    input_files: list[Path],
    output_csv: Path,
    allow_different_headers: bool,
    sort_by: str | None,
    drop_duplicates: str | None = None,
) -> tuple[int, int]:
    if not input_files:
        raise FileNotFoundError("No CSV files found to merge. Check the folder and pattern.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    header: list[str] | None = None
    rows: list[list[str]] = []

    for input_file in input_files:
        with input_file.open("r", newline="", encoding="utf-8-sig") as input_handle:
            reader = csv.reader(input_handle)

            try:
                current_header = next(reader)
            except StopIteration:
                continue

            if header is None:
                header = current_header
            elif current_header != header and not allow_different_headers:
                raise ValueError(
                    "Header mismatch in "
                    f"{input_file}.\nExpected: {header}\nFound:    {current_header}"
                )

            rows.extend(reader)

    if header is None:
        raise FileNotFoundError("CSV files were empty; no header found.")

    if drop_duplicates:
        key_indices = [header.index(column.strip()) for column in drop_duplicates.split(",")]
        unique: dict[tuple[str, ...], list[str]] = {}
        for row in rows:
            unique[tuple(row[i] for i in key_indices)] = row
        removed = len(rows) - len(unique)
        rows = list(unique.values())
        print(f"Removed {removed} duplicated row(s)")

    if sort_by is not None:
        try:
            sort_index = header.index(sort_by)
        except ValueError as error:
            raise ValueError(f"Sort column not found in header: {sort_by}") from error

        def sort_key(row: list[str]) -> tuple[int, float, str]:
            if sort_index >= len(row) or row[sort_index] == "":
                return (2, 0.0, "")
            try:
                return (0, float(row[sort_index]), "")
            except ValueError:
                return (1, 0.0, row[sort_index])

        rows.sort(key=sort_key)

    with output_csv.open("w", newline="", encoding="utf-8") as output_handle:
        writer = csv.writer(output_handle)
        writer.writerow(header)
        writer.writerows(rows)

    return len(input_files), len(rows)


def main() -> None:
    args = parse_args()
    input_files = find_input_files(
        args.input_dir, args.output_csv, args.pattern, args.recursive
    )
    file_count, row_count = merge_csvs(
        input_files, args.output_csv, args.allow_different_headers, args.sort_by, args.drop_duplicates
    )
    print(f"Merged {file_count} CSV file(s) into {args.output_csv}")
    print(f"Wrote {row_count} data row(s)")


if __name__ == "__main__":
    main()
