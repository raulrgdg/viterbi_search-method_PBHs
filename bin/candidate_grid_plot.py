#!/usr/bin/env python3

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter
import numpy as np


DEFAULT_CSV = Path("outputs/reports/search_results_signal_pack3.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a distance vs chirp-mass candidate grid from a search-results CSV.",
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=DEFAULT_CSV,
        help="Input CSV with columns real_mchirp, real_distance, candidate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PNG path. Defaults to <csv_stem>_candidate_grid.png in the same folder.",
    )
    return parser.parse_args()


def parse_candidate(value: str) -> bool:
    return value.strip().lower() == "true"


def load_grid(csv_path: Path) -> tuple[np.ndarray, list[float], list[float]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV vacio: {csv_path}")

    required_fields = {"real_mchirp", "real_distance", "candidate"}
    missing_fields = required_fields.difference(reader.fieldnames or [])
    if missing_fields:
        missing_list = ", ".join(sorted(missing_fields))
        raise ValueError(f"Faltan columnas requeridas en {csv_path}: {missing_list}")

    mchirps = sorted({float(row["real_mchirp"]) for row in rows})
    distances = sorted({float(row["real_distance"]) for row in rows})
    mchirp_index = {value: idx for idx, value in enumerate(mchirps)}
    distance_index = {value: idx for idx, value in enumerate(distances)}

    grid = np.zeros((len(distances), len(mchirps)), dtype=float)
    for row in rows:
        mchirp = float(row["real_mchirp"])
        distance = float(row["real_distance"])
        if parse_candidate(row["candidate"]):
            grid[distance_index[distance], mchirp_index[mchirp]] = 1.0

    return grid, mchirps, distances

def plot_grid(grid: np.ndarray, mchirps: list[float], distances: list[float], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
    x_min, x_max = 3e-4, 1e-1
    y_min, y_max = 8e-1, 2.5e2

    mchirps_array = np.asarray(mchirps, dtype=float)
    distances_kpc = np.asarray(distances, dtype=float) * 1e3
    max_distance_kpc = np.full(len(mchirps), np.nan, dtype=float)

    for idx in range(len(mchirps)):
        candidate_rows = np.flatnonzero(grid[:, idx] > 0)
        if candidate_rows.size:
            max_distance_kpc[idx] = distances_kpc[candidate_rows[-1]]

    valid = ~np.isnan(max_distance_kpc)
    x_values = mchirps_array[valid]
    y_values = max_distance_kpc[valid]

    ax.fill_between(x_values, y_min, y_values, color="#b7d8ff", alpha=0.45, zorder=1)
    ax.plot(x_values, y_values, color="#1f5aa6", linewidth=2.2, zorder=2)

    ax.set_title("Candidate Distribution", pad=10, fontsize=14)
    ax.set_xlabel(r"Chirp mass $M_c$", fontsize=12)
    ax.set_ylabel(r"Distance $d_L$ (kpc)", fontsize=12)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_minor_formatter(NullFormatter())

    ax.tick_params(axis="x", rotation=0, labelsize=10, direction="in", top=True)
    ax.tick_params(axis="y", labelsize=10, direction="in", right=True)
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", top=True, right=True)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    csv_path = args.csv_path.resolve()
    if not csv_path.exists():
        print(f"No existe el CSV: {csv_path}", file=sys.stderr)
        return 1

    output_path = args.output.resolve() if args.output else csv_path.with_name(f"{csv_path.stem}_grid.png")

    try:
        grid, mchirps, distances = load_grid(csv_path)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    plot_grid(grid, mchirps, distances, output_path)
    print(f"CSV: {csv_path}")
    print(f"Plot saved to: {output_path}")
    print(f"Grid shape: {grid.shape[0]} distances x {grid.shape[1]} chirp masses")
    print(f"True candidates: {int(grid.sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
