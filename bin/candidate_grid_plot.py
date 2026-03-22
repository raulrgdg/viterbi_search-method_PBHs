#!/usr/bin/env python3

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_CSV = Path("outputs/reports/search_results_signal_final-big-search-flag_perc-90-new-recorte-total-window-v4.csv")


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

    cmap = ListedColormap(["white", "#b7d8ff"])
    image = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_title("Candidate Distribution", pad=10, fontsize=14)
    ax.set_xlabel(r"Chirp mass $M_c$", fontsize=12)
    ax.set_ylabel(r"Distance $d_L$ (Mpc)", fontsize=12)

    x_tick_count = min(5, len(mchirps))
    y_tick_count = min(5, len(distances))
    x_ticks = np.linspace(0, len(mchirps) - 1, num=x_tick_count, dtype=int)
    y_ticks = np.linspace(0, len(distances) - 1, num=y_tick_count, dtype=int)
    x_ticks = np.unique(x_ticks)
    y_ticks = np.unique(y_ticks)

    ax.set_xticks(x_ticks, labels=[f"{mchirps[idx]:.0e}" for idx in x_ticks])
    ax.set_yticks(y_ticks, labels=[f"{distances[idx]:.3f}" for idx in y_ticks])

    ax.tick_params(axis="x", rotation=0, labelsize=10, direction="in", top=True)
    ax.tick_params(axis="y", labelsize=10, direction="in", right=True)
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", top=True, right=True)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    cbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03, ticks=[0, 1])
    cbar.ax.set_yticklabels(["No", "Yes"])
    cbar.set_label("Candidate", rotation=90, labelpad=10, fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    csv_path = args.csv_path.resolve()
    if not csv_path.exists():
        print(f"No existe el CSV: {csv_path}", file=sys.stderr)
        return 1

    output_path = args.output.resolve() if args.output else csv_path.with_name(f"{csv_path.stem}_candidate_grid.png")

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
