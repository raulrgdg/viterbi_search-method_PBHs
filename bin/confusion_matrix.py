#!/usr/bin/env python3

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_candidate(value: str) -> bool:
    return value.strip().lower() == "true"


def plot_confusion_matrix(matrix: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.8), constrained_layout=True)

    im = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks([0, 1], labels=["Predicted Signal", "Predicted Noise"])
    ax.set_yticks([0, 1], labels=["True Signal", "True Noise"])
    ax.set_title("Confusion Matrix", pad=12, fontsize=14)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = "white" if value > matrix.max() * 0.5 else "black"
            ax.text(j, i, f"{value}", ha="center", va="center", color=color, fontsize=12, fontweight="semibold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Count", rotation=90, labelpad=10)

    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/reports/metrics.csv")
    output_path = csv_path.with_name("confusion_matrix.png")

    tp = fp = tn = fn = 0

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            truth = int(row["label"]) == 1
            pred = parse_candidate(row["candidate"])

            if truth and pred:
                tp += 1
            elif not truth and pred:
                fp += 1
            elif not truth and not pred:
                tn += 1
            else:
                fn += 1

    matrix = np.array([[tp, fn], [fp, tn]])
    plot_confusion_matrix(matrix, output_path)

    print(f"File: {csv_path}")
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"TN: {tn}")
    print(f"FN: {fn}")
    print()
    print("Confusion matrix:")
    print("                    Pred Signal  Pred Noise")
    print(f"True Signal         {tp:>11}  {fn:>10}")
    print(f"True Noise          {fp:>11}  {tn:>10}")
    print()
    print(f"Plot saved to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
