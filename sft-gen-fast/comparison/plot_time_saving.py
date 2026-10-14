#!/usr/bin/env python3
"""Computational time per 32768 s chunk: pipeline with MakeSFTs vs with fast_sft.

Whole noise search per chunk (13 Tsft), 5 chunks each. MakeSFTs (12
processes, chunks of tests 2 and 3): sum of its 13 SFT times (it also reads
the frames) plus the remapping, Viterbi and candidate time, which is the same
code in both pipelines; fast_sft: measured end to end, run alone
(results/timing_noise_standalone.csv).

    python plot_time_saving.py   # -> results/time_saving.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"


def main():
    sft = pd.concat([pd.read_csv(RESULTS / "maps_2packs.csv"), pd.read_csv(RESULTS / "injections_maps.csv")])
    fast = pd.read_csv(RESULTS / "timing_noise_standalone.csv")
    search_after_sft = (fast.remap_viterbi + fast.candidate).median()
    total_makesfts = sft.groupby("pack").time_makesfts_s.sum().values + search_after_sft
    total_fast = fast.total.values

    fig, ax = plt.subplots(figsize=(4, 4))
    medians = [np.median(total_makesfts), np.median(total_fast)]
    ax.bar([0, 1], medians, 0.6, color=["C1", "C0"])
    for i, values in enumerate((total_makesfts, total_fast)):
        ax.scatter(np.full(len(values), i), values, color="k", s=12, zorder=3)
    for i, m in enumerate(medians):
        ax.text(i, m * 1.25, f"{m / 60:.0f} min" if m > 120 else f"{m:.0f} s", ha="center")
    ax.text(0.97, 0.97, r"1 chunk: $T_{\rm obs} = 32768$ s ($\approx 9.1$ h)", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, bbox={"facecolor": "white", "edgecolor": "0.7", "boxstyle": "round"})
    ax.set_xticks([0, 1], ["MakeSFTs", "fast_sft"])
    ax.set_yscale("log")
    ax.set_ylim(10, 3e4)
    ax.set_ylabel("Whole search per chunk [s]")
    ax.set_title(f"{medians[0] / medians[1]:.0f}× faster", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS / "time_saving.png", dpi=300)
    print(f"Whole search per chunk: MakeSFTs {medians[0]:.0f} s, fast_sft {medians[1]:.0f} s "
          f"({medians[0] / medians[1]:.0f}x)")


if __name__ == "__main__":
    main()
