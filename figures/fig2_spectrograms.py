#!/usr/bin/env python3
"""Fig. 2: spectrogram of O3b H1 data with a loud long inspiral injected.

Top panel: standard spectrogram, where the inspiral is a chirp.
Bottom panel: remapped (t, f^-8/3) map, where it becomes a straight line.

Signal: Mc = 1e-2 Msun, dL = 1 kpc, pack 8, Tsft = 8 s.

    python figures/fig2_spectrograms.py            # first run generates the SFTs
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from _common import add_common_arguments, hours_since_start, injected_sfts, normalized_log_power, save, style_axis

from pbh_viterbi.paths import PLOTS_DIR
from pbh_viterbi.sft.maps import remapped_map

PACK, TSFT = 8, 8
SIGNAL = {"mchirp": 1e-2, "distance": 0.001, "ra": 1.0, "dec": 0.5, "pol": 0.2, "t_to_merger": 32780}
FONT_SIZE = 19


def main():
    args = add_common_arguments(argparse.ArgumentParser(description=__doc__)).parse_args()
    plt.rcParams.update({"font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE,
                         "xtick.labelsize": FONT_SIZE, "ytick.labelsize": FONT_SIZE})

    sfts = injected_sfts("fig2", PACK, TSFT, **SIGNAL, threads=args.threads, regenerate=args.regenerate,
                         o3_dir=args.o3_dir)
    hours = hours_since_start(sfts)
    t_edge = float(hours[-1] + TSFT / 3600.0)

    # Top: |SFT|^2 in (t, f).
    fig, ax = plt.subplots(figsize=(8.6, 5.4), constrained_layout=True)
    freqs = sfts.frequencies
    image = ax.imshow(normalized_log_power(np.abs(sfts.sft.T) ** 2), origin="lower", aspect="auto",
                      interpolation="none", cmap="viridis",
                      extent=[float(hours[0]), t_edge, float(freqs[0]), float(freqs[-1] + 1.0 / TSFT)])
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("f (Hz)")
    style_axis(ax)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Normalized log SFT power")
    cbar.ax.tick_params(direction="in")
    save(fig, PLOTS_DIR / "fig2_top_spectrogram.png", dpi=400)

    # Bottom: remapped (t, f^-8/3) map.
    x_grid, remapped = remapped_map(sfts.sft.T, TSFT)
    fig, ax = plt.subplots(figsize=(8.6, 5.6), constrained_layout=True)
    image = ax.imshow(normalized_log_power(remapped.T), origin="lower", aspect="auto", interpolation="none",
                      cmap="viridis",
                      extent=[float(hours[0]), t_edge, float(x_grid[0]), float(x_grid[-1] + np.median(np.diff(x_grid)))])
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel(r"$f^{-8/3}$ (Hz$^{-8/3}$)")
    style_axis(ax)
    cbar = fig.colorbar(image, ax=ax, pad=0.02)
    cbar.set_label("Normalized log SFT power")
    cbar.ax.tick_params(direction="in")
    save(fig, PLOTS_DIR / "fig2_bottom_remapped.png", dpi=400)


if __name__ == "__main__":
    main()
