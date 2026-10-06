#!/usr/bin/env python3
"""Fig. 4: stages of the search applied to an injected long inspiral.

Top left: remapped (t, f^-8/3) map. Top right: Viterbi track. Bottom left:
candidate isolation, with the selected window (green) expanded with smaller
windows (grey) up to the final bounds (dashed). Bottom right: isolated
candidate.

Signal: Mc = 1e-2 Msun, dL = 80 kpc, pack 10, Tsft = 9 s. The isolation is
the one of the production search (pbh_viterbi.search.candidates).

    python figures/fig4_pipeline_stages.py
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from _common import add_common_arguments, hours_since_start, injected_sfts, normalized_log_power, save, style_axis

from pbh_viterbi.paths import PLOTS_DIR
from pbh_viterbi.search.candidates import isolate_candidate
from pbh_viterbi.sft.maps import remapped_map, viterbi_track

PACK, TSFT = 10, 9
SIGNAL = {"mchirp": 1e-2, "distance": 0.080, "ra": 1.0, "dec": 0.5, "pol": 0.2}
FONT_SIZE = 16
YLABEL = r"$f^{-8/3}$ (Hz$^{-8/3}$)"


def limits(ax, hours, x_grid, close_right=True):
    t_step = TSFT / 3600.0
    ax.set_xlim(float(hours[0]), float(hours[-1] + (t_step if close_right else 0.0)))
    ax.set_ylim(float(x_grid[0]), float(x_grid[-1] + np.median(np.diff(x_grid))))
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel(YLABEL)
    style_axis(ax)


def draw_region(ax, start, end, color, alpha, linewidth):
    left, right = start * TSFT / 3600.0, end * TSFT / 3600.0
    ax.axvspan(left, right, color=color, alpha=alpha, linewidth=0)
    for edge in (left, right):
        ax.axvline(edge, color=color, linewidth=linewidth, linestyle="--", alpha=0.95)


def main():
    args = add_common_arguments(argparse.ArgumentParser(description=__doc__)).parse_args()
    plt.rcParams.update({"font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE, "xtick.labelsize": FONT_SIZE,
                         "ytick.labelsize": FONT_SIZE, "axes.linewidth": 0.9, "savefig.dpi": 400})

    sfts = injected_sfts("fig4", PACK, TSFT, **SIGNAL, threads=args.threads, regenerate=args.regenerate,
                         o3_dir=args.o3_dir)
    hours = hours_since_start(sfts)
    x_grid, power = remapped_map(sfts.sft.T, TSFT)
    track_index = viterbi_track(power)
    track = x_grid[track_index]

    expansion_windows = []
    status, _windows, optimal, expanded = isolate_candidate(track_index, track, power, TSFT,
                                                            accepted_windows=expansion_windows)
    if status is not None:
        raise SystemExit(f"Candidate isolation failed: {status}")
    initial = (int(np.min(optimal["starts"])), int(np.max(optimal["ends"])))
    final_start = min(s for s, _ in [initial, *expansion_windows])
    final_end = max(e for _, e in [initial, *expansion_windows])

    fig, ((ax_map, ax_track), (ax_iso, ax_final)) = plt.subplots(2, 2, figsize=(14.0, 8.8), constrained_layout=True)

    image = ax_map.imshow(normalized_log_power(power.T), origin="lower", aspect="auto", interpolation="none",
                          cmap="viridis",
                          extent=[float(hours[0]), float(hours[-1] + TSFT / 3600.0), float(x_grid[0]),
                                  float(x_grid[-1] + np.median(np.diff(x_grid)))])
    ax_map.set_xlabel("Time (hours)")
    ax_map.set_ylabel(YLABEL)
    style_axis(ax_map)
    cbar = fig.colorbar(image, ax=ax_map, pad=0.02)
    cbar.set_label("Normalized log SFT power", fontsize=15)
    cbar.ax.tick_params(direction="in")

    ax_track.plot(hours, track, color="black", linewidth=1.5, label="Viterbi track")
    limits(ax_track, hours, x_grid)
    ax_track.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#b0b0b0", framealpha=1.0)

    ax_iso.plot(hours, track, color="black", linewidth=1.5, solid_capstyle="round", zorder=4)
    draw_region(ax_iso, *initial, color="#1b9e77", alpha=0.14, linewidth=1.2)
    for start, end in expansion_windows:
        draw_region(ax_iso, start, end, color="#c7c7c7", alpha=0.28, linewidth=0.9)
    for edge in (final_start, final_end):
        ax_iso.axvline(edge * TSFT / 3600.0, color="black", linewidth=1.7, linestyle="--", zorder=5)
    limits(ax_iso, hours, x_grid)

    isolated = slice(final_start, final_end)
    ax_final.plot(hours[isolated], track[isolated], color="black", linewidth=1.5, solid_capstyle="round", zorder=4)
    limits(ax_final, hours, x_grid, close_right=False)

    save(fig, PLOTS_DIR / "fig4_pipeline_stages.png", bbox_inches="tight")
    print(f"Candidate: Mc = {expanded['final_mass']:.4g} Msun, NMSE = {expanded['final_nmse']:.4g}")


if __name__ == "__main__":
    main()
