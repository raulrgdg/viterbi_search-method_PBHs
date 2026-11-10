#!/usr/bin/env python3
"""Pairwise figures of the multi-detector study (noise packs 52 and 27, quick-test d50).

  maps_segment_vs_chunk_pack<P>.png    H1 per-segment high-pass vs chunk high-pass (map normalisation 'bin')
  maps_chunk_vs_sftnorm_pack<P>.png    H1 chunk high-pass, normalisation 'bin' vs 'bin+sft'
  d50_A_vs_B2.png, d50_B2_vs_C2.png    d50 per chirp mass from results/multi_detector/quick_test (quick_test.py --configs A B2 C2)

Each map row: (1) spectrogram (raw log10|SFT|^2 for the high-pass comparison, the normalised map
in frequency for the normalisation comparison, whose raw SFTs are identical), (2) remapped map with
the Viterbi track and the isolated block, (3) zoom on the isolated block.

    python studies/multi_detector/plot_pairs.py --out results/multi_detector/pair_plots
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pbh_viterbi.config import FMAX, FMIN  # noqa: E402
from pbh_viterbi.o3.frames import read_pack_frames  # noqa: E402
from pbh_viterbi.o3.packs import pack_window  # noqa: E402
from pbh_viterbi.search.candidates import isolate_candidate  # noqa: E402
from pbh_viterbi.sft.fast import highpass, sft_matrix  # noqa: E402
from pbh_viterbi.sft.maps import build_remap_geometry, first_bin_frequency, normalized_power, track_product  # noqa: E402
from pbh_viterbi.workflows.common import concat_segments  # noqa: E402

PACKS = {52: 14.75, 27: 11.625}  # noise packs that set the threshold of B, at the Tsft B selected
LABELS = {"A": "H1, v1.2 maps (per-segment high-pass)",
          "B": "H1, chunk high-pass, 'bin'",
          "B2": "H1, v1.3 maps (chunk high-pass, 'bin+sft')",
          "C2": "H1+L1, v1.3 maps"}
COLORS = {"A": "#888780", "B": "#2a78d6", "B2": "#0b3d91", "C2": "#a8320c"}


def row_data(matrix, tsft, normalisation):
    prod = track_product(matrix, tsft, FMIN, normalisation)
    status, _, _, expanded = isolate_candidate(prod["track_index"], prod["track_freq"], prod["power"], tsft)
    block = None
    if expanded is not None:
        block = (expanded["block_start"] * tsft, expanded["block_end"] * tsft, float(expanded.get("final_nmse", np.nan)))
    prod["normalisation"] = normalisation
    return prod, block, status or "ok"


def draw_row(ax, row, matrix, tsft, label, first_col, fallback_block=None):
    """``row`` = row_data(...); without an isolated block the zoom uses ``fallback_block`` (other row)."""
    prod, block, status = row
    normalisation = prod["normalisation"]
    freqs = first_bin_frequency(tsft, FMIN) + np.arange(matrix.shape[0]) / tsft
    t = np.arange(matrix.shape[1]) * tsft
    x_grid = build_remap_geometry(tsft, FMIN, matrix.shape[0])["x_new"]
    norm_map = normalized_power(matrix, normalisation).T  # (bins, time)

    a = ax[0]
    if first_col == "raw":
        img = np.log10(np.abs(matrix) ** 2)
        im = a.pcolormesh(t / 3600, freqs, img, shading="auto", cmap="viridis", rasterized=True,
                          vmin=np.percentile(img, 5), vmax=np.percentile(img, 99.5))
        a.set_title(f"{label}\nspectrogram log$_{{10}}$|SFT|$^2$")
    else:
        im = a.pcolormesh(t / 3600, freqs, norm_map, shading="auto", cmap="magma", rasterized=True,
                          vmin=0, vmax=np.percentile(norm_map, 99.9))
        a.set_title(f"{label}\nnormalised map in frequency (max {norm_map.max():.0f})")
    a.set_ylabel("f (Hz)")
    plt.colorbar(im, ax=a)

    a = ax[1]
    power = prod["power"]
    im = a.pcolormesh(t / 3600, x_grid, power.T, shading="auto", cmap="magma", rasterized=True,
                      vmin=0, vmax=np.nanpercentile(power, 99.9))
    a.plot(t / 3600, prod["track_freq"], color="cyan", lw=0.7, label="Viterbi track")
    if block:
        a.axvspan(block[0] / 3600, block[1] / 3600, color="lime", alpha=0.2,
                  label=f"isolated block, NMSE = {block[2]:.1e}")
    else:
        a.text(0.02, 0.95, f"isolation: {status}", transform=a.transAxes, color="w")
    a.set_title("remapped map (t, f$^{-8/3}$) + Viterbi track")
    a.set_ylabel("f$^{-8/3}$ (Hz$^{-8/3}$)")
    a.legend(loc="upper right", fontsize=8)
    plt.colorbar(im, ax=a)

    a = ax[2]
    ftrack = prod["track_freq"] ** (-3 / 8)
    zoom = block or fallback_block
    if zoom:
        b0, b1 = zoom[0], zoom[1]
        sel = (t >= b0 - 0.3 * (b1 - b0)) & (t <= b1 + 0.3 * (b1 - b0))
        inb = (t >= b0) & (t <= b1)
        fsel = (freqs >= ftrack[inb].min() - 3) & (freqs <= ftrack[inb].max() + 3)
        im = a.pcolormesh(t[sel] / 3600, freqs[fsel], norm_map[np.ix_(fsel, sel)], shading="auto", cmap="magma",
                          vmin=0, vmax=8, rasterized=True)
        a.plot(t[sel] / 3600, ftrack[sel], color="cyan", lw=0.9)
        a.axvline(b0 / 3600, color="lime", ls="--")
        a.axvline(b1 / 3600, color="lime", ls="--")
        a.set_ylim(freqs[fsel][0], freqs[fsel][-1])
        if block:
            a.set_title(f"selected track (zoom on the block)\n{ftrack[inb].min():.1f}–{ftrack[inb].max():.1f} Hz, "
                        f"NMSE = {block[2]:.1e}")
        else:
            a.set_title(f"no isolated block ({status}): track in the\nblock window of the other row")
        plt.colorbar(im, ax=a)
    else:
        a.text(0.5, 0.5, f"no isolated block ({status})", ha="center", transform=a.transAxes)
    a.set_ylabel("f (Hz)")


def map_figures(out):
    for pack, tsft in PACKS.items():
        t_start, _ = pack_window(pack)
        strain, fs = concat_segments(read_pack_frames(ROOT / f"data/o3/O3b-pack{pack}", t_start))
        seg = sft_matrix(strain, fs, tsft, FMIN, FMAX - FMIN)
        chunk = sft_matrix(highpass(strain, fs, FMIN), fs, tsft, FMIN, FMAX - FMIN, prefiltered=True)
        del strain
        for name, rows, first_col in (
                ("segment_vs_chunk", [(seg, "bin", LABELS["A"]), (chunk, "bin", LABELS["B"])], "raw"),
                ("chunk_vs_sftnorm", [(chunk, "bin", LABELS["B"]), (chunk, "bin+sft", LABELS["B2"])], "norm")):
            fig, ax = plt.subplots(2, 3, figsize=(19, 9.5))
            data = [row_data(matrix, tsft, normalisation) for matrix, normalisation, _ in rows]
            for r, ((matrix, _, label), row) in enumerate(zip(rows, data)):
                other = next((d[1] for d in data if d[1] is not None), None)
                draw_row(ax[r], row, matrix, tsft, label, first_col, fallback_block=other)
            for a in ax[-1]:
                a.set_xlabel("time from the start of the pack (h)")
            fig.suptitle(f"Noise, pack {pack}, H1, Tsft = {tsft} s", fontsize=15)
            fig.tight_layout()
            fig.savefig(out / f"maps_{name}_pack{pack}.png", dpi=110)
            plt.close(fig)


def d50_figures(out, quick_dir):
    pm = pd.read_csv(quick_dir / "per_mass.csv")
    signals = pd.read_csv(Path(__file__).resolve().parent / "quick_signals.csv")
    for pair in (("A", "B2"), ("B2", "C2")):
        fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
        g0 = None
        for c in pair:
            g = pm[(pm["k"] == 1) & (pm["config"] == c)].sort_values("mchirp").reset_index(drop=True)
            dmax = np.array([signals.loc[np.isclose(signals["mchirp"], m), "distance"].max() * 1e3 for m in g["mchirp"]])
            lower = g["d50_kpc"].isna().values
            y = np.where(lower, dmax, g["d50_kpc"].values)
            ax.plot(g["mchirp"], y, "-o", color=COLORS[c], lw=2, label=LABELS[c])
            for m, d in zip(g["mchirp"][lower], dmax[lower]):
                ax.plot([m], [d], "^", color=COLORS[c], ms=11)
                ax.annotate(f"> {d:.0f}", (m, d), textcoords="offset points", xytext=(7, 4), fontsize=9, color=COLORS[c])
            if g0 is None:
                g0 = (y, lower)
            else:
                for m, y0, lo0, y1, lo1 in zip(g["mchirp"], g0[0], g0[1], y, lower):
                    if lo0:  # reference is a lower limit: no ratio
                        continue
                    ax.annotate(f"{'> ' if lo1 else ''}×{y1 / y0:.1f}", (m, np.sqrt(y0 * y1)),
                                textcoords="offset points", xytext=(6, -3), fontsize=9, color="k")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.set_xlabel(r"chirp mass $\mathcal{M}$ (M$_\odot$)")
        ax.set_ylabel(r"$d_{50}$ (kpc)")
        ax.set_title("Distance of 50% detection (FAR ~3%, 10 evaluation packs)\n"
                     "triangle = lower limit; number = d$_{50}$ ratio", fontsize=10)
        ax.legend(fontsize=9, loc="lower right")
        fig.savefig(out / f"d50_{pair[0]}_vs_{pair[1]}.png", dpi=140)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "results/multi_detector/pair_plots")
    parser.add_argument("--quick-test", type=Path, default=ROOT / "results/multi_detector/quick_test")
    parser.add_argument("--skip-maps", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    d50_figures(args.out, args.quick_test)
    if not args.skip_maps:
        map_figures(args.out)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
