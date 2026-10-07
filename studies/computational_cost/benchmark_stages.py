#!/usr/bin/env python3
"""Wall-clock time of the stages that follow SFT generation, for one 32768 s chunk.

For every Tsft, a noise-like SFT matrix of the real size is normalised and
remapped to (t, f^-8/3), tracked with Viterbi, and the candidate search is
run on it. No LIGO data are needed. See docs/computational_cost.md.

    python studies/computational_cost/benchmark_stages.py
"""

import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pbh_viterbi.config import FMIN, PACK_DURATION, TSFT_VALUES  # noqa: E402
from pbh_viterbi.paths import DEFAULT_NOISE_BACKGROUND  # noqa: E402
from pbh_viterbi.search.candidates import search_candidate  # noqa: E402
from pbh_viterbi.search.power import load_noise_background  # noqa: E402
from pbh_viterbi.sft.maps import (  # noqa: E402
    build_remap_geometry,
    n_frequency_bins,
    normalized_power,
    remap_to_fm83,
    viterbi_track,
)


def main():
    rng = np.random.default_rng(0)
    means, stds = load_noise_background(DEFAULT_NOISE_BACKGROUND)
    viterbi_track(rng.chisquare(2, (50, 20)))  # import and compile soapcw before timing

    print(" Tsft  n_SFT  n_bins   pixels  remap[s]  viterbi[s]  candidate[s]")
    totals = np.zeros(3)
    for tsft in TSFT_VALUES:
        nbins, nsft = n_frequency_bins(tsft), PACK_DURATION // tsft
        geometry = build_remap_geometry(tsft, FMIN, nbins)
        sft = rng.normal(size=(nbins, nsft)) + 1j * rng.normal(size=(nbins, nsft))

        t0 = time.perf_counter()
        power = remap_to_fm83(normalized_power(sft), geometry["x_inc"], geometry["x_new"])
        t1 = time.perf_counter()
        track = viterbi_track(power)
        t2 = time.perf_counter()
        product = {"tsft": tsft, "track_index": track, "track_freq": geometry["x_new"][track], "power": power}
        search_candidate([product], means, stds)  # candidate search if this Tsft is selected
        t3 = time.perf_counter()

        times = np.array([t1 - t0, t2 - t1, t3 - t2])
        totals += times
        print(f"{tsft:5d} {nsft:6d} {nbins:7d} {nsft * nbins:8d} {times[0]:9.2f} {times[1]:11.2f} {times[2]:13.2f}")

    print(f"total remap {totals[0]:.1f} s, Viterbi {totals[1]:.1f} s (all Tsft); "
          f"candidate search {totals[2] / len(TSFT_VALUES):.2f} s on average for the selected Tsft")
    print(f"peak memory of this process: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")


if __name__ == "__main__":
    main()
