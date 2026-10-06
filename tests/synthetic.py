"""Deterministic synthetic Viterbi products used by the regression tests."""

import numpy as np

from pbh_viterbi.search.fitting import slope_of_mass
from pbh_viterbi.sft.maps import build_remap_geometry, n_frequency_bins

PACK_SECONDS = 32768


def synthetic_products(mchirp=1e-2, tsft=63, start_frac=0.3, duration_frac=0.4, amplitude=3.0, seed=0):
    """Noise-like remapped map with a straight inspiral track (no Viterbi needed).

    The track follows the noise maximum outside the signal and the injected
    line inside it, mimicking what Viterbi returns for a loud signal.
    """
    rng = np.random.default_rng(seed)
    nbins = n_frequency_bins(tsft)
    n_time = PACK_SECONDS // tsft
    x_grid = build_remap_geometry(tsft, 61.1, nbins)["x_new"]
    power = 1.38 * rng.chisquare(2, size=(n_time, nbins))  # mean ~2.8, as in real remapped maps

    track = rng.integers(0, nbins, n_time).astype(float)
    track = np.round(np.convolve(track, np.ones(25) / 25, mode="same"))
    t0, t1 = int(start_frac * n_time), int((start_frac + duration_frac) * n_time)
    dx = x_grid[1] - x_grid[0]
    x0 = x_grid[int(0.9 * nbins)]
    slope_bins = slope_of_mass(mchirp) * tsft / dx
    for t in range(t0, t1):
        idx = int(np.clip(round((x0 - x_grid[0]) / dx + slope_bins * (t - t0)), 0, nbins - 1))
        track[t] = idx
        power[t, idx] += amplitude
    track = track.astype(int)
    return {"tsft": tsft, "track_index": track, "track_freq": x_grid[track], "power": power}
