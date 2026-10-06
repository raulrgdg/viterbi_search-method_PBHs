"""Remapped time-frequency maps (t, f^-8/3) and Viterbi tracking (Secs. II A-B, III A)."""

import logging
import tempfile
import time

import numpy as np

from pbh_viterbi.config import FBAND, FMIN, NUM_FRAMES, FRAME_LENGTH, VITERBI_TRANSITION_LOG_PROBS
from pbh_viterbi.sft.load import load_sft_matrix
from pbh_viterbi.sft.make_sfts import make_sfts

log = logging.getLogger(__name__)


def n_frequency_bins(tsft, band=FBAND):
    """Number of frequency bins of an SFT of duration ``tsft`` covering ``band``."""
    return round(band / (1 / tsft))


def build_remap_geometry(tsft, fmin, nbins):
    """Frequency grid of the SFTs and the uniform f^-8/3 grid they are remapped onto."""
    freqs = fmin + np.arange(nbins) * (1 / tsft)
    if np.any(freqs <= 0):
        raise ValueError("Frequencies must be > 0 to use f^-8/3.")
    x_inc = freqs[::-1] ** (-8 / 3)  # increasing in x
    x_new = np.linspace(x_inc.min(), x_inc.max(), nbins)
    return {"freqs": freqs, "x_inc": x_inc, "x_new": x_new}


def normalized_power(sft_matrix):
    """|SFT| normalised per frequency bin by its median over time.

    Input shape (n_bins, n_sft); output shape (n_sft, n_bins).
    """
    magnitude = np.abs(sft_matrix)
    noise_floor = np.median(magnitude, axis=1) / (2 * np.log(2))
    return (magnitude / noise_floor[:, np.newaxis]).T


def remap_to_fm83(power, x_inc, x_new, fill_value=np.nan):
    """Linearly interpolate each time step from the f grid onto the uniform f^-8/3 grid."""
    power = np.asarray(power)
    power_inc = power[:, ::-1]
    remapped = np.empty((power_inc.shape[0], x_new.size), dtype=float)
    for i in range(power_inc.shape[0]):
        remapped[i] = np.interp(x_new, x_inc, power_inc[i], left=fill_value, right=fill_value)
    return remapped


def remapped_map(sft_matrix, tsft, fmin=FMIN):
    """Return (x_grid, remapped_power) for an SFT matrix of shape (n_bins, n_sft)."""
    geometry = build_remap_geometry(tsft, fmin, sft_matrix.shape[0])
    remapped = remap_to_fm83(normalized_power(sft_matrix), geometry["x_inc"], geometry["x_new"])
    return geometry["x_new"], remapped


def viterbi_track(remapped_power):
    """Most likely track (bin index per time step) through a remapped map, via soapcw."""
    import soapcw

    result = soapcw.single_detector(VITERBI_TRANSITION_LOG_PROBS, remapped_power, lookup_table=None)
    return np.asarray(result.vit_track, dtype=int)


def process_tsft(tsft, t_start, t_end, framecache, num_threads, fmin=FMIN, band=FBAND, verbose_sft=False):
    """Build the remapped map of one pack for one Tsft and track it with Viterbi.

    SFTs are written to a temporary folder and discarded afterwards. Returns a
    dict with ``tsft``, ``track_index`` (bin per time step), ``track_freq``
    (track in f^-8/3 units) and ``power`` (remapped map, shape (n_sft, n_bins)).
    """
    nbins = n_frequency_bins(tsft, band)
    n_sft = int(NUM_FRAMES * FRAME_LENGTH / tsft)
    geometry = build_remap_geometry(tsft, fmin, nbins)

    with tempfile.TemporaryDirectory(prefix=f"sft-tsft{tsft}-") as sft_dir:
        tic = time.perf_counter()
        make_sfts(t_start, t_end, tsft, framecache, sft_dir, num_threads, fmin=fmin, band=band, verbose=verbose_sft)
        log.info("MakeSFTs tsft=%s s took %.1f s", tsft, time.perf_counter() - tic)
        sft_matrix = load_sft_matrix(sft_dir, t_start, tsft, nbins, n_sft)

    remapped = remap_to_fm83(normalized_power(sft_matrix), geometry["x_inc"], geometry["x_new"])
    track_index = viterbi_track(remapped)
    return {
        "tsft": tsft,
        "track_index": track_index,
        "track_freq": np.asarray(geometry["x_new"][track_index], dtype=float),
        "power": np.asarray(remapped, dtype=float),
    }
