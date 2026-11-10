"""Remapped time-frequency maps (t, f^-8/3) and Viterbi tracking (Secs. II A-B, III A).

Two ways of building the SFTs of a chunk give the same maps:

  * ``fast`` (default): in memory from the strain array (pbh_viterbi.sft.fast);
  * ``makesfts``: lalpulsar_MakeSFTs on frame files (the method of the paper;
    integer Tsft only).
"""

import logging
import tempfile
import time

import numpy as np

from pbh_viterbi.config import (FBAND, FMIN, MAP_NORMALISATION, MAP_NORMALISATION_MODES, PACK_DURATION,
                                VITERBI_TRANSITION_LOG_PROBS)

log = logging.getLogger(__name__)


def n_frequency_bins(tsft, band=FBAND):
    """Number of frequency bins of an SFT of duration ``tsft`` covering ``band``."""
    return int(round(band * tsft))


def first_bin_frequency(tsft, fmin=FMIN):
    """Frequency of the first SFT bin, round(fmin * tsft) / tsft (as in MakeSFTs)."""
    return round(fmin * tsft) / tsft


def build_remap_geometry(tsft, fmin, nbins):
    """Frequency grid of the SFTs and the uniform f^-8/3 grid they are remapped onto.

    The grid starts at the frequency of the first SFT bin, not at the nominal
    ``fmin`` (they differ by up to half a bin; v1.0 used the nominal value).
    """
    freqs = first_bin_frequency(tsft, fmin) + np.arange(nbins) / tsft
    if np.any(freqs <= 0):
        raise ValueError("Frequencies must be > 0 to use f^-8/3.")
    x_inc = freqs[::-1] ** (-8 / 3)  # increasing in x
    x_new = np.linspace(x_inc.min(), x_inc.max(), nbins)
    return {"freqs": freqs, "x_inc": x_inc, "x_new": x_new}


def normalized_power(sft_matrix, normalisation=MAP_NORMALISATION):
    """|SFT| normalised per frequency bin by its median over time.

    With ``normalisation="bin+sft"`` every SFT is then also divided by its median
    over the band (rescaled so that a stationary SFT keeps a factor ~1), which
    flattens glitches that raise the whole band in a few SFTs. A signal fills a few
    bins of an SFT, so it barely moves that median.
    Input shape (n_bins, n_sft); output shape (n_sft, n_bins).
    """
    if normalisation not in MAP_NORMALISATION_MODES:
        raise ValueError(f"Unknown map normalisation {normalisation!r}; expected one of {MAP_NORMALISATION_MODES}")
    magnitude = np.abs(sft_matrix)
    if not np.all(np.isfinite(magnitude)):
        bad = np.flatnonzero(~np.all(np.isfinite(magnitude), axis=0))
        raise ValueError(f"{bad.size} SFT(s) contain non-finite values (first: SFT {bad[0]}); "
                         "the strain of this chunk has NaN/inf samples.")
    noise_floor = np.median(magnitude, axis=1) / (2 * np.log(2))
    if np.any(noise_floor <= 0):
        raise ValueError(f"{int(np.sum(noise_floor <= 0))} frequency bin(s) have zero median |SFT| over the chunk; "
                         "the strain of this chunk is mostly zero (missing or zero-padded frames).")
    power = magnitude / noise_floor[:, np.newaxis]
    if normalisation == "bin+sft":
        sft_floor = np.median(power, axis=0) / (2 * np.log(2))
        if np.any(sft_floor <= 0):
            bad = np.flatnonzero(sft_floor <= 0)
            raise ValueError(f"{bad.size} SFT(s) have zero median |SFT| over the band (first: SFT {bad[0]}), so "
                             "'bin+sft' cannot normalise them: the strain has a gap or zero-padded frame there "
                             "(o3.download pads short frames with zeros).")
        power = power / sft_floor[np.newaxis, :]
    return power.T


def remap_to_fm83(power, x_inc, x_new, fill_value=np.nan):
    """Linearly interpolate each time step from the f grid onto the uniform f^-8/3 grid."""
    power = np.asarray(power)
    power_inc = power[:, ::-1]
    remapped = np.empty((power_inc.shape[0], x_new.size), dtype=float)
    for i in range(power_inc.shape[0]):
        remapped[i] = np.interp(x_new, x_inc, power_inc[i], left=fill_value, right=fill_value)
    return remapped


def remapped_map(sft_matrix, tsft, fmin=FMIN, normalisation=MAP_NORMALISATION):
    """Return (x_grid, remapped_power) for an SFT matrix of shape (n_bins, n_sft)."""
    geometry = build_remap_geometry(tsft, fmin, sft_matrix.shape[0])
    remapped = remap_to_fm83(normalized_power(sft_matrix, normalisation), geometry["x_inc"], geometry["x_new"])
    return geometry["x_new"], remapped


def viterbi_track(remapped_power):
    """Most likely track (bin index per time step) through a remapped map, via soapcw."""
    import soapcw

    result = soapcw.single_detector(VITERBI_TRANSITION_LOG_PROBS, remapped_power, lookup_table=None)
    return np.asarray(result.vit_track, dtype=int)


def track_product(sft_matrix, tsft, fmin=FMIN, normalisation=MAP_NORMALISATION):
    """Remapped map and Viterbi track of one SFT matrix (n_bins, n_sft).

    Returns a dict with ``tsft``, ``track_index`` (bin per time step),
    ``track_freq`` (track in f^-8/3 units) and ``power`` (remapped map,
    shape (n_sft, n_bins)).
    """
    x_grid, remapped = remapped_map(sft_matrix, tsft, fmin, normalisation)
    track_index = viterbi_track(remapped)
    return {
        "tsft": float(tsft),
        "track_index": track_index,
        "track_freq": np.asarray(x_grid[track_index], dtype=float),
        "power": np.asarray(remapped, dtype=float),
    }


def process_tsft_fast(tsft, strain, sample_rate, fmin=FMIN, band=FBAND, prefiltered=False,
                      normalisation=MAP_NORMALISATION):
    """Track product of one Tsft from the strain array of a chunk (in-memory SFTs)."""
    from pbh_viterbi.sft.fast import sft_matrix

    return track_product(sft_matrix(strain, sample_rate, tsft, fmin, band, prefiltered=prefiltered), tsft, fmin,
                         normalisation)


def network_track_product(sft_matrices, tsft, fmin=FMIN, normalisation=MAP_NORMALISATION):
    """Track product of the summed map of several detectors (same chunk, same Tsft).

    Each detector's |SFT| map is normalised by its own per-bin median
    (``normalized_power``), the maps are summed with equal weights and divided by
    sqrt(N), and the result is remapped and tracked once. The sqrt(N) keeps the
    pixel noise fluctuation of a single map: soapcw adds the transition
    log-probabilities to the map values, so the prior keeps the weight it has in
    the one-detector search. The inter-detector frequency offset (Doppler,
    arrival-time delay) is far below one bin for these signals, so the signal
    falls in the same pixel of every map (studies/multi_detector).
    """
    shapes = {m.shape for m in sft_matrices}
    if len(shapes) != 1:
        raise ValueError(f"SFT matrices of the detectors differ in shape: {sorted(shapes)}")
    geometry = build_remap_geometry(tsft, fmin, sft_matrices[0].shape[0])
    summed = np.sum([normalized_power(m, normalisation) for m in sft_matrices], axis=0) / np.sqrt(len(sft_matrices))
    remapped = remap_to_fm83(summed, geometry["x_inc"], geometry["x_new"])
    track_index = viterbi_track(remapped)
    return {
        "tsft": float(tsft),
        "track_index": track_index,
        "track_freq": np.asarray(geometry["x_new"][track_index], dtype=float),
        "power": np.asarray(remapped, dtype=float),
    }


def process_tsft_fast_network(tsft, strains, sample_rate, fmin=FMIN, band=FBAND, prefiltered=False,
                              normalisation=MAP_NORMALISATION):
    """Track product of one Tsft from the strain arrays of several detectors (in-memory SFTs)."""
    from pbh_viterbi.sft.fast import sft_matrix

    return network_track_product([sft_matrix(s, sample_rate, tsft, fmin, band, prefiltered=prefiltered)
                                  for s in strains], tsft, fmin, normalisation)


def process_tsft(tsft, t_start, t_end, framecache, num_threads, fmin=FMIN, band=FBAND, verbose_sft=False,
                 normalisation=MAP_NORMALISATION):
    """Track product of one Tsft with lalpulsar_MakeSFTs on frame files (integer Tsft).

    SFTs are written to a temporary folder and discarded afterwards.
    """
    from pbh_viterbi.sft.load import load_sft_matrix
    from pbh_viterbi.sft.make_sfts import make_sfts

    if float(tsft) != int(tsft):
        raise ValueError(f"lalpulsar_MakeSFTs needs an integer Tsft, got {tsft}; use the fast SFT method.")
    tsft = int(tsft)
    nbins = n_frequency_bins(tsft, band)
    n_sft = int(PACK_DURATION // tsft)
    with tempfile.TemporaryDirectory(prefix=f"sft-tsft{tsft}-") as sft_dir:
        tic = time.perf_counter()
        make_sfts(t_start, t_end, tsft, framecache, sft_dir, num_threads, fmin=fmin, band=band, verbose=verbose_sft)
        log.info("MakeSFTs tsft=%s s took %.1f s", tsft, time.perf_counter() - tic)
        sft_matrix = load_sft_matrix(sft_dir, t_start, tsft, nbins, n_sft)
    return track_product(sft_matrix, tsft, fmin, normalisation)
