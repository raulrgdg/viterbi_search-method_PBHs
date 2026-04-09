import os

import matplotlib.pyplot as plt
import numpy as np
import soapcw as soap

from pipeline.sft.load_sft import LoadSFT

VITERBI_TRANSITION_LOG_PROBS = np.log([0.30, 0.35, 0.35])


def load_sfts(sftdir_h1, gpstime_start, tsft, nbins, nsft):
    """Load a sequence of H1 SFT files into a complex array with one bulk read."""
    sft_paths = [
        f"{sftdir_h1}/H-1_H1_{tsft}SFT_MSFT-{gpstime_start + k * tsft}-{tsft}.sft"
        for k in range(nsft)
    ]
    missing_paths = [path for path in sft_paths if not os.path.exists(path)]
    if missing_paths:
        raise FileNotFoundError(
            f"Missing {len(missing_paths)} SFT files for tsft={tsft}. First missing file: {missing_paths[0]}"
        )

    sft_bundle = LoadSFT(";".join(sft_paths), norm_timebin_power=True).H1.sft
    if sft_bundle.shape != (nsft, nbins):
        raise ValueError(
            f"Unexpected SFT array shape {sft_bundle.shape}; expected ({nsft}, {nbins}) for tsft={tsft}"
        )

    return np.asarray(sft_bundle, dtype=np.complex128).T


def build_remap_geometry(tsft, fmin, nbins):
    """Precompute the static frequency/remap geometry for a given tsft."""
    delta_f = 1 / tsft
    freqs = fmin + np.arange(nbins) * delta_f
    if np.any(freqs <= 0):
        raise ValueError("freqs debe ser >0 para usar f^{-8/3}")

    x_inc = freqs[::-1] ** (-8 / 3)
    x_new = np.linspace(x_inc.min(), x_inc.max(), nbins)
    return {
        "freqs": freqs,
        "x_inc": x_inc,
        "x_new": x_new,
    }


def preprocess_data(data_h1, tsft, fmin, fmax, freqs=None):
    """Normalize SFT amplitudes and return the associated frequency grid."""
    if freqs is None:
        delta_f = 1 / tsft
        nbins = data_h1.shape[0]
        freqs = fmin + np.arange(nbins) * delta_f
    magnitude = np.abs(data_h1)
    psd_h1 = np.median(magnitude, axis=1) / (2 * np.log(2))
    cshuster_h1 = (magnitude / psd_h1[:, np.newaxis]).T
    return cshuster_h1, freqs


def remap_CShuster_to_fm83(cshuster, freqs, x_new=None, fill_value=np.nan, x_inc=None):
    """Remap a frequency grid into an evenly sampled ``f^(-8/3)`` coordinate."""
    cshuster = np.asarray(cshuster)

    if x_inc is None:
        freqs = np.asarray(freqs)
        if np.any(freqs <= 0):
            raise ValueError("freqs debe ser >0 para usar f^{-8/3}")
        x_inc = freqs[::-1] ** (-8 / 3)
    else:
        x_inc = np.asarray(x_inc)
    c_inc = cshuster[:, ::-1]

    if x_new is None:
        # Use a uniform x-grid so imshow/Viterbi consume a regular coordinate system.
        x_new = np.linspace(x_inc.min(), x_inc.max(), c_inc.shape[1])

    c_new = np.empty((c_inc.shape[0], x_new.size), dtype=float)
    for i in range(c_inc.shape[0]):
        c_new[i] = np.interp(x_new, x_inc, c_inc[i], left=fill_value, right=fill_value)

    return x_new, c_new

def run_viterbi(cshuster_h1, freqs_filtered, tsft, fmin, fmax, output_txt, output_power, output_index, output_png):
    """Run Viterbi tracking and persist the selected track products."""
    one_tracks_ng = soap.single_detector(
        VITERBI_TRANSITION_LOG_PROBS,
        cshuster_h1,
        lookup_table=None,
    )

    np.save(output_power, cshuster_h1)
    track_freqs = freqs_filtered[one_tracks_ng.vit_track]
    vit_tracks = one_tracks_ng.vit_track
    np.savetxt(output_txt, track_freqs, fmt="%.10f")
    np.savetxt(output_index, vit_tracks, fmt="%d")
    if output_png:
        soap.plots.plot_single(cshuster_h1, soapout=one_tracks_ng, tsft=tsft, fmin=fmin, fmax=fmax)
        plt.savefig(output_png)
        plt.close()

    return track_freqs, one_tracks_ng
