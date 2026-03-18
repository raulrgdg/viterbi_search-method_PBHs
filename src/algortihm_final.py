import os

import matplotlib.pyplot as plt
import numpy as np
import soapcw as soap

from modulo import LoadSFT

VITERBI_TRANSITION_LOG_PROBS = np.log([0.30, 0.35, 0.35])


def load_sfts(sftdir_h1, gpstime_start, tsft, nbins, nsft):
    """Load a sequence of H1 SFT files into a complex array."""
    data_h1 = np.zeros((nbins, nsft), dtype=np.complex128)
    for k in range(nsft):
        sftfile_h1 = (
            f"{sftdir_h1}/H-1_H1_{tsft}SFT_MSFT-{gpstime_start + k * tsft}-{tsft}.sft"
        )
        if not os.path.exists(sftfile_h1):
            print(f"⚠️ SFT file missing: {sftfile_h1}")
            break
        data_h1[:, k] = LoadSFT(sftfile_h1, norm_timebin_power=True).H1.sft[0]
    return data_h1


def preprocess_data(data_h1, tsft, fmin, fmax):
    """Normalize SFT amplitudes and return the associated frequency grid."""
    delta_f = 1 / tsft
    nbins = data_h1.shape[0]

    freqs = fmin + np.arange(nbins) * delta_f
    psd_h1 = np.median(np.abs(data_h1), axis=1) / (2 * np.log(2))
    cshuster_h1 = np.transpose(np.abs(data_h1) / (psd_h1[:, np.newaxis]))
    return cshuster_h1, freqs

def remap_CShuster_to_fm83(cshuster, freqs, x_new=None, fill_value=np.nan):
    """Remap a frequency grid into an evenly sampled ``f^(-8/3)`` coordinate."""
    freqs = np.asarray(freqs)
    cshuster = np.asarray(cshuster)

    if np.any(freqs <= 0):
        raise ValueError("freqs debe ser >0 para usar f^{-8/3}")

    x = freqs ** (-8 / 3)
    x_inc = x[::-1]
    c_inc = cshuster[:, ::-1]

    if x_new is None:
        # Use a uniform x-grid so imshow/Viterbi consume a regular coordinate system.
        x_new = np.linspace(x_inc.min(), x_inc.max(), freqs.size)

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
