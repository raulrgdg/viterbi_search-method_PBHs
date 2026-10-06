"""Optimal SNR of long inspirals and noise PSD estimation (Appendix A)."""

import numpy as np

from pbh_viterbi.config import (
    FMAX,
    FMIN,
    FRAME_LENGTH,
    IFO,
    INCLINATION,
    NUM_FRAMES,
    SAMPLE_RATE,
    t_to_merger_for_mchirp,
)

PSD_SEGMENT_SECONDS = 512
PSD_STRIDE_SECONDS = 256
PSD_AVG_METHOD = "median"


def _trapezoid(y, x):
    return np.trapezoid(y, x=x) if hasattr(np, "trapezoid") else np.trapz(y, x=x)


def welch_psd(strain, sample_rate=SAMPLE_RATE):
    """Welch PSD of one frame: 512 s Hann segments, 50% overlap, median average."""
    from pycbc.psd import welch

    return welch(
        strain,
        seg_len=int(PSD_SEGMENT_SECONDS * sample_rate),
        seg_stride=int(PSD_STRIDE_SECONDS * sample_rate),
        avg_method=PSD_AVG_METHOD,
    )


def load_psd_csv(path):
    """Read a two-column (frequency_hz, psd) CSV as (freqs, psd) arrays."""
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.shape[0] < 2:
        raise ValueError(f"PSD file has too few frequency bins: {path}")
    return np.asarray(data[:, 0], dtype=float), np.asarray(data[:, 1], dtype=float)


def frame_snr2(h_detector, psd_freqs, psd_values, flow=FMIN, fhigh=FMAX):
    """rho^2 = 4 int_{flow}^{fhigh} |h(f)|^2 / S(f) df for one frame, Eq. (A3).

    The PSD is linearly interpolated onto the frequency grid of the waveform.
    """
    htilde = h_detector.to_frequencyseries()
    freqs = np.asarray(htilde.sample_frequencies)
    band = (freqs > flow) & (freqs < fhigh)
    if np.count_nonzero(band) < 2:
        raise ValueError(f"No frequency bins in the SNR band [{flow}, {fhigh}] Hz.")
    psd = np.interp(freqs[band], psd_freqs, psd_values)
    return 4 * _trapezoid(np.abs(np.asarray(htilde)[band]) ** 2 / psd, x=freqs[band])


def frame_polarizations(m1, m2, distance, t_start, num_frames=NUM_FRAMES, frame_length=FRAME_LENGTH,
                        sample_rate=SAMPLE_RATE):
    """(hp, hc) of an injection starting at ``t_start``, one pair per frame."""
    from pycbc.conversions import mchirp_from_mass1_mass2

    from pbh_viterbi.waveform.taylor_t3 import TaylorT3

    mchirp = mchirp_from_mass1_mass2(m1, m2)
    waveform = TaylorT3(m1=m1, m2=m2, distance=distance, inclination=INCLINATION, sampling_rate=sample_rate,
                        coal_time=int(t_start + t_to_merger_for_mchirp(mchirp)))
    return [
        waveform.tdstrain(t_start + i * frame_length, t_start + (i + 1) * frame_length, PyCBC_TimeSeries=True)
        for i in range(num_frames)
    ]


def optimal_snr(polarizations, ra, dec, pol, psd_freqs, psd_values, flow=FMIN, fhigh=FMAX, ifo=IFO):
    """Optimal SNR summed incoherently over frames, Eqs. (A4)-(A5)."""
    from pycbc.detector import Detector

    detector = Detector(ifo)
    snr2 = sum(
        frame_snr2(detector.project_wave(hp, hc, ra, dec, pol, method="lal"), psd_freqs, psd_values, flow, fhigh)
        for hp, hc in polarizations
    )
    return float(np.sqrt(snr2))
