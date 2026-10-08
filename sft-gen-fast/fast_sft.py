"""In-memory SFT generation, equivalent to lalpulsar_MakeSFTs as used by pbh_viterbi.

lalpulsar_MakeSFTs (with -w rectangular, -f FMIN, -F FMIN, -B BAND) processes
each SFT segment independently:

  1. high-pass filter the segment with a 10th-order Butterworth filter whose
     amplitude is 0.5 at the high-pass frequency (lal.ButterworthREAL8TimeSeries);
  2. FFT it and multiply by the sampling interval dt;
  3. keep round(BAND * Tsft) bins starting at bin round(FMIN * Tsft);
  4. store the result in single precision (.sft files).

Here the same steps run in memory on the strain of a whole chunk, without
writing frames or SFT files. The result has the shape (n_bins, n_sft) expected
by pbh_viterbi.sft.maps.
"""

import numpy as np

import lal

HIGHPASS_ORDER = 10
HIGHPASS_ATTENUATION = 0.5


def highpass(segment, sample_rate, f_highpass):
    """Butterworth high-pass of one segment, as done by MakeSFTs."""
    series = lal.CreateREAL8TimeSeries("x", lal.LIGOTimeGPS(0), 0, 1.0 / sample_rate, lal.DimensionlessUnit,
                                       len(segment))
    series.data.data = np.asarray(segment, dtype=np.float64)
    params = lal.PassBandParamStruc()
    params.nMax = HIGHPASS_ORDER
    params.f2 = f_highpass
    params.a2 = HIGHPASS_ATTENUATION
    params.f1 = -1.0
    params.a1 = -1.0
    lal.ButterworthREAL8TimeSeries(series, params)
    return series.data.data


def sft_matrix(strain, sample_rate, tsft, fmin, band, f_highpass=None, single_precision=True):
    """SFTs of a contiguous strain array as a complex array of shape (n_bins, n_sft).

    The strain must start at the first SFT; a tail shorter than ``tsft`` is dropped.
    """
    f_highpass = fmin if f_highpass is None else f_highpass
    n = int(round(tsft * sample_rate))
    n_sft = len(strain) // n
    first_bin = int(round(fmin * tsft))
    n_bins = int(round(band * tsft))

    segments = np.asarray(strain[: n_sft * n], dtype=np.float64).reshape(n_sft, n)
    filtered = np.empty_like(segments)
    for k in range(n_sft):
        filtered[k] = highpass(segments[k], sample_rate, f_highpass)

    sfts = np.fft.rfft(filtered, axis=1)[:, first_bin:first_bin + n_bins] / sample_rate
    if single_precision:
        sfts = sfts.astype(np.complex64)
    return np.asarray(sfts, dtype=np.complex128).T
