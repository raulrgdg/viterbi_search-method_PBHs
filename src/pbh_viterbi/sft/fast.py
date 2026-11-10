"""In-memory SFT generation, equivalent to lalpulsar_MakeSFTs as used by the search.

lalpulsar_MakeSFTs (with -w rectangular, -f FMIN, -F FMIN, -B BAND) processes
each SFT segment independently:

  1. high-pass filter the segment with a 10th-order Butterworth filter whose
     amplitude is 0.5 at the high-pass frequency (lal.ButterworthREAL8TimeSeries);
  2. FFT it and multiply by the sampling interval dt;
  3. keep round(BAND * Tsft) bins starting at bin round(FMIN * Tsft);
  4. store the result in single precision.

The same steps run here in memory on the strain of a whole chunk, without
writing frames or SFT files: the remapped maps and Viterbi tracks are identical
to those built from MakeSFTs (up to the last bit of single precision), ~100x
faster per chunk (docs/fast_sft.md).

Since v1.3 the search high-passes the whole chunk once by default (``prefiltered``,
``--sft-highpass chunk``), which MakeSFTs cannot do; ``--sft-highpass segment``
keeps the per-segment filter described above.
"""

import numpy as np

HIGHPASS_ORDER = 10
HIGHPASS_ATTENUATION = 0.5


def highpass(segment, sample_rate, f_highpass):
    """Butterworth high-pass of one segment, as done by MakeSFTs."""
    import lal

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


def samples_per_sft(tsft, sample_rate):
    """Number of samples of one SFT; Tsft must be a whole number of samples."""
    n = tsft * sample_rate
    if abs(n - round(n)) > 1e-6:
        raise ValueError(f"Tsft = {tsft} s is not a whole number of samples at {sample_rate} Hz.")
    return int(round(n))


def sft_matrix(strain, sample_rate, tsft, fmin, band, f_highpass=None, single_precision=True, prefiltered=False):
    """SFTs of a contiguous strain array as a complex array of shape (n_bins, n_sft).

    The strain must start at the first SFT; a tail shorter than ``tsft`` is dropped.
    The first bin is round(fmin * tsft), i.e. frequency round(fmin * tsft) / tsft.
    With ``prefiltered`` the strain was already high-passed as a whole (``highpass`` on
    the chunk) and the segments are not filtered again.
    """
    f_highpass = fmin if f_highpass is None else f_highpass
    n = samples_per_sft(tsft, sample_rate)
    n_sft = len(strain) // n
    first_bin = int(round(fmin * tsft))
    n_bins = int(round(band * tsft))

    segments = np.asarray(strain[: n_sft * n], dtype=np.float64).reshape(n_sft, n)
    if prefiltered:
        filtered = segments
    else:
        filtered = np.empty_like(segments)
        for k in range(n_sft):
            filtered[k] = highpass(segments[k], sample_rate, f_highpass)

    sfts = np.fft.rfft(filtered, axis=1)[:, first_bin:first_bin + n_bins] / sample_rate
    if single_precision:
        sfts = sfts.astype(np.complex64)
    return np.asarray(sfts, dtype=np.complex128).T
