"""Reading SFT files written by lalpulsar_MakeSFTs."""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pbh_viterbi.config import IFO


@dataclass
class SFTData:
    """SFTs of one detector: ``sft`` has shape (n_sft, n_bins)."""

    sft: np.ndarray
    epochs: np.ndarray
    frequencies: np.ndarray
    tsft: float
    delta_f: float
    fmin: float


def read_sfts(sft_paths, detector=IFO, fmin=None, fmax=None):
    """Load a list of SFT files with LALPulsar and return the data of one detector."""
    import lalpulsar

    catalogue = lalpulsar.SFTdataFind(";".join(str(path) for path in sft_paths), lalpulsar.SFTConstraints())
    multi_sfts = lalpulsar.LoadMultiSFTs(catalogue, -1 if fmin is None else fmin, -1 if fmax is None else fmax)

    for det_sfts in multi_sfts.data:
        first = det_sfts.data[0]
        if first.name != detector:
            continue
        n_sft, n_bins = det_sfts.length, first.data.length
        sft = np.zeros((n_sft, n_bins), dtype=np.complex128)
        epochs = np.zeros(n_sft)
        for i, single in enumerate(det_sfts.data):
            sft[i, :] = single.data.data
            epochs[i] = single.epoch
        return SFTData(
            sft=sft,
            epochs=epochs,
            frequencies=np.arange(n_bins) * first.deltaF + first.f0,
            tsft=1.0 / first.deltaF,
            delta_f=first.deltaF,
            fmin=first.f0,
        )

    raise ValueError(f"No SFTs of detector {detector!r} were found.")


def expected_sft_paths(sft_dir, t_start, tsft, n_sft, detector=IFO):
    """Paths of the ``n_sft`` consecutive SFTs MakeSFTs writes from ``t_start``."""
    return [
        Path(sft_dir) / f"{detector[0]}-1_{detector}_{tsft}SFT_MSFT-{t_start + k * tsft}-{tsft}.sft"
        for k in range(n_sft)
    ]


def load_sft_matrix(sft_dir, t_start, tsft, n_bins, n_sft, detector=IFO):
    """Load the SFTs of one pack as a complex array of shape (n_bins, n_sft).

    Fails if any expected SFT is missing or the array has an unexpected shape.
    """
    paths = expected_sft_paths(sft_dir, t_start, tsft, n_sft, detector)
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} SFT files for tsft={tsft}; first: {missing[0]}")

    sft = read_sfts(paths, detector).sft
    if sft.shape != (n_sft, n_bins):
        raise ValueError(f"Unexpected SFT array shape {sft.shape}; expected ({n_sft}, {n_bins}) for tsft={tsft}")
    return np.asarray(sft, dtype=np.complex128).T


def load_sft_dir(sft_dir, detector=IFO):
    """Load every SFT in a folder (sorted by name), e.g. for plotting."""
    paths = sorted(Path(sft_dir).glob("*.sft"))
    if not paths:
        raise FileNotFoundError(f"No SFT files were found in {sft_dir}")
    return read_sfts(paths, detector)
