"""Shared helpers of the 14-Tsft test (Tsft = 1 s added to the 13 of the paper)."""

import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(REPO / "src"))
from fast_sft import sft_matrix  # noqa: E402

from pbh_viterbi.config import FMAX, FMIN, FRAME_LENGTH, INCLINATION, IFO, TSFT_VALUES  # noqa: E402
from pbh_viterbi.o3.frames import read_pack_frames  # noqa: E402
from pbh_viterbi.o3.packs import pack_window  # noqa: E402
from pbh_viterbi.search.power import total_track_power  # noqa: E402
from pbh_viterbi.sft.maps import build_remap_geometry, normalized_power, remap_to_fm83, viterbi_track  # noqa: E402

TSFT_13 = tuple(TSFT_VALUES)
TSFT_14 = (1,) + TSFT_13
RESULTS = HERE / "results"
LOGS = HERE / "logs"
PAPER_BACKGROUND = REPO / "paper_data" / "calibration" / "noise_power_background.csv"
BACKGROUND_14 = RESULTS / "noise_background_14tsft.csv"

# The 106 packs of the old repository plus packs 87 and 88 downloaded into data/o3.
O3_DIRS = (REPO.parent / "viterbi_search-method_PBHs" / "data" / "raw" / "O3_data", REPO / "data" / "o3")


def pack_dir(pack):
    for root in O3_DIRS:
        path = root / f"O3b-pack{pack}"
        if len(list(path.glob("*512HZ.gwf"))) == 8:
            return path
    raise FileNotFoundError(f"O3b pack {pack} not found in {O3_DIRS}")


def read_pack(pack):
    """Raw segments (pycbc TimeSeries) and t_start of one pack."""
    t_start = pack_window(pack)[0]
    return read_pack_frames(pack_dir(pack), t_start), t_start


def concat(segments):
    return np.concatenate([np.asarray(s, dtype=np.float64) for s in segments]), 1.0 / segments[0].delta_t


def products(strain, sample_rate, tsft_values=TSFT_14, fmin=FMIN, fmax=FMAX):
    """Per-Tsft remapped maps and Viterbi tracks (same dicts as pbh_viterbi.sft.maps.process_tsft)."""
    out = []
    for tsft in tsft_values:
        sfts = sft_matrix(strain, sample_rate, tsft, fmin, fmax - fmin)
        geometry = build_remap_geometry(tsft, fmin, sfts.shape[0])
        power = remap_to_fm83(normalized_power(sfts), geometry["x_inc"], geometry["x_new"])
        track = viterbi_track(power)
        out.append({"tsft": tsft, "track_index": track, "track_freq": np.asarray(geometry["x_new"][track], dtype=float),
                    "power": np.asarray(power, dtype=float)})
    return out


def track_powers(prods):
    return {int(p["tsft"]): float(total_track_power(p["track_index"], p["power"])) for p in prods}


def inject_in_memory(raw_segments, t_start, m1, m2, distance, t_to_merger, ra, dec, pol, inc=INCLINATION,
                     ifo=IFO, frame_length=FRAME_LENGTH):
    """Same as pbh_viterbi.waveform.injection.inject_signal, without writing frames.

    Returns the injected segments (list of pycbc TimeSeries).
    """
    from pycbc.detector import Detector

    from pbh_viterbi.waveform.taylor_t3 import TaylorT3

    waveform = TaylorT3(m1=m1, m2=m2, distance=distance, inclination=inc,
                        sampling_rate=1 / raw_segments[0].delta_t, coal_time=int(t_start + t_to_merger))
    detector = Detector(ifo)
    injected = []
    for i, segment in enumerate(raw_segments):
        t0 = t_start + i * frame_length
        hp, hc = waveform.tdstrain(t0, t0 + frame_length, PyCBC_TimeSeries=True)
        injected.append(segment.inject(detector.project_wave(hp, hc, ra, dec, pol, method="lal")))
    return injected


def load_background(path):
    with open(path) as handle:
        rows = list(csv.DictReader(handle))
    return ({int(r["tsft"]): float(r["mean_total_power"]) for r in rows},
            {int(r["tsft"]): float(r["std_total_power"]) for r in rows})


def append_csv(rows, path):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if new:
            writer.writeheader()
        writer.writerows(rows)


def done_keys(path, keys):
    """Keys already written to a CSV (to resume an interrupted run)."""
    try:
        with open(path) as handle:
            return {tuple(r[k] for k in keys) for r in csv.DictReader(handle)}
    except FileNotFoundError:
        return set()
