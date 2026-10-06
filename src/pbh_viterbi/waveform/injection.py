"""Injection of simulated long inspirals into real detector frames."""

import logging
from pathlib import Path

import numpy as np

from pbh_viterbi.config import (
    CHANNEL,
    DEC_RANGE,
    DISTANCE_GRID,
    FRAME_LENGTH,
    IFO,
    INCLINATION,
    MASS_RATIO,
    MCHIRP_GRID,
    POL_RANGE,
    RA_RANGE,
)
from pbh_viterbi.o3.frames import injected_frame_label
from pbh_viterbi.waveform.taylor_t3 import TaylorT3

log = logging.getLogger(__name__)


def equal_component_masses(mchirp, q=MASS_RATIO):
    """Component masses (Msun) for a chirp mass and mass ratio, forcing m1 == m2 when q == 1."""
    from pycbc.pnutils import mchirp_q_to_mass1_mass2

    m1, m2 = mchirp_q_to_mass1_mass2(mchirp, q=q)
    if q == 1.0 and m1 != m2:
        m2 = m1
    return m1, m2


def build_signal_grid(mchirp_grid=MCHIRP_GRID, distance_grid=DISTANCE_GRID):
    """Injected population as a list of (m1, m2, distance_mpc), chirp mass outermost."""
    grid = []
    for mchirp in mchirp_grid:
        m1, m2 = equal_component_masses(mchirp)
        for distance in distance_grid:
            grid.append((m1, m2, distance))
    return grid


def sample_sky(seed, pack, signal_index):
    """Draw (ra, dec, pol) uniformly, reproducibly for a given pack and signal.

    The draw depends only on (seed, pack, signal_index), so it does not change
    with the number of jobs a campaign is split into.
    """
    rng = np.random.default_rng([int(seed), int(pack), int(signal_index)])
    return {
        "ra": float(rng.uniform(*RA_RANGE)),
        "dec": float(rng.uniform(*DEC_RANGE)),
        "pol": float(rng.uniform(*POL_RANGE)),
    }


def inject_signal(m1, m2, distance, t_to_merger, ra, dec, pol, t_start, raw_segments, output_root,
                  inc=INCLINATION, ifo=IFO, channel=CHANNEL, frame_length=FRAME_LENGTH):
    """Add a TaylorT3 inspiral to each raw frame and write the result as GWF files.

    The binary coalesces ``t_to_merger`` seconds after ``t_start``. The waveform
    is generated frame by frame and projected onto the detector with LAL.

    Returns ``(coal_time, frame_dir)``.
    """
    from pycbc import frame as pycbc_frame
    from pycbc.conversions import mchirp_from_mass1_mass2
    from pycbc.detector import Detector

    coal_time = int(t_start + t_to_merger)
    waveform = TaylorT3(
        m1=m1, m2=m2, distance=distance, inclination=inc,
        sampling_rate=1 / raw_segments[0].delta_t, coal_time=coal_time,
    )

    mchirp = mchirp_from_mass1_mass2(m1, m2)
    distance_str = f"{distance:.3f}".replace(".", "_")
    frame_dir = Path(output_root) / f"{ifo}_inject_mc-{mchirp:.0e}_dl-{distance_str}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    detector = Detector(ifo)
    for i, segment in enumerate(raw_segments):
        t0 = t_start + i * frame_length
        hp, hc = waveform.tdstrain(t0, t0 + frame_length, PyCBC_TimeSeries=True)
        projected = detector.project_wave(hp, hc, ra, dec, pol, method="lal")
        label = injected_frame_label(mchirp, distance, coal_time, t0, ifo, frame_length)
        pycbc_frame.write_frame(str(frame_dir / f"{label}.gwf"), channel, segment.inject(projected))

    log.debug("Injected mchirp=%.3g Msun at %.3f Mpc into %s", mchirp, distance, frame_dir)
    return coal_time, frame_dir
