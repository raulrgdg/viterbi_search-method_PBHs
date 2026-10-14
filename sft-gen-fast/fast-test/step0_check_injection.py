#!/usr/bin/env python3
"""Step 0: in-memory injection == injection written to GWF frames and read back.

One Mc = 0.085 Msun signal in pack 25. Compares the strain bit by bit and the
Viterbi products of Tsft = 1 and 88 s. Output: results/step0_injection_check.csv
"""

import tempfile

import numpy as np
from common import RESULTS, append_csv, concat, inject_in_memory, products, read_pack

from pbh_viterbi.config import DEFAULT_SKY_SEED, t_to_merger_for_mchirp
from pbh_viterbi.o3.frames import frame_start_times, injected_frame_label
from pbh_viterbi.waveform.injection import equal_component_masses, inject_signal, sample_sky

PACK, MCHIRP, DISTANCE, SIGNAL_ID = 25, 0.085, 0.07498942093324558, 0


def main():
    from pycbc import frame as pycbc_frame
    from pycbc.conversions import mchirp_from_mass1_mass2

    from pbh_viterbi.config import CHANNEL, FRAME_LENGTH

    raw, t_start = read_pack(PACK)
    m1, m2 = equal_component_masses(MCHIRP)
    mchirp = mchirp_from_mass1_mass2(m1, m2)
    sky = sample_sky(DEFAULT_SKY_SEED, PACK, SIGNAL_ID)
    args = (m1, m2, DISTANCE, t_to_merger_for_mchirp(mchirp), sky["ra"], sky["dec"], sky["pol"])

    memory, fs = concat(inject_in_memory(raw, t_start, *args))
    with tempfile.TemporaryDirectory(prefix="step0-") as work:
        coal_time, frame_dir = inject_signal(*args, t_start, raw, work)
        segments = [pycbc_frame.read_frame(
            f"{frame_dir}/{injected_frame_label(mchirp, DISTANCE, coal_time, s)}.gwf", CHANNEL,
            start_time=s, end_time=s + FRAME_LENGTH) for s in frame_start_times(t_start)]
        frames, _ = concat(segments)

    row = {"pack": PACK, "mchirp": mchirp, "distance": DISTANCE,
           "strain_identical": bool(np.array_equal(memory, frames)),
           "strain_max_abs_diff": float(np.max(np.abs(memory - frames)))}
    for p_mem, p_frm in zip(products(memory, fs, (1, 88)), products(frames, fs, (1, 88))):
        t = p_mem["tsft"]
        row[f"map_identical_tsft{t}"] = bool(np.array_equal(p_mem["power"], p_frm["power"], equal_nan=True))
        row[f"track_identical_tsft{t}"] = bool(np.array_equal(p_mem["track_index"], p_frm["track_index"]))
    append_csv([row], RESULTS / "step0_injection_check.csv")
    print(row, flush=True)


if __name__ == "__main__":
    main()
