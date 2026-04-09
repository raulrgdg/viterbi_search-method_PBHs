import os

import numpy as np
from pycbc import frame
from pycbc.conversions import mchirp_from_mass1_mass2
from pycbc.detector import Detector

from pipeline.waveforms.my_taylor_t3 import myTaylorT3


def _build_input_gwf_files(input_dir, ifo, t_start, num_frames, frame_length):
    """Build the expected raw input frame paths for a given job window."""
    return [
        f"{input_dir}/{ifo[0]}-{ifo}_GWOSC_O3b_4KHZ_R1-{t_start + i * frame_length}-{frame_length}_resampled_512HZ.gwf"
        for i in range(num_frames)
    ]


def _load_existing_data(input_gwf_files, channel_name, t_start, num_frames, frame_length):
    """Load raw strain segments from disk for later injection."""
    existing_data = []
    for i in range(num_frames):
        start_time = t_start + i * frame_length
        end_time = start_time + frame_length
        data = frame.read_frame(
            input_gwf_files[i],
            channel_name,
            start_time=start_time,
            end_time=end_time,
        )
        existing_data.append(data)
    return existing_data


def inject_signal_into_real_data(
    m1, m2, distance, t_to_merg , ra, dec, pol, inc,
    ifo, t_start, num_frames, frame_length, data_dir,
    input_dir, channel_name, existing_data=None, verbose=True):
    """Inject a synthetic signal into raw strain frames and write new frame files."""
    if existing_data is None:
        os.makedirs(input_dir, exist_ok=True)
        input_gwf_files = _build_input_gwf_files(input_dir, ifo, t_start, num_frames, frame_length)
        existing_data = _load_existing_data(
            input_gwf_files=input_gwf_files,
            channel_name=channel_name,
            t_start=t_start,
            num_frames=num_frames,
            frame_length=frame_length,
        )
    else:
        if len(existing_data) != num_frames:
            raise ValueError(
                f"existing_data tiene {len(existing_data)} segmentos, se esperaban {num_frames}."
            )

    coal_time = int(t_start + t_to_merg)  # Coalescence time

    wf_generator = myTaylorT3(
        m1=m1, m2=m2, distance=distance, inclination=inc,
        sampling_rate=1 / existing_data[0].delta_t, coal_time=coal_time
    )
    if verbose:
        print('GW generated.')

    mchirp = mchirp_from_mass1_mass2(m1, m2)
    distance_str = f"{distance:.3f}".replace(".", "_")
    output_dir = os.path.join(data_dir, f"{ifo}_inject_mc-{mchirp:.0e}_dl-{distance_str}")
    os.makedirs(output_dir, exist_ok=True)

    frame_name_template = "_O3b_mc_%.0e_dL_%.3f_tc_%.f_%.f-%.f.gwf"

    detector = Detector(ifo)
    for i in range(num_frames):
        t0 = t_start + i * frame_length
        tf = t0 + frame_length

        hp, hc = wf_generator.tdstrain(t0, tf, PyCBC_TimeSeries=True)
        projected_strain = detector.project_wave(hp, hc, ra, dec, pol, method="lal")

        injected_strain = existing_data[i].inject(projected_strain)

        output_file = os.path.join(
            output_dir,
            ifo + frame_name_template % (mchirp, distance, coal_time, t0, tf - t0)
        )
        frame.write_frame(output_file, channel_name, injected_strain)

    if verbose:
        print("Injection Done!")

    return coal_time
