import os
import time
import urllib.error
import numpy as np
from gwpy.timeseries import TimeSeries as GWpyTimeSeries
from pycbc import frame as pycbc_frame
from pycbc import types as pycbc_types


def download_o3_real_data(
    *,
    input_dir: str,
    ifo: str,
    run_label: tuple[int, int],
    num_frames: int,
    frame_length: int,
    fs_target: int,
    channel_name: str,
    verbose: bool = True,
    retry_attempts: int = 3,
    retry_wait_seconds: int = 5,
):
    """Download open O3 strain, resample it, and persist it as GWF frames."""
    t_start, t_end = run_label

    os.makedirs(input_dir, exist_ok=True)
    expected_n = frame_length * fs_target
    generated_files = []
    retryable_errors = (
        urllib.error.ContentTooShortError,
        urllib.error.URLError,
        urllib.error.HTTPError,
        TimeoutError,
        ConnectionError,
        OSError,
    )

    for i in range(num_frames):
        seg_start = t_start + i * frame_length
        seg_end = seg_start + frame_length

        ts = None
        last_error = None
        for attempt in range(1, retry_attempts + 2):
            try:
                # Force fresh download on retries to avoid reusing partial cache files.
                ts = GWpyTimeSeries.fetch_open_data(
                    ifo,
                    seg_start,
                    seg_end,
                    sample_rate=4096,
                    verbose=verbose,
                    cache=(attempt == 1),
                )
                break
            except retryable_errors as exc:
                last_error = exc
                if attempt > retry_attempts:
                    raise RuntimeError(
                        f"No se pudo descargar el segmento [{seg_start}, {seg_end}) "
                        f"despues de {retry_attempts + 1} intentos."
                    ) from exc

                wait_s = retry_wait_seconds * attempt
                print(
                    f"Descarga fallida para [{seg_start}, {seg_end}) "
                    f"(intento {attempt}/{retry_attempts + 1}): {exc}. "
                    f"Reintentando en {wait_s}s..."
                )
                time.sleep(wait_s)

        if ts is None:
            raise RuntimeError(
                f"No se pudo obtener datos para [{seg_start}, {seg_end})"
            ) from last_error

        if int(ts.sample_rate.value) != fs_target:
            ts = ts.resample(fs_target)

        data = np.asarray(ts.value, dtype=np.float64)
        if data.size < expected_n:
            data = np.pad(data, (0, expected_n - data.size), mode="constant")
        elif data.size > expected_n:
            data = data[:expected_n]

        pycbc_ts = pycbc_types.TimeSeries(data, delta_t=1.0 / fs_target, epoch=seg_start)
        output_file = os.path.join(
            input_dir,
            f"{ifo[0]}-{ifo}_GWOSC_O3b_4KHZ_R1-{seg_start}-{frame_length}_resampled_512HZ.gwf",
        )
        pycbc_frame.write_frame(output_file, channel_name, pycbc_ts)
        generated_files.append(output_file)
        print(f"Real data descargada y guardada: {output_file}")

    return generated_files
