"""Download O3b strain from GWOSC, resample it to 512 Hz and store it as GWF frames.

Example:
    python -m pbh_viterbi.o3.download --packs 1-12
    python -m pbh_viterbi.o3.download --packs all --n-jobs 5 --job-id 0
"""

import argparse
import logging
import time
import urllib.error
from pathlib import Path

import numpy as np

from pbh_viterbi.config import CHANNEL, FRAME_LENGTH, GWOSC_SAMPLE_RATE, IFO, NUM_FRAMES, SAMPLE_RATE
from pbh_viterbi.jobs import split_for_job
from pbh_viterbi.o3.frames import frame_start_times, raw_frame_name
from pbh_viterbi.o3.packs import pack_window, parse_pack_list
from pbh_viterbi.paths import O3_DIR, ensure_dir, o3_pack_dir

log = logging.getLogger(__name__)

RETRYABLE_ERRORS = (
    urllib.error.ContentTooShortError,
    urllib.error.URLError,
    urllib.error.HTTPError,
    TimeoutError,
    ConnectionError,
    OSError,
)


def _fetch_with_retries(ifo, start, end, retry_attempts, retry_wait_seconds):
    from gwpy.timeseries import TimeSeries

    for attempt in range(1, retry_attempts + 2):
        try:
            # Disable the GWOSC cache on retries so partial downloads are not reused.
            return TimeSeries.fetch_open_data(
                ifo, start, end, sample_rate=GWOSC_SAMPLE_RATE, verbose=False, cache=(attempt == 1)
            )
        except RETRYABLE_ERRORS as exc:
            if attempt > retry_attempts:
                raise RuntimeError(
                    f"Could not download [{start}, {end}) after {retry_attempts + 1} attempts."
                ) from exc
            wait_s = retry_wait_seconds * attempt
            log.warning("Download of [%d, %d) failed (attempt %d): %s. Retrying in %ds.",
                        start, end, attempt, exc, wait_s)
            time.sleep(wait_s)
    raise AssertionError("unreachable")


def download_pack(pack, o3_dir=None, ifo=IFO, channel=CHANNEL, sample_rate=SAMPLE_RATE,
                  retry_attempts=3, retry_wait_seconds=5):
    """Download one pack into ``<o3_dir>/O3b-pack<pack>`` and return the written files."""
    from pycbc import frame as pycbc_frame
    from pycbc import types as pycbc_types

    t_start, _ = pack_window(pack)
    output_dir = ensure_dir(o3_pack_dir(pack, o3_dir))
    expected_samples = FRAME_LENGTH * sample_rate
    written = []

    for start in frame_start_times(t_start, NUM_FRAMES, FRAME_LENGTH):
        series = _fetch_with_retries(ifo, start, start + FRAME_LENGTH, retry_attempts, retry_wait_seconds)
        if int(series.sample_rate.value) != sample_rate:
            series = series.resample(sample_rate)

        data = np.asarray(series.value, dtype=np.float64)
        if data.size < expected_samples:
            data = np.pad(data, (0, expected_samples - data.size), mode="constant")
        elif data.size > expected_samples:
            data = data[:expected_samples]

        output_file = Path(output_dir) / raw_frame_name(start, ifo, FRAME_LENGTH)
        pycbc_frame.write_frame(
            str(output_file), channel, pycbc_types.TimeSeries(data, delta_t=1.0 / sample_rate, epoch=start)
        )
        written.append(output_file)
        log.info("Saved %s", output_file)

    return written


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--packs", default="all", help='"all", "5", "1,2,3" or "1-12,37-48".')
    parser.add_argument("--o3-dir", type=Path, default=O3_DIR, help="Destination folder.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Total number of parallel jobs.")
    parser.add_argument("--job-id", type=int, default=0, help="Index of this job in [0, n_jobs).")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    packs = split_for_job(parse_pack_list(args.packs), args.n_jobs, args.job_id)
    if not packs:
        log.info("Job %d/%d has no packs assigned.", args.job_id, args.n_jobs)
        return
    log.info("Job %d/%d downloads packs %s into %s", args.job_id, args.n_jobs, packs, args.o3_dir)
    for index, pack in enumerate(packs, start=1):
        log.info("[%d/%d] Downloading O3b pack %d", index, len(packs), pack)
        download_pack(pack, args.o3_dir)
    log.info("Download finished.")


if __name__ == "__main__":
    main()
