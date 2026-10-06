"""Pieces shared by the noise and injected search entry points."""

import argparse
import logging
from pathlib import Path

from pbh_viterbi.config import FMAX, FMIN, MAKE_SFT_THREADS, TSFT_VALUES
from pbh_viterbi.paths import DEFAULT_NOISE_BACKGROUND, O3_DIR, o3_pack_dir
from pbh_viterbi.sft.maps import process_tsft

log = logging.getLogger(__name__)


def add_common_arguments(parser):
    parser.add_argument("--n-jobs", type=int, default=1, help="Total number of parallel jobs.")
    parser.add_argument("--job-id", type=int, default=0, help="Index of this job in [0, n_jobs).")
    parser.add_argument("--o3-dir", type=Path, default=O3_DIR, help="Folder with the O3b-packN frame folders.")
    parser.add_argument("--noise-background", type=Path, default=DEFAULT_NOISE_BACKGROUND,
                        help="CSV with the per-Tsft mean/std of the noise track power (nsigma normalisation).")
    parser.add_argument("--tsft", type=int, nargs="+", default=list(TSFT_VALUES), help="SFT durations (s).")
    parser.add_argument("--fmin", type=float, default=FMIN, help="Lower edge of the search band (Hz).")
    parser.add_argument("--fmax", type=float, default=FMAX, help="Upper edge of the search band (Hz).")
    parser.add_argument("--threads", type=int, default=MAKE_SFT_THREADS,
                        help="Parallel lalpulsar_MakeSFTs processes per Tsft.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging, including MakeSFTs output.")
    return parser


def setup_logging(verbose):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def existing_pack_dir(pack, o3_dir):
    pack_dir = o3_pack_dir(pack, o3_dir)
    if not pack_dir.is_dir():
        raise FileNotFoundError(f"O3 pack not found: {pack_dir}. Download it first with pbh_viterbi.o3.download.")
    return pack_dir


def tsft_products(tsft_values, t_start, t_end, framecache, threads, fmin=FMIN, fmax=FMAX, verbose=False):
    """Remapped maps and Viterbi tracks of one chunk for every Tsft."""
    results = []
    for tsft in tsft_values:
        log.debug("Processing tsft=%d s", tsft)
        results.append(process_tsft(tsft, t_start, t_end, framecache, threads, fmin=fmin, band=fmax - fmin,
                                    verbose_sft=verbose))
    return results


def new_parser(description):
    return argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
