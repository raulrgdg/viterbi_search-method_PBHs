"""Pieces shared by the noise and injected search entry points."""

import argparse
import logging
import tempfile
from pathlib import Path

import numpy as np

from pbh_viterbi.config import DEFAULT_TSFT_LIST, FMAX, FMIN, MAKE_SFT_THREADS, SFT_METHOD, TSFT_LISTS
from pbh_viterbi.paths import O3_DIR, noise_background_path, o3_pack_dir
from pbh_viterbi.sft.maps import process_tsft, process_tsft_fast

log = logging.getLogger(__name__)


def add_common_arguments(parser):
    parser.add_argument("--n-jobs", type=int, default=1, help="Total number of parallel jobs.")
    parser.add_argument("--job-id", type=int, default=0, help="Index of this job in [0, n_jobs).")
    parser.add_argument("--o3-dir", type=Path, default=O3_DIR, help="Folder with the O3b-packN frame folders.")
    parser.add_argument("--tsft-list", choices=sorted(TSFT_LISTS), default=DEFAULT_TSFT_LIST,
                        help="SFT durations: '95' or '90' (nsigma loss below 5%% or 10%%) or 'paper' (13 values).")
    parser.add_argument("--tsft", type=float, nargs="+", default=None,
                        help="Explicit SFT durations (s); overrides --tsft-list.")
    parser.add_argument("--noise-background", type=Path, default=None,
                        help="CSV with the per-Tsft mean/std of the noise track power (nsigma normalisation). "
                             "Default: calibration/noise_background_tsft<list>.csv.")
    parser.add_argument("--fmin", type=float, default=FMIN, help="Lower edge of the search band (Hz).")
    parser.add_argument("--fmax", type=float, default=FMAX, help="Upper edge of the search band (Hz).")
    parser.add_argument("--sft-method", choices=("fast", "makesfts"), default=SFT_METHOD,
                        help="In-memory SFTs (default) or lalpulsar_MakeSFTs on frame files (paper method).")
    parser.add_argument("--threads", type=int, default=MAKE_SFT_THREADS,
                        help="Parallel lalpulsar_MakeSFTs processes per Tsft (makesfts method only).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging, including MakeSFTs output.")
    return parser


def resolve_tsft_and_background(args):
    """Fill ``args.tsft`` and ``args.noise_background`` from ``--tsft-list`` when not given."""
    if args.tsft is None:
        args.tsft = list(TSFT_LISTS[args.tsft_list])
    if args.noise_background is None:
        args.noise_background = noise_background_path(args.tsft_list)
    return args


def setup_logging(verbose):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def existing_pack_dir(pack, o3_dir):
    pack_dir = o3_pack_dir(pack, o3_dir)
    if not pack_dir.is_dir():
        raise FileNotFoundError(f"O3 pack not found: {pack_dir}. Download it first with pbh_viterbi.o3.download.")
    return pack_dir


def concat_segments(segments):
    """Contiguous float64 strain and sample rate of a list of pycbc TimeSeries."""
    return np.concatenate([np.asarray(s, dtype=np.float64) for s in segments]), 1.0 / segments[0].delta_t


def fast_products(segments, tsft_values, fmin=FMIN, fmax=FMAX):
    """Remapped maps and Viterbi tracks of one chunk for every Tsft, with in-memory SFTs."""
    strain, sample_rate = concat_segments(segments)
    results = []
    for tsft in tsft_values:
        log.debug("Processing tsft=%g s", tsft)
        results.append(process_tsft_fast(tsft, strain, sample_rate, fmin=fmin, band=fmax - fmin))
    return results


def makesfts_products(tsft_values, t_start, t_end, framecache, threads, fmin=FMIN, fmax=FMAX, verbose=False):
    """Remapped maps and Viterbi tracks of one chunk for every Tsft, with lalpulsar_MakeSFTs."""
    results = []
    for tsft in tsft_values:
        log.debug("Processing tsft=%g s", tsft)
        results.append(process_tsft(tsft, t_start, t_end, framecache, threads, fmin=fmin, band=fmax - fmin,
                                    verbose_sft=verbose))
    return results


def chunk_products(args, segments, t_start, t_end, write_framecache):
    """Track products of one chunk with the SFT method selected in ``args``.

    ``segments`` is the strain of the chunk (list of pycbc TimeSeries). For the
    makesfts method, ``write_framecache(work_dir)`` must write the frames that
    hold this strain and return the path of their frame cache.
    """
    if args.sft_method == "fast":
        return fast_products(segments, args.tsft, args.fmin, args.fmax)
    with tempfile.TemporaryDirectory(prefix="makesfts-") as work_dir:
        framecache = write_framecache(work_dir)
        return makesfts_products(args.tsft, t_start, t_end, framecache, args.threads, args.fmin, args.fmax,
                                 args.verbose)


def new_parser(description):
    return argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
