"""Pieces shared by the noise and injected search entry points."""

import argparse
import logging
import tempfile
from pathlib import Path

import numpy as np

from pbh_viterbi.config import (DEFAULT_TSFT_LIST, DETECTORS, FMAX, FMIN, MAKE_SFT_THREADS, MAP_NORMALISATION,
                                MAP_NORMALISATION_MODES, SFT_HIGHPASS, SFT_HIGHPASS_MODES, SFT_METHOD,
                                SUPPORTED_DETECTORS, TSFT_LISTS, channel_for)
from pbh_viterbi.paths import O3_DIR, noise_background_path, o3_pack_dir
from pbh_viterbi.o3.frames import read_pack_frames
from pbh_viterbi.sft.fast import highpass
from pbh_viterbi.sft.maps import process_tsft, process_tsft_fast, process_tsft_fast_network

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
                             "Default: calibration/noise_background_tsft<list><suffix>.csv, the suffix naming the "
                             "detectors, high-pass and normalisation (paths.config_suffix).")
    parser.add_argument("--fmin", type=float, default=FMIN, help="Lower edge of the search band (Hz).")
    parser.add_argument("--fmax", type=float, default=FMAX, help="Upper edge of the search band (Hz).")
    parser.add_argument("--sft-method", choices=("fast", "makesfts"), default=SFT_METHOD,
                        help="In-memory SFTs (default) or lalpulsar_MakeSFTs on frame files (paper method).")
    parser.add_argument("--sft-highpass", choices=SFT_HIGHPASS_MODES, default=SFT_HIGHPASS,
                        help="High-pass of the in-memory SFTs: 'segment' (default, every SFT segment, as MakeSFTs) "
                             "or 'chunk' (the whole chunk once).")
    parser.add_argument("--map-normalisation", choices=MAP_NORMALISATION_MODES, default=MAP_NORMALISATION,
                        help="'bin' (default): every frequency bin divided by its median over the chunk; "
                             "'bin+sft': then every SFT by its median over the band (flattens broadband glitches).")
    parser.add_argument("--threads", type=int, default=MAKE_SFT_THREADS,
                        help="Parallel lalpulsar_MakeSFTs processes per Tsft (makesfts method only).")
    parser.add_argument("--per-tsft-output", action="store_true",
                        help="Also isolate a candidate on every Tsft and write one row per chunk and "
                             "Tsft to results/search/per_tsft_*.csv (Tsft-selection study).")
    parser.add_argument("--detectors", nargs="+", choices=SUPPORTED_DETECTORS, default=list(DETECTORS),
                        help="Detectors whose normalised maps are summed (default H1; e.g. --detectors H1 L1). "
                             "Every detector needs its frames of the same chunk in the pack folder. "
                             "In-memory SFTs only.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging, including MakeSFTs output.")
    return parser


def resolve_tsft_and_background(args):
    """Fill ``args.tsft`` and ``args.noise_background`` from ``--tsft-list`` when not given."""
    if args.tsft is None:
        args.tsft = list(TSFT_LISTS[args.tsft_list])
    args.detectors = list(dict.fromkeys(args.detectors))  # unique, in the given order
    if len(args.detectors) > 1 and args.sft_method != "fast":
        raise ValueError("Several detectors need the in-memory SFTs (--sft-method fast).")
    if args.sft_highpass != "segment" and args.sft_method != "fast":
        raise ValueError(f"--sft-method {args.sft_method} high-passes every SFT segment (lalpulsar_MakeSFTs); "
                         "add --sft-highpass segment ('chunk' needs --sft-method fast).")
    if args.noise_background is None:
        args.noise_background = noise_background_path(args.tsft_list, args.detectors, args.sft_highpass,
                                                      args.map_normalisation)
    if getattr(args, "mode", "search") != "background" and not Path(args.noise_background).exists():
        raise FileNotFoundError(
            f"No nsigma background {args.noise_background} for Tsft list {args.tsft_list}, detectors "
            f"{'+'.join(args.detectors)}, --sft-highpass {args.sft_highpass}, --map-normalisation "
            f"{args.map_normalisation}. Compute it with the same options (scripts/run_local.sh background, or "
            "noise_search --mode background and analysis/noise_background_stats.py) or pass --noise-background.")
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


def read_detector_frames(pack_dir, t_start, detectors):
    """Raw frames of a chunk: a list of segments for one detector, ``{ifo: segments}`` for several."""
    if list(detectors) == ["H1"]:
        return read_pack_frames(pack_dir, t_start)
    return {ifo: read_pack_frames(pack_dir, t_start, channel=channel_for(ifo), ifo=ifo) for ifo in detectors}


def fast_products(segments, tsft_values, fmin=FMIN, fmax=FMAX, sft_highpass=SFT_HIGHPASS,
                  normalisation=MAP_NORMALISATION):
    """Remapped maps and Viterbi tracks of one chunk for every Tsft, with in-memory SFTs.

    ``segments`` is a list of segments (one detector) or ``{ifo: segments}``
    (several detectors: their normalised maps are summed). With ``sft_highpass="chunk"``
    each detector's strain is high-passed once as a whole instead of per SFT segment.
    ``normalisation`` is the map normalisation (``normalized_power``).
    """
    chunk_hp = sft_highpass == "chunk"
    if isinstance(segments, dict) and len(segments) == 1:
        segments = next(iter(segments.values()))
    if isinstance(segments, dict):
        strains = [concat_segments(s) for s in segments.values()]
        if len({len(s) for s, _ in strains}) != 1 or len({rate for _, rate in strains}) != 1:
            raise ValueError("The detectors' strain differs in length or sample rate for this chunk.")
        sample_rate = strains[0][1]
        arrays = [highpass(s, sample_rate, fmin) if chunk_hp else s for s, _ in strains]
        return [process_tsft_fast_network(tsft, arrays, sample_rate, fmin=fmin, band=fmax - fmin, prefiltered=chunk_hp,
                                           normalisation=normalisation)
                for tsft in tsft_values]
    strain, sample_rate = concat_segments(segments)
    if chunk_hp:
        strain = highpass(strain, sample_rate, fmin)
    results = []
    for tsft in tsft_values:
        log.debug("Processing tsft=%g s", tsft)
        results.append(process_tsft_fast(tsft, strain, sample_rate, fmin=fmin, band=fmax - fmin, prefiltered=chunk_hp,
                                          normalisation=normalisation))
    return results


def makesfts_products(tsft_values, t_start, t_end, framecache, threads, fmin=FMIN, fmax=FMAX, verbose=False,
                      normalisation=MAP_NORMALISATION):
    """Remapped maps and Viterbi tracks of one chunk for every Tsft, with lalpulsar_MakeSFTs."""
    results = []
    for tsft in tsft_values:
        log.debug("Processing tsft=%g s", tsft)
        results.append(process_tsft(tsft, t_start, t_end, framecache, threads, fmin=fmin, band=fmax - fmin,
                                    verbose_sft=verbose, normalisation=normalisation))
    return results


def chunk_products(args, segments, t_start, t_end, write_framecache):
    """Track products of one chunk with the SFT method selected in ``args``.

    ``segments`` is the strain of the chunk (list of pycbc TimeSeries). For the
    makesfts method, ``write_framecache(work_dir)`` must write the frames that
    hold this strain and return the path of their frame cache.
    """
    if args.sft_method == "fast":
        return fast_products(segments, args.tsft, args.fmin, args.fmax, args.sft_highpass, args.map_normalisation)
    with tempfile.TemporaryDirectory(prefix="makesfts-") as work_dir:
        framecache = write_framecache(work_dir)
        return makesfts_products(args.tsft, t_start, t_end, framecache, args.threads, args.fmin, args.fmax,
                                 args.verbose, args.map_normalisation)


def new_parser(description):
    return argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
