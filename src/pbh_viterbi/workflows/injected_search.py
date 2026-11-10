"""Search over O3b noise with simulated long-inspiral signals injected (Table I).

The 600-signal population (20 chirp masses x 30 distances, see config.py) is
injected, one signal at a time, into the 32768 s of one O3b pack. For each
signal the pipeline adds the waveform to the strain in memory, builds the
remapped maps for all Tsft, tracks them with Viterbi and isolates the candidate.
With --sft-method makesfts the injected frames are written to a temporary
folder and lalpulsar_MakeSFTs is used instead (paper method).

By default the signals processed for a pack are the slice assigned to that
pack by the campaign (see pbh_viterbi.campaign); use --signals to pick
another range. The signals of the slice are split across --n-jobs jobs.

Sky location and polarisation are drawn uniformly from (--sky-seed, pack,
signal index), so a campaign is reproducible whatever its job split, and they
are written to the output with the other injection parameters, so the optimal
SNR of every injection can be computed afterwards (analysis/injection_snr.py).
With --sky RA DEC POL every signal uses that fixed sky position instead.

--signal-ids picks some signals of the grid, and --signal-list replaces the grid by a CSV
of signals (signal_id, mchirp, distance), e.g. studies/multi_detector/quick_signals.csv.

Output: results/search/search_results_injected_pack-<pack>.csv

Examples:
    python -m pbh_viterbi.workflows.injected_search --pack 1 --signals 300:301
    python -m pbh_viterbi.workflows.injected_search --pack 1 --n-jobs 10 --job-id 3
"""

import logging
from pathlib import Path

from pbh_viterbi.campaign import assignment_for_pack
from pbh_viterbi.config import DEFAULT_SKY_SEED, INCLINATION, t_to_merger_for_mchirp
from pbh_viterbi.jobs import split_for_job
from pbh_viterbi.o3.frames import write_injected_framecache
from pbh_viterbi.o3.packs import pack_window
from pbh_viterbi.results import (
    PER_TSFT_FIELDNAMES,
    append_rows,
    csv_name,
    finish_shard,
    per_tsft_csv_name,
    per_tsft_rows,
    result_row,
    shard_path,
    write_rows,
)
from pbh_viterbi.search.candidates import per_tsft_candidates, search_candidate
from pbh_viterbi.search.power import check_background, load_noise_background
from pbh_viterbi.waveform.injection import (build_signal_grid, equal_component_masses, inject_in_memory,
                                            inject_in_memory_network, inject_signal, sample_sky)
from pbh_viterbi.workflows.common import (
    add_common_arguments,
    chunk_products,
    existing_pack_dir,
    new_parser,
    read_detector_frames,
    resolve_tsft_and_background,
    setup_logging,
)

log = logging.getLogger(__name__)


def read_signal_list(path):
    """``{signal_id: (m1, m2, distance)}`` from a CSV with signal_id, mchirp and distance (Mpc)."""
    import csv

    signals = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            signal_id = int(row["signal_id"])
            if signal_id in signals:
                raise ValueError(f"Signal id {signal_id} appears twice in {path}")
            m1, m2 = equal_component_masses(float(row["mchirp"]))
            signals[signal_id] = (m1, m2, float(row["distance"]))
    if not signals:
        raise ValueError(f"No signals in {path}")
    return signals


def parse_signal_ids(text):
    """Sorted unique signal ids of a comma-separated list; an empty or malformed list is an error."""
    try:
        ids = sorted({int(v) for v in text.split(",") if v.strip()})
    except ValueError as exc:
        raise ValueError(f"--signal-ids must be comma-separated integers, got {text!r}") from exc
    if not ids:
        raise ValueError(f"--signal-ids is empty: {text!r}")
    return ids


def parse_signal_range(text, n_signals):
    start, end = (int(v) for v in text.split(":"))
    if not 0 <= start < end <= n_signals:
        raise ValueError(f"--signals must satisfy 0 <= start < end <= {n_signals}")
    return start, end


def parse_args():
    parser = add_common_arguments(new_parser(__doc__))
    parser.add_argument("--pack", type=int, required=True, help="O3b pack to inject into.")
    parser.add_argument("--signals", help="Signal index range START:END of the population grid "
                                          "(default: the campaign slice of the pack).")
    parser.add_argument("--signal-list", type=Path,
                        help="CSV with columns signal_id, mchirp (Msun), distance (Mpc): signals outside the "
                             "population grid (equal masses, t_to_merger of the chirp mass). The sky comes from "
                             "(sky seed, pack, signal_id); use ids >= 1000 to keep them apart from the grid.")
    parser.add_argument("--signal-ids", help="Comma-separated signal indices of the population grid "
                                             "(overrides --signals), e.g. 218,219,250.")
    parser.add_argument("--sky-seed", type=int, default=DEFAULT_SKY_SEED)
    parser.add_argument("--sky", type=float, nargs=3, metavar=("RA", "DEC", "POL"),
                        help="Fixed sky position and polarisation (rad) for every signal.")
    return resolve_tsft_and_background(parser.parse_args())


def injected_products(args, m1, m2, distance, mchirp, t_to_merger, sky, t_start, t_end, raw_segments):
    """Track products of one injection with the selected SFT method."""
    if args.sft_method == "fast" and isinstance(raw_segments, dict):
        _, segments = inject_in_memory_network(m1, m2, distance, t_to_merger, sky["ra"], sky["dec"], sky["pol"],
                                               t_start, raw_segments)
        return chunk_products(args, segments, t_start, t_end, None)
    if args.sft_method == "fast":
        _, segments = inject_in_memory(m1, m2, distance, t_to_merger, sky["ra"], sky["dec"], sky["pol"],
                                       t_start, raw_segments)
        return chunk_products(args, segments, t_start, t_end, None)

    def write_framecache(work_dir):
        coal_time, frame_dir = inject_signal(m1, m2, distance, t_to_merger, sky["ra"], sky["dec"], sky["pol"],
                                             t_start, raw_segments, work_dir)
        return write_injected_framecache(f"{work_dir}/framecache", frame_dir, mchirp, distance, coal_time, t_start)

    return chunk_products(args, None, t_start, t_end, write_framecache)


def main():
    args = parse_args()
    setup_logging(args.verbose)
    from pycbc.conversions import mchirp_from_mass1_mass2

    if args.signals and (args.signal_list or args.signal_ids):
        raise ValueError("--signals cannot be combined with --signal-list or --signal-ids")
    grid = build_signal_grid()
    if args.signal_list:
        grid = read_signal_list(args.signal_list)
        chosen = sorted(grid)
        if args.signal_ids:
            wanted = parse_signal_ids(args.signal_ids)
            missing = sorted(set(wanted) - set(grid))
            if missing:
                raise ValueError(f"--signal-ids {missing} are not in --signal-list {args.signal_list}")
            chosen = wanted
    elif args.signal_ids:
        chosen = parse_signal_ids(args.signal_ids)
        if chosen[0] < 0 or chosen[-1] >= len(grid):
            raise ValueError(f"--signal-ids must be in [0, {len(grid)}) for the population grid")
    elif args.signals:
        start, end = parse_signal_range(args.signals, len(grid))
    else:
        assignment = assignment_for_pack(args.pack)
        start, end = assignment.signal_start, assignment.signal_end
        log.info("Pack %d: campaign signals [%d, %d)", args.pack, start, end)

    if not (args.signal_list or args.signal_ids):
        chosen = list(range(start, end))
    signal_ids = split_for_job(chosen, args.n_jobs, args.job_id)
    name = csv_name(injected=True, pack=args.pack)
    output = shard_path(name, args.n_jobs, args.job_id)
    log.info("Job %d/%d (%s, %s SFTs, %s high-pass, %s normalisation, %d Tsft): %d signal(s) %s -> %s",
             args.job_id, args.n_jobs, "+".join(args.detectors), args.sft_method, args.sft_highpass,
             args.map_normalisation, len(args.tsft), len(signal_ids),
             f"[{signal_ids[0]}..{signal_ids[-1]}]" if signal_ids else "[]", output)
    write_rows([], output)
    per_tsft_name = per_tsft_csv_name(injected=True, pack=args.pack)  # Tsft-selection study
    per_tsft_output = shard_path(per_tsft_name, args.n_jobs, args.job_id) if args.per_tsft_output else None
    if per_tsft_output is not None:
        write_rows([], per_tsft_output, PER_TSFT_FIELDNAMES)

    if signal_ids:
        noise_means, noise_stds = load_noise_background(args.noise_background)
        check_background(args.tsft, noise_means, args.noise_background)
        t_start, t_end = pack_window(args.pack)
        raw_segments = read_detector_frames(existing_pack_dir(args.pack, args.o3_dir), t_start, args.detectors)

        for count, signal_id in enumerate(signal_ids, start=1):
            m1, m2, distance = grid[signal_id]
            mchirp = mchirp_from_mass1_mass2(m1, m2)
            t_to_merger = t_to_merger_for_mchirp(mchirp)
            if args.sky:
                sky = dict(zip(("ra", "dec", "pol"), args.sky))
            else:
                sky = sample_sky(args.sky_seed, args.pack, signal_id)
            log.info("[%d/%d] signal %d: mchirp=%.4g Msun, dL=%.4g Mpc, ra=%.3f, dec=%.3f, pol=%.3f",
                     count, len(signal_ids), signal_id, mchirp, distance, sky["ra"], sky["dec"], sky["pol"])

            products = injected_products(args, m1, m2, distance, mchirp, t_to_merger, sky, t_start, t_end,
                                         raw_segments)
            result = search_candidate(products, noise_means, noise_stds)
            append_rows([result_row(args.pack, result, injected=True, mchirp=mchirp, distance=distance, sky=sky,
                                    signal_id=signal_id, inclination=INCLINATION, t_to_merger=t_to_merger)],
                        output)
            if per_tsft_output is not None:
                rows = result.get("per_tsft") or per_tsft_candidates(products, noise_means, noise_stds)
                append_rows(per_tsft_rows(args.pack, rows, injected=True, signal_id=signal_id), per_tsft_output,
                            PER_TSFT_FIELDNAMES)

    finish_shard(name, args.n_jobs, args.job_id)
    if per_tsft_output is not None:
        finish_shard(per_tsft_name, args.n_jobs, args.job_id, PER_TSFT_FIELDNAMES)
    log.info("Job finished.")


if __name__ == "__main__":
    main()
