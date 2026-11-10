"""Search over O3b noise (no injections).

Two modes:

  search      Run the full pipeline on each pack and write one trigger
              (nsigma, NMSE) per pack to results/search/search_results_noise.csv.
              These triggers are the background used to set the FAR threshold.

  background  Write the total Viterbi track power of each pack and Tsft to
              results/search/noise_track_power.csv. Summarise it with
              analysis/noise_background_stats.py to obtain the mean/std per
              Tsft that normalise nsigma (Eq. 17). Needed once per Tsft list
              and band; calibration/ holds the files of the shipped lists.

The SFT durations come from --tsft-list (default "95"), the band from
config.py (61.1-126.8 Hz), the same for every search product.

Examples:
    python -m pbh_viterbi.workflows.noise_search --packs 3
    python -m pbh_viterbi.workflows.noise_search --packs all --n-jobs 108 --job-id 0
    python -m pbh_viterbi.workflows.noise_search --mode background --tsft-list 90 --packs all
"""

import logging

from pbh_viterbi.jobs import split_for_job
from pbh_viterbi.o3.frames import write_raw_framecache
from pbh_viterbi.o3.packs import pack_window, parse_pack_list
from pbh_viterbi.results import (
    FIELDNAMES,
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
from pbh_viterbi.search.power import check_background, load_noise_background, total_track_power
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

BACKGROUND_CSV = "noise_track_power.csv"
BACKGROUND_FIELDS = ["pack", "tsft", "total_power"]


def parse_args():
    parser = add_common_arguments(new_parser(__doc__))
    parser.add_argument("--mode", choices=("search", "background"), default="search")
    parser.add_argument("--packs", default="all", help='"all", "5", "1,2,3" or "1-12,37-48".')
    return resolve_tsft_and_background(parser.parse_args())


def process_pack(pack, args, noise_background, per_tsft_output=None):
    t_start, t_end = pack_window(pack)
    pack_dir = existing_pack_dir(pack, args.o3_dir)
    segments = read_detector_frames(pack_dir, t_start, args.detectors) if args.sft_method == "fast" else None
    products = chunk_products(args, segments, t_start, t_end,
                              lambda work_dir: write_raw_framecache(f"{work_dir}/framecache", pack_dir, t_start))

    if args.mode == "background":
        return [
            {"pack": pack, "tsft": float(p["tsft"]), "total_power": float(total_track_power(p["track_index"], p["power"]))}
            for p in products
        ]
    result = search_candidate(products, *noise_background)
    if per_tsft_output is not None:  # per-Tsft products (Tsft-selection study)
        rows = result.get("per_tsft") or per_tsft_candidates(products, *noise_background)
        append_rows(per_tsft_rows(pack, rows, injected=False), per_tsft_output, PER_TSFT_FIELDNAMES)
    return [result_row(pack, result, injected=False)]


def main():
    args = parse_args()
    setup_logging(args.verbose)

    background = args.mode == "background"
    name, fields = (BACKGROUND_CSV, BACKGROUND_FIELDS) if background else (csv_name(injected=False), FIELDNAMES)
    noise_background = None
    if not background:
        noise_background = load_noise_background(args.noise_background)
        check_background(args.tsft, noise_background[0], args.noise_background)

    packs = split_for_job(parse_pack_list(args.packs), args.n_jobs, args.job_id)
    output = shard_path(name, args.n_jobs, args.job_id)
    log.info("Job %d/%d (%s mode, %s, %s SFTs, %s high-pass, %s normalisation, %d Tsft): packs %s -> %s",
             args.job_id, args.n_jobs, args.mode, "+".join(args.detectors), args.sft_method, args.sft_highpass,
             args.map_normalisation, len(args.tsft), packs, output)
    write_rows([], output, fields)
    per_tsft = args.per_tsft_output and not background
    per_tsft_name = per_tsft_csv_name(injected=False)
    per_tsft_output = shard_path(per_tsft_name, args.n_jobs, args.job_id) if per_tsft else None
    if per_tsft:
        write_rows([], per_tsft_output, PER_TSFT_FIELDNAMES)

    for index, pack in enumerate(packs, start=1):
        log.info("[%d/%d] Pack %d", index, len(packs), pack)
        append_rows(process_pack(pack, args, noise_background, per_tsft_output), output, fields)

    finish_shard(name, args.n_jobs, args.job_id, fields)
    if per_tsft:
        finish_shard(per_tsft_name, args.n_jobs, args.job_id, PER_TSFT_FIELDNAMES)
    log.info("Job finished.")


if __name__ == "__main__":
    main()
