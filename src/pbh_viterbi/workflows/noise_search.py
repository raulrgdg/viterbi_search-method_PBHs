"""Search over O3b noise (no injections).

Two modes:

  search      Run the full pipeline on each pack and write one trigger
              (nsigma, NMSE) per pack to results/search/search_results_noise.csv.
              These triggers are the background used to set the FAR threshold.

  background  Write the total Viterbi track power of each pack and Tsft to
              results/search/noise_track_power.csv. Summarise it with
              analysis/noise_background_stats.py to obtain the mean/std per
              Tsft that normalise nsigma (Eq. 17).

Examples:
    python -m pbh_viterbi.workflows.noise_search --packs 1-108 --n-jobs 108 --job-id 0
    python -m pbh_viterbi.workflows.noise_search --mode background --packs 3
"""

import logging
import tempfile

from pbh_viterbi.jobs import split_for_job
from pbh_viterbi.o3.frames import write_raw_framecache
from pbh_viterbi.o3.packs import pack_window, parse_pack_list
from pbh_viterbi.results import FIELDNAMES, append_rows, csv_name, finish_shard, result_row, shard_path, write_rows
from pbh_viterbi.search.candidates import search_candidate
from pbh_viterbi.search.power import load_noise_background, total_track_power
from pbh_viterbi.workflows.common import (
    add_common_arguments,
    existing_pack_dir,
    new_parser,
    setup_logging,
    tsft_products,
)

log = logging.getLogger(__name__)

BACKGROUND_CSV = "noise_track_power.csv"
BACKGROUND_FIELDS = ["pack", "tsft", "total_power"]


def parse_args():
    parser = add_common_arguments(new_parser(__doc__))
    parser.add_argument("--mode", choices=("search", "background"), default="search")
    parser.add_argument("--packs", default="all", help='"all", "5", "1,2,3" or "1-12,37-48".')
    return parser.parse_args()


def process_pack(pack, args, noise_background):
    t_start, t_end = pack_window(pack)
    pack_dir = existing_pack_dir(pack, args.o3_dir)
    with tempfile.TemporaryDirectory(prefix=f"noise-pack{pack}-") as work_dir:
        framecache = write_raw_framecache(f"{work_dir}/framecache", pack_dir, t_start)
        products = tsft_products(args.tsft, t_start, t_end, framecache, args.threads, args.fmin, args.fmax, args.verbose)

    if args.mode == "background":
        return [
            {"pack": pack, "tsft": int(p["tsft"]), "total_power": float(total_track_power(p["track_index"], p["power"]))}
            for p in products
        ]
    return [result_row(pack, search_candidate(products, *noise_background), injected=False)]


def main():
    args = parse_args()
    setup_logging(args.verbose)

    background = args.mode == "background"
    name, fields = (BACKGROUND_CSV, BACKGROUND_FIELDS) if background else (csv_name(injected=False), FIELDNAMES)
    noise_background = None if background else load_noise_background(args.noise_background)

    packs = split_for_job(parse_pack_list(args.packs), args.n_jobs, args.job_id)
    output = shard_path(name, args.n_jobs, args.job_id)
    log.info("Job %d/%d (%s mode): packs %s -> %s", args.job_id, args.n_jobs, args.mode, packs, output)
    write_rows([], output, fields)

    for index, pack in enumerate(packs, start=1):
        log.info("[%d/%d] Pack %d", index, len(packs), pack)
        append_rows(process_pack(pack, args, noise_background), output, fields)

    finish_shard(name, args.n_jobs, args.job_id, fields)
    log.info("Job finished.")


if __name__ == "__main__":
    main()
