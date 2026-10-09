#!/usr/bin/env python3
"""Test 2: SFTs, remapped maps and Viterbi tracks of MakeSFTs vs fast_sft, all 13 Tsft.

Both methods run on the same raw frames of 2 random packs (pipeline band
61.1-126.8 Hz). One row per (pack, Tsft) in results/maps_2packs.csv.

    python maps_2packs.py --threads 12
"""

import argparse
import tempfile

from common import O3_DIR, RESULTS, append_csv, available_packs, compare_chunk, concat_strain, random_packs

from pbh_viterbi.config import FMAX, FMIN, TSFT_VALUES
from pbh_viterbi.o3.frames import read_pack_frames, write_raw_framecache
from pbh_viterbi.o3.packs import pack_window
from pbh_viterbi.paths import o3_pack_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-packs", type=int, default=2)
    parser.add_argument("--threads", type=int, default=12, help="Parallel MakeSFTs processes.")
    parser.add_argument("--fmax", type=float, default=FMAX)
    parser.add_argument("--output", default=str(RESULTS / "maps_2packs.csv"))
    args = parser.parse_args()

    packs = random_packs(args.n_packs, available_packs(), stream=2)
    print(f"Packs: {packs}", flush=True)
    for pack in packs:
        t_start, t_end = pack_window(pack)
        pack_dir = o3_pack_dir(pack, O3_DIR)
        strain, fs = concat_strain(read_pack_frames(pack_dir, t_start))
        with tempfile.TemporaryDirectory(prefix=f"cmp-maps-pack{pack}-") as work:
            cache = write_raw_framecache(f"{work}/framecache", pack_dir, t_start)
            rows, _, _ = compare_chunk(t_start, t_end, cache, strain, fs, TSFT_VALUES, args.threads, FMIN,
                                       args.fmax, {"pack": pack})
        append_csv(rows, args.output)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
