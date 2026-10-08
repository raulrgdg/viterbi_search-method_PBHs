#!/usr/bin/env python3
"""Compare the remapped power maps and Viterbi tracks built from MakeSFTs and from fast_sft.

    python compare_maps.py --pack 4 --tsft 88 7
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fast_sft import sft_matrix  # noqa: E402

from pbh_viterbi.config import FMIN  # noqa: E402
from pbh_viterbi.o3.frames import read_pack_frames, write_raw_framecache  # noqa: E402
from pbh_viterbi.o3.packs import pack_window  # noqa: E402
from pbh_viterbi.paths import o3_pack_dir  # noqa: E402
from pbh_viterbi.sft.maps import process_tsft, build_remap_geometry, normalized_power, remap_to_fm83, viterbi_track  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=int, default=4)
    parser.add_argument("--tsft", type=int, nargs="+", default=[88, 7])
    parser.add_argument("--fmax", type=float, default=127.2)
    parser.add_argument("--threads", type=int, default=24)
    args = parser.parse_args()
    band = args.fmax - FMIN

    t_start, t_end = pack_window(args.pack)
    pack_dir = o3_pack_dir(args.pack)
    segments = read_pack_frames(pack_dir, t_start)
    strain = np.concatenate([np.asarray(s, dtype=np.float64) for s in segments])
    for tsft in args.tsft:
        with tempfile.TemporaryDirectory() as work:
            cache = write_raw_framecache(f"{work}/cache", pack_dir, t_start)
            ref = process_tsft(tsft, t_start, t_end, cache, args.threads, band=band)
        sfts = sft_matrix(strain, 512.0, tsft, FMIN, band)
        geometry = build_remap_geometry(tsft, FMIN, sfts.shape[0])
        power = remap_to_fm83(normalized_power(sfts), geometry["x_inc"], geometry["x_new"])
        track = viterbi_track(power)
        rel = np.abs(power - ref["power"]) / np.abs(ref["power"])
        print(f"tsft={tsft}: map {power.shape}, identical={np.array_equal(power, ref['power'])}, "
              f"max rel diff {np.nanmax(rel):.1e}, track identical={np.array_equal(track, ref['track_index'])}")


if __name__ == "__main__":
    main()
