#!/usr/bin/env python3
"""Compare lalpulsar_MakeSFTs output with a direct in-memory FFT of the same strain.

Finds which FFT bins MakeSFTs returns and how much its output differs from the
direct FFT before and after the per-bin median normalisation used by the search.

    python diagnose_makesfts.py --pack 4 --tsft 88 7
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pbh_viterbi.config import FBAND, FMIN, NUM_FRAMES, FRAME_LENGTH  # noqa: E402
from pbh_viterbi.o3.frames import read_pack_frames, write_raw_framecache  # noqa: E402
from pbh_viterbi.o3.packs import pack_window  # noqa: E402
from pbh_viterbi.paths import o3_pack_dir  # noqa: E402
from pbh_viterbi.sft.load import load_sft_matrix  # noqa: E402
from pbh_viterbi.sft.make_sfts import make_sfts  # noqa: E402
from pbh_viterbi.sft.maps import n_frequency_bins, normalized_power  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=int, default=4)
    parser.add_argument("--tsft", type=int, nargs="+", default=[88, 7])
    parser.add_argument("--threads", type=int, default=24)
    args = parser.parse_args()

    t_start, t_end = pack_window(args.pack)
    pack_dir = o3_pack_dir(args.pack)
    segments = read_pack_frames(pack_dir, t_start)
    strain = np.concatenate([np.asarray(s, dtype=np.float64) for s in segments])
    fs = 1.0 / segments[0].delta_t

    for tsft in args.tsft:
        nbins = n_frequency_bins(tsft)
        nsft = int(NUM_FRAMES * FRAME_LENGTH / tsft)
        with tempfile.TemporaryDirectory() as work:
            cache = write_raw_framecache(f"{work}/cache", pack_dir, t_start)
            make_sfts(t_start, t_end, tsft, cache, f"{work}/sfts", args.threads)
            ref = load_sft_matrix(f"{work}/sfts", t_start, tsft, nbins, nsft)  # (nbins, nsft)

        n = int(tsft * fs)
        fft = np.fft.rfft(strain[: nsft * n].reshape(nsft, n), axis=1).T  # (n/2+1, nsft)
        # Bin offset that best matches MakeSFTs (correlation of |SFT| along frequency).
        best = max(range(int(FMIN * tsft) - 3, int(FMIN * tsft) + 4),
                   key=lambda k0: np.corrcoef(np.abs(fft[k0:k0 + nbins, 0]), np.abs(ref[:, 0]))[0, 1])
        fast = fft[best:best + nbins]
        ratio = np.abs(ref) / np.abs(fast)
        print(f"tsft={tsft}: first bin {best} ({best / tsft:.3f} Hz, FMIN*tsft={FMIN * tsft:.2f}), "
              f"|SFT| ratio median {np.median(ratio):.4e}, spread over time "
              f"{np.median(np.std(ratio, axis=1) / np.mean(ratio, axis=1)):.2e}")
        # Ratio as a function of frequency (filter + normalisation) and of time (transients).
        per_bin = np.median(ratio, axis=1)
        per_sft = np.median(ratio / per_bin[:, None], axis=0)
        print(f"   per-bin gain: first {per_bin[0]:.4e} last {per_bin[-1]:.4e}; "
              f"per-SFT deviation max {np.max(np.abs(per_sft - 1)):.2e} at SFT {np.argmax(np.abs(per_sft - 1))}")
        a, b = normalized_power(ref), normalized_power(fast)
        rel = np.abs(a - b) / a
        print(f"   normalised power: max rel diff {rel.max():.2e}, median {np.median(rel):.2e}")


if __name__ == "__main__":
    main()
