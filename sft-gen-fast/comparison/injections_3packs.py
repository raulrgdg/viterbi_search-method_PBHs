#!/usr/bin/env python3
"""Test 3: injected search with MakeSFTs vs fast_sft on the same injected frames.

Three signals of the paper population, each injected into one random pack
(sky drawn as in the campaign, pbh_viterbi.waveform.injection.sample_sky):

  * Mc = 1.12e-3 Msun, dL = 5.62e-3 Mpc  (low mass, long optimal Tsft, ~59% recovered in the paper)
  * Mc = 4.47e-3 Msun, dL = 3.16e-2 Mpc  (low mass, ~82% recovered)
  * Mc = 7.08e-2 Msun, dL = 7.50e-2 Mpc  (high mass, short optimal Tsft, ~50% recovered)

The distances are close to the detection limit, where a small difference in the
maps would be most likely to change the result. The injected frames are
written with the pipeline (inject_signal) and both SFT methods read those same
frames; the whole search then runs on each set of products.

Outputs: results/injections_maps.csv (one row per signal and Tsft) and
results/injections_results.csv (search result of each method per signal).

    python injections_3packs.py --threads 12
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np
from common import O3_DIR, RESULTS, append_csv, available_packs, compare_chunk, concat_strain, random_packs, \
    same_value

from pbh_viterbi.config import DEFAULT_SKY_SEED, FMAX, FMIN, TSFT_VALUES, t_to_merger_for_mchirp
from pbh_viterbi.o3.frames import frame_start_times, injected_frame_label, read_pack_frames, \
    write_injected_framecache
from pbh_viterbi.o3.packs import pack_window
from pbh_viterbi.paths import DEFAULT_NOISE_BACKGROUND, o3_pack_dir
from pbh_viterbi.results import result_row
from pbh_viterbi.search.candidates import search_candidate
from pbh_viterbi.search.power import load_noise_background
from pbh_viterbi.waveform.injection import build_signal_grid, inject_signal, sample_sky

SIGNALS = ((1.124e-3, 5.623e-3), (4.472e-3, 3.162e-2), (7.080e-2, 7.499e-2))  # (Mc, dL), nearest grid point
COMPARED = ("candidate", "nmse", "nsigma", "mass", "tsft", "status")


def grid_index(grid, mchirp, distance):
    from pycbc.conversions import mchirp_from_mass1_mass2

    cost = [abs(np.log(mchirp_from_mass1_mass2(m1, m2) / mchirp)) + abs(np.log(d / distance)) for m1, m2, d in grid]
    return int(np.argmin(cost))


def read_injected_strain(frame_dir, mchirp, distance, coal_time, t_start):
    from pycbc import frame as pycbc_frame
    from pbh_viterbi.config import CHANNEL, FRAME_LENGTH

    segments = []
    for start in frame_start_times(t_start):
        label = injected_frame_label(mchirp, distance, coal_time, start)
        segments.append(pycbc_frame.read_frame(str(Path(frame_dir) / f"{label}.gwf"), CHANNEL,
                                               start_time=start, end_time=start + FRAME_LENGTH))
    return concat_strain(segments)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threads", type=int, default=12, help="Parallel MakeSFTs processes.")
    parser.add_argument("--fmax", type=float, default=FMAX)
    args = parser.parse_args()
    from pycbc.conversions import mchirp_from_mass1_mass2

    grid = build_signal_grid()
    packs = random_packs(len(SIGNALS), available_packs(), stream=3)
    background = load_noise_background(DEFAULT_NOISE_BACKGROUND)
    print(f"Packs: {packs}", flush=True)

    for pack, (mc_target, dl_target) in zip(packs, SIGNALS):
        signal_id = grid_index(grid, mc_target, dl_target)
        m1, m2, distance = grid[signal_id]
        mchirp = mchirp_from_mass1_mass2(m1, m2)
        sky = sample_sky(DEFAULT_SKY_SEED, pack, signal_id)
        t_start, t_end = pack_window(pack)
        print(f"pack {pack}: signal {signal_id}, mchirp={mchirp:.4g}, dL={distance:.4g}, sky={sky}", flush=True)
        label = {"pack": pack, "signal_id": signal_id, "mchirp": mchirp, "distance": distance}

        raw_segments = read_pack_frames(o3_pack_dir(pack, O3_DIR), t_start)
        with tempfile.TemporaryDirectory(prefix=f"cmp-inject-pack{pack}-") as work:
            coal_time, frame_dir = inject_signal(m1, m2, distance, t_to_merger_for_mchirp(mchirp), sky["ra"],
                                                 sky["dec"], sky["pol"], t_start, raw_segments, work)
            cache = write_injected_framecache(f"{work}/framecache", frame_dir, mchirp, distance, coal_time, t_start)
            strain, fs = read_injected_strain(frame_dir, mchirp, distance, coal_time, t_start)
            rows, ref_products, fast_products = compare_chunk(t_start, t_end, cache, strain, fs, TSFT_VALUES,
                                                              args.threads, FMIN, args.fmax, label)
        append_csv(rows, RESULTS / "injections_maps.csv")

        results = {}
        for method, products in (("makesfts", ref_products), ("fast", fast_products)):
            results[method] = result_row(pack, search_candidate(products, *background), injected=True,
                                         mchirp=mchirp, distance=distance, sky=sky)
        row = {**label, **{k: sky[k] for k in ("ra", "dec", "pol")}}
        for key in COMPARED:
            row[f"{key}_makesfts"] = results["makesfts"][key]
            row[f"{key}_fast"] = results["fast"][key]
        row["identical"] = all(same_value(results["makesfts"][k], results["fast"][k]) for k in COMPARED)
        row["maps_identical_all_tsft"] = all(r["map_identical"] for r in rows)
        row["tracks_identical_all_tsft"] = all(r["track_identical"] for r in rows)
        append_csv([row], RESULTS / "injections_results.csv")
        print(", ".join(f"{k}={v}" for k, v in row.items()), flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
