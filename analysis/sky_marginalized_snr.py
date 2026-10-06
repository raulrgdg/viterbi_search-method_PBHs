#!/usr/bin/env python3
"""Sky-marginalised optimal SNR of the injected signals, per O3b pack.

For every pack, the PSD of its own data (mean of the Welch PSDs of its eight
frames) is used, and each signal of the pack's campaign slice is projected
with ``--n-sky`` random sky locations. Two CSVs are written: one row per sky
draw, and one summary row per (pack, signal) with the median, mean, RMS and
quantiles of the SNR over the draws. A third CSV (mchirp, distance,
optimal_snr = median) is the table used to colour Fig. 5.

The paper table (paper_data/snr/injection_snr_sky_median.csv) used one pack
per campaign slice (~12 h):

    python analysis/sky_marginalized_snr.py --packs 73,85,97 --n-sky 5
"""

import argparse
import csv
import logging
import time
from pathlib import Path

import numpy as np

import _setup  # noqa: F401  (adds src/ to sys.path)
from pbh_viterbi.campaign import assignment_for_pack
from pbh_viterbi.config import DEC_RANGE, POL_RANGE, RA_RANGE
from pbh_viterbi.o3.frames import read_pack_frames
from pbh_viterbi.o3.packs import pack_window, parse_pack_list
from pbh_viterbi.paths import O3_DIR, SEARCH_RESULTS_DIR, o3_pack_dir
from pbh_viterbi.snr import frame_polarizations, optimal_snr, welch_psd
from pbh_viterbi.waveform.injection import build_signal_grid

log = logging.getLogger("sky_marginalized_snr")


def pack_psd(pack, o3_dir):
    psds = [welch_psd(strain) for strain in read_pack_frames(o3_pack_dir(pack, o3_dir), pack_window(pack)[0])]
    return np.asarray(psds[0].sample_frequencies), np.mean(np.vstack([np.asarray(p) for p in psds]), axis=0)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--packs", required=True, help='e.g. "73" or "13-24".')
    parser.add_argument("--n-sky", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--max-signals", type=int, default=None, help="Only the first N signals of each slice.")
    parser.add_argument("--o3-dir", type=Path, default=O3_DIR)
    parser.add_argument("--output-prefix", type=Path, default=SEARCH_RESULTS_DIR / "pack_sky_snr")
    args = parser.parse_args()

    from pycbc.conversions import mchirp_from_mass1_mass2

    rng = np.random.default_rng(args.seed)
    grid = build_signal_grid()
    draws, summary = [], []
    for pack in parse_pack_list(args.packs):
        psd_freqs, psd_values = pack_psd(pack, args.o3_dir)
        assignment = assignment_for_pack(pack)
        t_start = pack_window(pack)[0]
        signal_ids = range(assignment.signal_start, assignment.signal_end)
        if args.max_signals is not None:
            signal_ids = list(signal_ids)[: args.max_signals]

        for signal_id in signal_ids:
            tic = time.time()
            m1, m2, distance = grid[signal_id]
            mchirp = float(mchirp_from_mass1_mass2(m1, m2))
            polarizations = frame_polarizations(m1, m2, distance, t_start)
            snrs = []
            for sky_index in range(args.n_sky):
                ra, dec, pol = rng.uniform(*RA_RANGE), rng.uniform(*DEC_RANGE), rng.uniform(*POL_RANGE)
                snr = optimal_snr(polarizations, ra, dec, pol, psd_freqs, psd_values)
                snrs.append(snr)
                draws.append({"pack": pack, "cluster": assignment.cluster, "signal_index": signal_id,
                              "mchirp": mchirp, "distance": distance, "sky_index": sky_index,
                              "ra": float(ra), "dec": float(dec), "pol": float(pol), "optimal_snr_pack_sky": snr})
            snrs = np.asarray(snrs)
            summary.append({
                "pack": pack, "cluster": assignment.cluster, "signal_index": signal_id, "mchirp": mchirp,
                "distance": distance, "n_sky": args.n_sky,
                "snr_median_pack_sky": float(np.median(snrs)),
                "snr_mean_pack_sky": float(np.mean(snrs)),
                "snr_rms_pack_sky": float(np.sqrt(np.mean(snrs ** 2))),
                "snr_std_pack_sky": float(np.std(snrs)),
                **{f"snr_p{int(q * 100):02d}_pack_sky": float(np.quantile(snrs, q)) for q in (0.05, 0.16, 0.84, 0.95)},
                "snr_min_pack_sky": float(np.min(snrs)),
                "snr_max_pack_sky": float(np.max(snrs)),
                "runtime_seconds": time.time() - tic,
            })
            log.info("pack %d signal %d: median SNR %.4g", pack, signal_id, summary[-1]["snr_median_pack_sky"])

    write_csv(Path(f"{args.output_prefix}_draws.csv"), draws)
    write_csv(Path(f"{args.output_prefix}_summary.csv"), summary)
    write_csv(Path(f"{args.output_prefix}_grid.csv"),
              [{"mchirp": f"{r['mchirp']:.8g}", "distance": f"{r['distance']:.8g}",
                "optimal_snr": f"{r['snr_median_pack_sky']:.6g}"} for r in summary])
    log.info("Written %s_{draws,summary,grid}.csv", args.output_prefix)


if __name__ == "__main__":
    main()
