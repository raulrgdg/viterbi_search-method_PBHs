#!/usr/bin/env python3
"""Optimal SNR of every injection of an injected-search result file (Appendix A).

The injected search writes, for each injection, the parameters needed to
regenerate it: pack, chirp mass, distance, sky location (ra, dec), polarisation,
inclination and time to merger. This script recomputes the projected waveform
and its optimal SNR, Eqs. (A3)-(A5), in the search band, using the PSD of the
pack's own data (mean of the Welch PSDs of its eight frames). The waveform of
each chirp mass is generated once per pack at 1 Mpc and scaled as 1/distance.

Output: the input rows with an extra column ``optimal_snr``.

    python analysis/injection_snr.py results/search/injected_campaign.csv \
        --output results/search/injected_campaign_snr.csv
    # split by packs over several jobs, then merge with analysis/merge_results.py
    python analysis/injection_snr.py results/search/injected_campaign.csv --packs 1-30 \
        --output results/search/snr/injected_snr_packs1-30.csv
"""

import argparse
import csv
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import _setup  # noqa: F401  (adds src/ to sys.path)
from pbh_viterbi.config import FMAX, FMIN
from pbh_viterbi.o3.frames import read_pack_frames
from pbh_viterbi.o3.packs import pack_window, parse_pack_list
from pbh_viterbi.paths import O3_DIR, o3_pack_dir
from pbh_viterbi.snr import frame_polarizations, optimal_snr, welch_psd
from pbh_viterbi.waveform.injection import equal_component_masses

log = logging.getLogger("injection_snr")
REFERENCE_DISTANCE = 1.0  # Mpc


def pack_psd(pack, o3_dir):
    psds = [welch_psd(strain) for strain in read_pack_frames(o3_pack_dir(pack, o3_dir), pack_window(pack)[0])]
    return np.asarray(psds[0].sample_frequencies), np.mean(np.vstack([np.asarray(p) for p in psds]), axis=0)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_csv", type=Path, help="Injected-search results (columns of pbh_viterbi.results).")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--packs", default=None, help='Only these packs, e.g. "1-30" (default: all in the file).')
    parser.add_argument("--o3-dir", type=Path, default=O3_DIR)
    parser.add_argument("--fmin", type=float, default=FMIN)
    parser.add_argument("--fmax", type=float, default=FMAX)
    args = parser.parse_args()

    with args.input_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames) + ["optimal_snr"]
        rows = [r for r in reader if r.get("injected") in ("1", 1)]
    required = {"pack", "mchirp", "distance", "ra", "dec", "pol", "inclination", "t_to_merger"}
    if missing := required - set(fields):
        raise SystemExit(f"{args.input_csv} lacks the columns {sorted(missing)} (produced by pipeline v1.1).")

    by_pack = defaultdict(list)
    for row in rows:
        by_pack[int(row["pack"])].append(row)
    packs = sorted(by_pack) if args.packs is None else [p for p in parse_pack_list(args.packs) if p in by_pack]

    out = []
    for pack in packs:
        tic = time.time()
        psd_freqs, psd_values = pack_psd(pack, args.o3_dir)
        t_start = pack_window(pack)[0]
        waveforms = {}
        for row in by_pack[pack]:
            mchirp, inclination, t_to_merger = float(row["mchirp"]), float(row["inclination"]), float(row["t_to_merger"])
            key = (round(mchirp, 12), inclination, t_to_merger)
            if key not in waveforms:
                m1, m2 = equal_component_masses(mchirp)
                waveforms[key] = frame_polarizations(m1, m2, REFERENCE_DISTANCE, t_start, inclination=inclination,
                                                     t_to_merger=t_to_merger)
            snr = optimal_snr(waveforms[key], float(row["ra"]), float(row["dec"]), float(row["pol"]), psd_freqs,
                              psd_values, args.fmin, args.fmax)
            out.append({**row, "optimal_snr": snr * REFERENCE_DISTANCE / float(row["distance"])})
        log.info("pack %d: %d injections in %.0f s", pack, len(by_pack[pack]), time.time() - tic)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out)
    log.info("%d rows written to %s", len(out), args.output)


if __name__ == "__main__":
    main()
