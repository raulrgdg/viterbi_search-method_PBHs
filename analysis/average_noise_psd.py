#!/usr/bin/env python3
"""Average noise PSD of the O3b H1 data (Appendix A, Fig. 8).

Every downloaded frame is one noise realisation; its PSD is estimated with
Welch's method (512 s Hann segments, 50% overlap, median average) and the
realisation PSDs are averaged, Eq. (A1). The result is used for every SNR
in the paper.

    python analysis/average_noise_psd.py                 # all packs in data/o3
    python analysis/average_noise_psd.py --packs 1-12
"""

import argparse
import logging
from pathlib import Path

import numpy as np

import _setup  # noqa: F401  (adds src/ to sys.path)
from pbh_viterbi.o3.frames import read_pack_frames
from pbh_viterbi.o3.packs import pack_window, parse_pack_list
from pbh_viterbi.paths import O3_DIR, SEARCH_RESULTS_DIR, o3_pack_dir
from pbh_viterbi.snr import welch_psd

log = logging.getLogger("average_noise_psd")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--packs", default="all")
    parser.add_argument("--o3-dir", type=Path, default=O3_DIR)
    parser.add_argument("--fmin", type=float, default=10.0, help="Lowest frequency kept in the output (Hz).")
    parser.add_argument("--fmax", type=float, default=256.0, help="Highest frequency kept in the output (Hz).")
    parser.add_argument("--output", type=Path, default=SEARCH_RESULTS_DIR / "average_noise_psd.csv")
    args = parser.parse_args()

    packs = [p for p in parse_pack_list(args.packs) if o3_pack_dir(p, args.o3_dir).is_dir()]
    if not packs:
        raise SystemExit(f"No downloaded packs found in {args.o3_dir}")

    freqs, psds = None, []
    for pack in packs:
        log.info("Pack %d", pack)
        for strain in read_pack_frames(o3_pack_dir(pack, args.o3_dir), pack_window(pack)[0]):
            psd = welch_psd(strain)
            if freqs is None:
                freqs = np.asarray(psd.sample_frequencies, dtype=float)
            elif not np.allclose(freqs, psd.sample_frequencies):
                raise ValueError("All PSDs must share the same frequency grid.")
            psds.append(np.asarray(psd, dtype=float))

    average = np.mean(np.vstack(psds), axis=0)
    keep = (freqs >= args.fmin) & (freqs <= args.fmax)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output, np.column_stack([freqs[keep], average[keep]]), delimiter=",",
               header="frequency_hz,average_psd", comments="")
    log.info("Average of %d realisations from %d packs written to %s", len(psds), len(packs), args.output)


if __name__ == "__main__":
    main()
