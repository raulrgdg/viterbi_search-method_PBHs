#!/usr/bin/env python3
"""Dense-Tsft check of nsigma / max nsigma = sqrt(Tsft / T_ref) near T_lim.

Tsft list: the smallest set of multiples of 1/8 s (integer samples at 512 Hz)
giving nsigma / max >= 0.95 for every chirp mass in [2e-4, 0.1] Msun according
to tsft_list_remapped.py (45 values, 2.6-422 s). The noise background (mean
and std of the Viterbi track power) is measured for every Tsft on the 108 O3b
packs; then 3 chirp masses (the lightest, middle and heaviest of
../injection_check) are injected at 5 distances x 3 random packs.

    python run.py background --n-jobs 8 --job-id k
    python run.py stats
    python run.py inject --n-jobs 8 --job-id k
"""

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2] / "src"))
from plot_tsft_lim import tsft_lim  # noqa: E402
from tsft_list_remapped import optimal_list  # noqa: E402

from pbh_viterbi.config import DEFAULT_SKY_SEED, FMAX, FMIN, t_to_merger_for_mchirp  # noqa: E402
from pbh_viterbi.o3.frames import read_pack_frames  # noqa: E402
from pbh_viterbi.o3.packs import pack_window  # noqa: E402
from pbh_viterbi.paths import o3_pack_dir  # noqa: E402
from pbh_viterbi.search.power import total_track_power  # noqa: E402
from pbh_viterbi.waveform.injection import equal_component_masses, inject_in_memory, sample_sky  # noqa: E402
from pbh_viterbi.workflows.common import fast_products  # noqa: E402


def read_pack(pack):
    t_start = pack_window(pack)[0]
    return read_pack_frames(o3_pack_dir(pack), t_start), t_start


def products(segments, tsfts):
    return fast_products(segments, tsfts, FMIN, FMAX, sft_highpass="segment", normalisation="bin")  # v1.1-v1.2 maps


def append_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if new:
            writer.writeheader()
        writer.writerows(rows)


def done_keys(path, keys):
    try:
        with open(path) as handle:
            return {tuple(r[k] for k in keys) for r in csv.DictReader(handle)}
    except FileNotFoundError:
        return set()

RESULTS = HERE / "results"
BACKGROUND = RESULTS / "noise_background.csv"
MIN_NORMED, MCHIRP_RANGE = 0.95, (2e-4, 0.1)
# (chirp mass Msun, d50 Mpc), as in ../injection_check
SIGNALS = ((5.63e-4, 1.0e-3), (3.166e-3, 2.7e-2), (5.013e-2, 1.0e-1))
N_DIST, N_PACKS, SEED = 5, 3, 20261015


def tsft_list():
    mchirps = np.geomspace(*MCHIRP_RANGE, 600)
    lo, hi = float(tsft_lim(MCHIRP_RANGE[1], FMIN, FMAX)), float(tsft_lim(MCHIRP_RANGE[0], FMIN, FMAX))
    candidates = np.unique(np.round(np.geomspace(0.8 * lo, hi, 2000) * 8) / 8)
    return [float(t) for t in optimal_list(mchirps, candidates, FMIN, FMAX, MIN_NORMED, 1.03)]


def track_powers(prods):
    return {float(p["tsft"]): float(total_track_power(p["track_index"], p["power"])) for p in prods}


def background(n_jobs, job_id):
    tsfts = tsft_list()
    output = RESULTS / f"noise_power.job{job_id}.csv"
    done = {int(k[0]) for k in done_keys(output, ["pack"])}
    for pack in list(range(1, 109))[job_id::n_jobs]:
        if pack in done:
            continue
        powers = track_powers(products(read_pack(pack)[0], tsfts))
        append_csv([{"pack": pack, "tsft": t, "total_power": p} for t, p in powers.items()], output)
        print(f"pack {pack} done", flush=True)


def stats():
    values = defaultdict(list)
    for path in sorted(RESULTS.glob("noise_power.job*.csv")):
        with path.open() as handle:
            for row in csv.DictReader(handle):
                values[float(row["tsft"])].append(float(row["total_power"]))
    BACKGROUND.unlink(missing_ok=True)
    append_csv([{"tsft": t, "mean_total_power": statistics.mean(v), "std_total_power": statistics.stdev(v),
                 "n_packs": len(v)} for t, v in sorted(values.items())], BACKGROUND)
    print(f"{len(values)} Tsft, packs per Tsft: {sorted({len(v) for v in values.values()})}")


def injections():
    rng = np.random.default_rng(SEED)
    return [(i, j, mc, float(dist), int(pack))
            for i, (mc, d50) in enumerate(SIGNALS)
            for j, dist in enumerate(np.geomspace(d50 / 30, d50, N_DIST))
            for pack in rng.choice(np.arange(1, 109), N_PACKS, replace=False)]


def inject(n_jobs, job_id):
    from pycbc.conversions import mchirp_from_mass1_mass2

    tsfts = tsft_list()
    with BACKGROUND.open() as handle:
        rows = list(csv.DictReader(handle))
    mean = {float(r["tsft"]): float(r["mean_total_power"]) for r in rows}
    std = {float(r["tsft"]): float(r["std_total_power"]) for r in rows}
    output = RESULTS / f"injections.job{job_id}.csv"
    done = done_keys(output, ["i_mass", "i_dist", "pack"])
    for i, j, mc, dist, pack in injections()[job_id::n_jobs]:
        if (str(i), str(j), str(pack)) in done:
            continue
        m1, m2 = equal_component_masses(mc)
        mchirp = mchirp_from_mass1_mass2(m1, m2)
        sky = sample_sky(DEFAULT_SKY_SEED, pack, 1000 * i + j)
        raw, t_start = read_pack(pack)
        _, segments = inject_in_memory(m1, m2, dist, t_to_merger_for_mchirp(mchirp), sky["ra"], sky["dec"],
                                       sky["pol"], t_start, raw)
        powers = track_powers(products(segments, tsfts))
        append_csv([{"i_mass": i, "i_dist": j, "pack": pack, "mchirp": mchirp, "distance": dist,
                     **{f"nsigma_tsft{t:g}": (p - mean[t]) / std[t] for t, p in powers.items()}}], output)
        print(f"Mc={mchirp:.4g} dL={dist:.3g} pack {pack} done", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("list", "background", "stats", "inject"))
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--job-id", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "list":
        print(", ".join(f"{t:g}" for t in tsft_list()))
    elif args.mode == "background":
        background(args.n_jobs, args.job_id)
    elif args.mode == "stats":
        stats()
    else:
        inject(args.n_jobs, args.job_id)


if __name__ == "__main__":
    main()
