#!/usr/bin/env python3
"""Step 4: experimental check of the optimal-Tsft condition in the remapped map.

In x = f^-8/3 the signal moves at constant speed K = (256/5) pi^(8/3) (G Mc/c^3)^(5/3),
and Viterbi can follow it only if it moves at most one remapped bin per SFT:

    K * B * Tsft^2 <= f0^(-8/3) - f1^(-8/3),   B = f1 - f0
    -> T_lim = sqrt((f0^(-8/3) - f1^(-8/3)) / (K B)) ~= 17.7 s (Mc / 1e-2 Msun)^(-5/6)

Prediction: the selected Tsft is the largest Tsft of the grid <= T_lim.

The 10 grid chirp masses >= 4.5e-3 Msun except 0.085 (step 2), 10 distances
each (1.8e-3 to 2.4e-2 Mpc, loud enough for a clear track), each injection in
a random pack (fixed seed), 13 paper Tsft, band 61.1-126.8 Hz.

    python step4_tsft_formula.py --n-jobs 8 --job-id k   # job k -> results/tsft_formula.job<k>.csv
    python step4_tsft_formula.py --summary               # results/tsft_formula.csv, tsft_formula_summary.csv
"""

import argparse

import numpy as np
from common import PAPER_BACKGROUND, RESULTS, TSFT_13, append_csv, concat, done_keys, inject_in_memory, \
    load_background, products, read_pack, track_powers

from pbh_viterbi.config import DEFAULT_SKY_SEED, DISTANCE_GRID, FMAX, FMIN, MCHIRP_GRID, t_to_merger_for_mchirp
from pbh_viterbi.search.candidates import search_candidate
from pbh_viterbi.waveform.injection import build_signal_grid, sample_sky

T_SUN = 1.32712440018e20 / 299792458**3
MCHIRPS = [mc for mc in MCHIRP_GRID if mc >= 4.4e-3 and not np.isclose(mc, 0.085)]
DISTANCE_INDICES = range(5, 15)  # 1.78e-3 ... 2.37e-2 Mpc
PACK_SEED = 20261013


def t_lim(mchirp, fmin=FMIN, fmax=FMAX):
    k = 256 / 5 * np.pi ** (8 / 3) * (T_SUN * mchirp) ** (5 / 3)
    return float(np.sqrt((fmin ** (-8 / 3) - fmax ** (-8 / 3)) / (k * (fmax - fmin))))


def predicted_tsft(mchirp):
    return max(t for t in TSFT_13 if t <= t_lim(mchirp))


def injections():
    """(mchirp index in the grid, distance index, pack) of the 100 injections."""
    rng = np.random.default_rng(PACK_SEED)
    out = []
    for mc in MCHIRPS:
        i_mc = int(np.argmin(np.abs(MCHIRP_GRID - mc)))
        for i_d in DISTANCE_INDICES:
            out.append((i_mc, i_d, int(rng.integers(1, 109))))
    return out


def run(n_jobs, job_id):
    from pycbc.conversions import mchirp_from_mass1_mass2

    mean, std = load_background(PAPER_BACKGROUND)
    grid = build_signal_grid()
    output = RESULTS / f"tsft_formula.job{job_id}.csv"
    done = done_keys(output, ["signal_id", "pack"])
    for i_mc, i_d, pack in injections()[job_id::n_jobs]:
        signal_id = i_mc * len(DISTANCE_GRID) + i_d
        if (str(signal_id), str(pack)) in done:
            continue
        m1, m2, distance = grid[signal_id]
        mchirp = mchirp_from_mass1_mass2(m1, m2)
        sky = sample_sky(DEFAULT_SKY_SEED, pack, signal_id)
        raw, t_start = read_pack(pack)
        strain, fs = concat(inject_in_memory(raw, t_start, m1, m2, distance, t_to_merger_for_mchirp(mchirp),
                                             sky["ra"], sky["dec"], sky["pol"]))
        prods = products(strain, fs, TSFT_13)
        result = search_candidate(prods, mean, std)
        nsigma = {t: (p - mean[t]) / std[t] for t, p in track_powers(prods).items()}
        row = {"signal_id": signal_id, "pack": pack, "mchirp": mchirp, "distance": distance,
               "t_lim": t_lim(mchirp), "tsft_predicted": predicted_tsft(mchirp), "tsft_selected": result["tsft"],
               "match": int(result["tsft"]) == predicted_tsft(mchirp), "nsigma": result["nsigma"],
               "status": result["status"], **{f"nsigma_tsft{t}": v for t, v in nsigma.items()}}
        append_csv([row], output)
        print(f"Mc={mchirp:.4g} dL={distance:.4g} pack {pack}: predicted {row['tsft_predicted']} s, "
              f"selected {row['tsft_selected']} s, nsigma={result['nsigma']:.2f}", flush=True)
    print("DONE", flush=True)


def summary():
    import pandas as pd

    df = pd.concat([pd.read_csv(p) for p in sorted(RESULTS.glob("tsft_formula.job*.csv"))], ignore_index=True)
    df = df.sort_values(["mchirp", "distance"])
    df.to_csv(RESULTS / "tsft_formula.csv", index=False)
    rows = []
    for mc, g in df.groupby("mchirp"):
        counts = g.tsft_selected.value_counts().sort_index()
        # Median nsigma of each Tsft relative to the predicted one.
        pred = int(g.tsft_predicted.iloc[0])
        rel = {t: (g[f"nsigma_tsft{t}"] / g[f"nsigma_tsft{pred}"]).median() for t in TSFT_13}
        neighbours = [t for t in TSFT_13 if abs(TSFT_13.index(t) - TSFT_13.index(pred)) <= 1]
        rows.append({"mchirp": mc, "t_lim": g.t_lim.iloc[0], "tsft_predicted": pred,
                     "n_match": int(g.match.sum()), "n": len(g),
                     "selected": "; ".join(f"{t} s x{n}" for t, n in counts.items()),
                     **{f"rel_nsigma_tsft{t}": round(rel[t], 3) for t in neighbours}})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "tsft_formula_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.to_string(index=False))
    print(f"\nSelected Tsft = predicted in {int(df.match.sum())}/{len(df)} injections")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--job-id", type=int, default=0)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    summary() if args.summary else run(args.n_jobs, args.job_id)


if __name__ == "__main__":
    main()
