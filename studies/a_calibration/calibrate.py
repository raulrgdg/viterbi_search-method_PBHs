#!/usr/bin/env python3
"""Calibration of a, the weight of -log10(NMSE_eff) in R_a = nsigma - a log10(NMSE_eff).

a enters both the hybrid Tsft selection (largest R_a among the isolated Tsft)
and the detection threshold (R_a >= t). It is chosen on a calibration set that
does not overlap the campaign: the injections of 10 packs outside it
(submit_condor.sh) and the noise search of the 141 packs.

For every a of the grid, each chunk's Tsft is re-selected from its per-Tsft
products, t is set on the noise at the target FAR (at most floor(FAR N) noise
packs above) and the efficiency is the fraction of calibration injections
above t. Cross-validation on the noise: in K folds of noise packs, t is set on
K-1 folds and the false alarms are counted on the held-out one, so the pooled
out-of-fold FAR (Clopper-Pearson interval) shows whether the threshold of each a
holds on noise it was not set on. a is the value of largest efficiency at the
target FAR (ties: the smallest a); the efficiency must be flat around it.

    python studies/a_calibration/calibrate.py --noise-per-tsft <per_tsft_noise.csv> \\
        --injections <dir with per_tsft_injected_pack-*.csv> [--far 0.01] [--out results/a_calibration]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pbh_viterbi.search import selection  # noqa: E402
from pbh_viterbi.search.ranking import clopper_pearson, pack_folds  # noqa: E402

A_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0, np.inf)
SLICES = {"light": (0, 200), "intermediate": (200, 400), "heavy": (400, 600)}
N_FOLDS = 3


def threshold(noise_stat, far):
    """t with at most floor(far N) noise values >= t."""
    values = np.sort(noise_stat)[::-1]
    k = int(np.floor(far * len(values) + 1e-9))
    return np.nextafter(values[k], np.inf) if np.isfinite(values[k]) else -np.inf


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--noise-per-tsft", type=Path, required=True)
    parser.add_argument("--injections", type=Path, required=True,
                        help="Folder with per_tsft_injected_pack-*.csv and search_results_injected_pack-*.csv")
    parser.add_argument("--far", type=float, default=0.01)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "a_calibration")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    noise = selection.prepare(pd.read_csv(args.noise_per_tsft))
    frames = [pd.read_csv(p) for p in sorted(args.injections.glob("per_tsft_injected_pack-*.csv"))]
    inj = selection.prepare(pd.concat(frames, ignore_index=True))
    n_noise, packs = noise["pack"].nunique(), sorted(inj["pack"].unique())
    n_inj = inj.groupby(["pack", "signal_id"]).ngroups
    folds = pack_folds(noise["pack"].unique(), N_FOLDS, seed=1)
    print(f"{n_noise} noise packs, {n_inj} calibration injections in packs {packs}, FAR target {args.far:.0%}")

    rows = []
    for a in A_GRID:
        sn = selection.select_by_ranking(noise, a).sort_values("pack")
        si = selection.select_by_ranking(inj, a)
        zn, zi = sn["stat"].to_numpy(), si["stat"].to_numpy()
        t = threshold(zn, args.far)
        det = zi >= t
        oof_false = 0
        for fold in range(N_FOLDS):
            test = sn["pack"].map(folds).to_numpy() == fold
            oof_false += int(np.sum(zn[test] >= threshold(zn[~test], args.far)))
        lo, hi = clopper_pearson(oof_false, n_noise, 0.68)
        row = dict(a=a, t=t, efficiency=det.mean(), noise_above=int(np.sum(zn >= t)),
                   oof_noise_above=oof_false, oof_far=oof_false / n_noise, oof_far_lo=lo, oof_far_hi=hi)
        for name, (s0, s1) in SLICES.items():
            m = ((si["signal_id"] >= s0) & (si["signal_id"] < s1)).to_numpy()
            row[f"eff_{name}"] = det[m].mean() if m.any() else np.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(args.out / f"calibration_far{args.far:g}.csv", index=False)
    best = table.loc[table["efficiency"].idxmax()]
    plateau = table[table["efficiency"] >= best["efficiency"] - 0.01]["a"]
    result = {"a": float(best["a"]), "far_target": args.far, "efficiency": float(best["efficiency"]),
              "plateau_within_1pct": [float(plateau.min()), float(plateau.max())],
              "noise_packs": int(n_noise), "calibration_packs": [int(p) for p in packs],
              "calibration_injections": int(n_inj)}
    (args.out / f"a_far{args.far:g}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(table.round(4).to_string(index=False))
    print(json.dumps(result, indent=2))

    finite = table[np.isfinite(table["a"])]
    fig, ax = plt.subplots(figsize=(7, 4))
    for col, color, label in (("efficiency", "#008300", "all"), ("eff_light", "#2a78d6", "light"),
                              ("eff_intermediate", "#eda100", "intermediate"), ("eff_heavy", "#e34948", "heavy")):
        ax.plot(finite["a"], finite[col], marker="o", ms=4, color=color, lw=2 if col == "efficiency" else 1.2,
                label=label)
    ax.axvline(best["a"], color="#6b6a64", ls=":", lw=1)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("a  (R_a = nσ − a log10 NMSE_eff)")
    ax.set_ylabel(f"efficiency at FAR {args.far:.0%}")
    ax.set_title(f"Calibration of a: {n_inj} injections in {len(packs)} packs, {n_noise} noise packs")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, color="#e4e3de")
    fig.savefig(args.out / f"fig_calibration_far{args.far:g}.png", dpi=160, bbox_inches="tight")


if __name__ == "__main__":
    main()
