#!/usr/bin/env python3
"""[UNDER TEST] Why v1.2 recovers fewer signals than the paper at 1.1-1.6e-3 Msun.

Four checks, on the v1.2 campaign and the paper campaign (paper_data):

1. recovered fraction and median nsigma per distance, v1.2 (linear threshold)
   vs paper (polynomial threshold), for the masses around the gap;
2. per-Tsft profile at the frontier distances (median nsigma, NMSE_eff, R_a and
   the fraction passing the threshold per Tsft, and the Tsft chosen by the
   hybrid selection), and whether *any* Tsft would pass the threshold;
3. trials factor: the threshold t and the recovered fractions when noise and
   injections are re-selected with fewer Tsft;
4. the paper polynomial applied to the v1.2 triggers (injections and noise).

    python analysis/low_mass_gap.py --campaign-dir results/search/campaign_v12 \\
        --out results/search/campaign_v12/low_mass_gap

Needs the per-Tsft products of the campaign (per_tsft/*.csv, per_tsft_noise.csv).
Writes the tables as CSV in --out and prints them. See analysis/low_mass_gap.md.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _setup  # noqa: F401  (adds src/ to sys.path)
from pbh_viterbi.config import TSFT_LISTS
from pbh_viterbi.paths import PAPER_DATA_DIR
from pbh_viterbi.search import selection as S
from pbh_viterbi.tsft import tsft_lim

PAPER_CSV = PAPER_DATA_DIR / "search_results" / "injected_search_O3b_H1_90packs.csv"
PAPER_NOISE = PAPER_DATA_DIR / "search_results" / "noise_search_O3b_H1_108packs.csv"
PAPER_THRESHOLD = PAPER_DATA_DIR / "calibration" / "far_threshold.json"
MASSES = (7.96e-4, 1.12e-3, 1.59e-3, 2.24e-3)
FRONTIER = {1.12e-3: (3.16e-3, 5.62e-3, 1e-2), 1.59e-3: (5.62e-3, 1e-2, 1.155e-2, 1.334e-2, 1.54e-2)}
CONTROL = (2e-4, 5.63e-4, 0.0355)


def near(series, value):
    return np.isclose(series, value, rtol=1e-2)


def paper_pass(coeffs, nsigma, nmse):
    nmse = pd.to_numeric(nmse, errors="coerce").where(lambda x: x > 0)
    return (nsigma >= np.polyval(coeffs, np.log10(nmse))).fillna(False)


def per_tsft_with_R(df, a):
    df = S.prepare(df)
    df["nmse_eff"] = S.nmse_eff(df)
    df["R"] = np.where(df["isolated"], df["nsigma"] - a * np.log10(df["nmse_eff"]), np.nan)
    return df


def threshold_on_noise(noise_R, far):
    """Smallest t with at most floor(FAR N) noise packs at or above it (as compute_threshold_linear)."""
    values = np.sort(np.nan_to_num(noise_R, nan=-np.inf))[::-1]
    k = int(np.floor(far * len(values) + 1e-9))
    return float(np.nextafter(values[k], np.inf)) if k < len(values) else -np.inf


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign-dir", type=Path, default=Path("results/search/campaign_v12"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    d = args.campaign_dir
    out = args.out or d / "low_mass_gap"
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)

    th3 = json.load(open(d / "threshold_far3.json"))
    a, t3 = th3["a"], th3["t"]
    coeffs = json.load(open(PAPER_THRESHOLD))["coefficients"]
    camp = pd.read_csv(d / "injected_campaign_snr.csv")
    camp["rec"] = camp["ranking_stat"] >= t3
    paper = pd.read_csv(PAPER_CSV)
    paper["rec"] = paper_pass(coeffs, paper["nsigma"], paper["nmse"])

    # 1. recovered fraction and nsigma per distance
    rows = []
    for m in MASSES:
        for dist, g in camp[near(camp["mchirp"], m)].groupby("distance"):
            p = paper[near(paper["mchirp"], m) & near(paper["distance"], dist)]
            rows.append({"mchirp": m, "d_kpc": dist * 1e3, "rec_v12": g["rec"].mean(), "rec_paper": p["rec"].mean(),
                         "nsigma_v12": g["nsigma"].median(), "nsigma_paper": p["nsigma"].median(),
                         "tsft_v12": g["tsft"].median(), "snr": g["optimal_snr"].median()})
    by_distance = pd.DataFrame(rows)
    by_distance.to_csv(out / "recovered_by_distance.csv", index=False)
    print("1. Recovered fraction per distance (v1.2 FAR 3% vs paper)")
    for m in MASSES:
        print(f"   Mc = {m:g}, T_lim = {float(tsft_lim(m)):.1f} s")
        print(by_distance[near(by_distance["mchirp"], m) & (by_distance["d_kpc"] <= 25)].round(2).to_string(index=False))

    # 2. per-Tsft profile at the frontier
    pt = pd.concat([pd.read_csv(f) for f in sorted((d / "per_tsft").glob("*.csv"))], ignore_index=True)
    pt = per_tsft_with_R(pt, a)
    meta = camp[["pack", "signal_id", "mchirp", "distance", "tsft", "rec"]].rename(columns={"tsft": "tsft_sel"})
    pt = pt.merge(meta, on=["pack", "signal_id"])
    print("\n2. Per-Tsft profile at the frontier distances (Tsft 20-240 s)")
    profiles, any_rows = [], []
    for m, distances in FRONTIER.items():
        for dist in distances:
            q = pt[near(pt["mchirp"], m) & near(pt["distance"], dist)]
            chunks = q.drop_duplicates(["pack", "signal_id"])
            prof = q[q["tsft"].between(20, 240)].groupby("tsft").agg(
                nsigma=("nsigma", "median"), log_nmse_eff=("nmse_eff", lambda x: np.nanmedian(np.log10(x))),
                R=("R", "median"), passes=("R", lambda x: (x >= t3).mean()))
            prof["chosen"] = chunks["tsft_sel"].value_counts().reindex(prof.index).fillna(0).astype(int)
            any_pass = (q.groupby(["pack", "signal_id"])["R"].max() >= t3).mean()
            any_rows.append({"mchirp": m, "d_kpc": dist * 1e3, "recovered": chunks["rec"].mean(), "any_tsft_passes": any_pass})
            profiles.append(prof.reset_index().assign(mchirp=m, d_kpc=dist * 1e3))
            print(f"   Mc = {m:g}, d = {dist * 1e3:.2f} kpc: recovered {chunks['rec'].mean():.2f}, any Tsft passes {any_pass:.2f}")
            print(prof.round(2).to_string())
    pd.concat(profiles).to_csv(out / "per_tsft_profile.csv", index=False)
    pd.DataFrame(any_rows).to_csv(out / "recovered_vs_any_tsft.csv", index=False)

    # 3. trials factor: fewer Tsft
    noise = per_tsft_with_R(pd.read_csv(d / "per_tsft_noise.csv"), a)
    full = np.array(TSFT_LISTS["95"])
    subsets = {"45 (list 95)": full, "23 (every 2nd)": full[::2], "15 (every 3rd)": full[::3],
               "32 (<= 88 s)": full[full <= 88.2]}
    chunk_mass = pt.drop_duplicates(["pack", "signal_id"])[["pack", "signal_id", "mchirp"]]
    rows = []
    for name, tsfts in subsets.items():
        noise_R = noise[noise["tsft"].isin(tsfts)].groupby("pack")["R"].max().to_numpy()
        sig = pt[pt["tsft"].isin(tsfts)].groupby(["pack", "signal_id"])["R"].max().fillna(-np.inf).rename("R")
        sig = sig.reset_index().merge(chunk_mass)
        for far in (0.03, 0.01):
            t = threshold_on_noise(noise_R, far)
            rec = sig["R"] >= t
            row = {"tsft_set": name, "far": far, "t": t, "total": rec.mean()}
            for m in (1.12e-3, 1.59e-3) + CONTROL:
                row[f"{m:g}"] = rec[near(sig["mchirp"], m)].mean()
            rows.append(row)
    trials = pd.DataFrame(rows)
    trials.to_csv(out / "trials_factor.csv", index=False)
    print("\n3. Re-selection with fewer Tsft (threshold re-set on noise)")
    print(trials.round(3).to_string(index=False))

    # 4. paper polynomial on the v1.2 triggers
    noise_v12 = pd.read_csv(d / "search_results_noise.csv")
    noise_paper = pd.read_csv(PAPER_NOISE)
    ok = noise_paper["nmse"] > 0
    low = np.log10(noise_paper.loc[ok, "nmse"]) < -2.2
    camp["paper_poly"] = paper_pass(coeffs, camp["nsigma"], camp["nmse"])
    print("\n4. Paper polynomial")
    print("   nsigma threshold vs log10 NMSE:", {x: round(float(np.polyval(coeffs, x)), 2) for x in (-1, -2, -2.2, -2.5, -3)})
    print(f"   paper noise with log10 NMSE < -2.2: {int(low.sum())} of {int(ok.sum())}")
    print(f"   v1.2 noise packs above it: {int(paper_pass(coeffs, noise_v12['nsigma'], noise_v12['nmse']).sum())}/{len(noise_v12)}"
          f" (raw NMSE), {int(paper_pass(coeffs, noise_v12['nsigma'], noise_v12['nmse_eff']).sum())}/{len(noise_v12)} (NMSE_eff)")
    rows = []
    for m, distances in FRONTIER.items():
        for dist in distances:
            p = paper[near(paper["mchirp"], m) & near(paper["distance"], dist)]
            c = camp[near(camp["mchirp"], m) & near(camp["distance"], dist)]
            pr = p[p["rec"]]
            rows.append({"mchirp": f"{m:g}", "d_kpc": dist * 1e3, "rec_paper": p["rec"].mean(),
                         "paper_rec_nsigma_lt_1.4": (pr["nsigma"] < 1.4).mean(),
                         "paper_rec_median_log_nmse": np.log10(pr["nmse"]).median(),
                         "rec_v12_linear": c["rec"].mean(), "rec_v12_paper_poly": c["paper_poly"].mean()})
    poly = pd.DataFrame(rows)
    poly.to_csv(out / "paper_polynomial_on_v12.csv", index=False)
    print(poly.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
