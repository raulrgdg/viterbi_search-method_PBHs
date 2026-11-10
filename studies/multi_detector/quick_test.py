#!/usr/bin/env python3
"""Quick efficiency test of the v1.3 map options (studies/multi_detector/README.md).

Configurations and inputs (per-Tsft rows, so the Tsft is re-selected offline for any a):
  A   H1, v1.2 maps (per-segment high-pass, 'bin')   noise: campaign v1.2 per_tsft_noise.csv, 61 L1 packs
  B   H1, chunk high-pass, 'bin'                     noise: results/qt_B_noise
  C   H1+L1, chunk high-pass, 'bin'                  noise: results/qt_C_noise
  B2  H1, v1.3 default (chunk high-pass, 'bin+sft')  noise: results/qt_B_sftnorm_noise
  C2  H1+L1, v1.3 default maps                       noise: results/qt_C_sftnorm_noise
  injections: results/qt_<config>, signals of quick_signals.csv (6 masses x 6 distances).

For every configuration:
  1. a in A_GRID: Tsft re-selected by R_a, t set on the 61 noise packs with at most k above
     (k = 1 -> FAR ~3%, k = 0 -> FAR ~1.5%); the a with the largest efficiency on the
     calibration packs (104, 113, 126) at k = 1 is kept.
  2. evaluation packs (the other 10): efficiency per mass and distance, d50, paired
     counts against A, and a bootstrap of the noise packs for the threshold uncertainty.

    python studies/multi_detector/quick_test.py --root <dir with results/qt_*> --configs A B2 C2 \\
        --a-noise <campaign_v1.2>/per_tsft_noise.csv --out results/multi_detector/quick_test
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pbh_viterbi.search import selection as S  # noqa: E402

HERE = Path(__file__).resolve().parent
ALL_CONFIGS = {"A": "H1, v1.2 maps", "B": "H1, chunk HP", "C": "H1+L1, chunk HP",
               "B2": "H1, v1.3 maps", "C2": "H1+L1, v1.3 maps"}
RESULT_DIRS = {"A": "qt_A", "B": "qt_B", "C": "qt_C", "B2": "qt_B_sftnorm", "C2": "qt_C_sftnorm"}
COLORS = {"A": "#888780", "B": "#2a78d6", "C": "#eb6834", "B2": "#0b3d91", "C2": "#a8320c"}
CONFIGS = dict(ALL_CONFIGS)
CAL_PACKS = (104, 113, 126)
A_GRID = (0.0, 1.0, 2.0, 3.0, 5.0)
KS = {1: "FAR ~3% (1/61)", 0: "FAR ~1.5% (0/61)"}
N_BOOT = 1000


def threshold(noise_stat, k):
    """Smallest t with at most k noise values >= t (as analysis/compute_threshold_linear.py)."""
    v = np.sort(np.nan_to_num(np.asarray(noise_stat, float), nan=-np.inf))[::-1]
    return float(np.nextafter(v[k], np.inf)) if k < len(v) else -np.inf


def selected(per_tsft, a):
    return S.select_by_ranking(S.prepare(per_tsft), a)


def d50(dist, eff):
    """Distance where the efficiency crosses 0.5 (log-linear interpolation; nan if it never does)."""
    order = np.argsort(dist)
    d, e = np.asarray(dist)[order], np.asarray(eff)[order]
    for i in range(1, len(d)):
        if e[i - 1] >= 0.5 > e[i]:
            return float(np.exp(np.interp(0.5, [e[i], e[i - 1]], [np.log(d[i]), np.log(d[i - 1])])))
    return np.nan if e[-1] >= 0.5 else (float(d[0]) if e[0] < 0.5 else np.nan)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--a-noise", type=Path, required=True, help="campaign v1.2 per_tsft_noise.csv (config A)")
    parser.add_argument("--out", type=Path, default=Path("results/multi_detector/quick_test"))
    parser.add_argument("--configs", nargs="+", default=["A", "B", "C"], choices=list(ALL_CONFIGS),
                        help="Configurations to compare; A is the reference of the paired counts and ratios. "
                             "B2/C2: the v1.3 default maps (results/qt_{B,C}_sftnorm).")
    args = parser.parse_args()
    global CONFIGS
    CONFIGS = {c: ALL_CONFIGS[c] for c in args.configs}
    args.out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    rng = np.random.default_rng(1)

    l1 = np.loadtxt(HERE.parents[1] / "calibration" / "l1_packs.txt", comments="#", usecols=0, dtype=int)
    signals = pd.read_csv(HERE / "quick_signals.csv")
    noise_raw = {}
    if "A" in CONFIGS:
        noise_raw["A"] = pd.read_csv(args.a_noise)
        noise_raw["A"] = noise_raw["A"][noise_raw["A"]["pack"].isin(l1)]
    for c in CONFIGS:
        if c != "A":
            noise_raw[c] = pd.read_csv(args.root / f"results/{RESULT_DIRS[c]}_noise/search/per_tsft_noise.csv")
    inj_raw = {}
    for c in CONFIGS:
        files = [f for f in sorted((args.root / f"results/{RESULT_DIRS[c]}/search").glob("per_tsft_injected_pack-*.csv"))
                 if "job" not in f.name]
        inj_raw[c] = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    summary, chosen, eval_rows, boot = [], {}, {}, {}
    for c in CONFIGS:
        n_noise = noise_raw[c]["pack"].nunique()
        best = None
        for a in A_GRID:
            nz = selected(noise_raw[c], a).groupby("pack")["stat"].max()
            inj = selected(inj_raw[c], a).merge(signals, on="signal_id")
            for k in KS:
                t = threshold(nz.values, k)
                cal = inj[inj["pack"].isin(CAL_PACKS)]
                eff_cal = float((cal["stat"] >= t).mean())
                summary.append({"config": c, "a": a, "k": k, "t": t, "eff_cal": eff_cal,
                                "eff_eval": float((inj[~inj["pack"].isin(CAL_PACKS)]["stat"] >= t).mean()),
                                "n_noise": n_noise})
                if k == 1 and (best is None or eff_cal > best[1] + 1e-12 or
                               (abs(eff_cal - best[1]) < 1e-12 and abs(a - 2) < abs(best[0] - 2))):
                    best = (a, eff_cal)
        a = best[0]
        chosen[c] = a
        nz = selected(noise_raw[c], a).groupby("pack")["stat"].max()
        inj = selected(inj_raw[c], a).merge(signals, on="signal_id")
        ev = inj[~inj["pack"].isin(CAL_PACKS)].copy()
        for k in KS:
            ev[f"det{k}"] = ev["stat"] >= threshold(nz.values, k)
            boots = []
            for _ in range(N_BOOT):
                sample = rng.choice(nz.values, size=len(nz), replace=True)
                boots.append(float((ev["stat"] >= threshold(sample, k)).mean()))
            boot[(c, k)] = np.percentile(boots, [16, 84])
        eval_rows[c] = ev

    summ = pd.DataFrame(summary)
    summ.to_csv(args.out / "a_scan.csv", index=False)
    print("a scan (efficiency on the calibration packs, and on the evaluation packs for reference):")
    print(summ.pivot_table(index=["config", "a"], columns="k", values=["eff_cal", "eff_eval", "t"]).round(3).to_string())
    print("\nchosen a:", chosen)

    keys = ["pack", "signal_id"]
    lines = []
    per_mass = []
    for k, lab in KS.items():
        lines.append(f"\n{lab}")
        for c in CONFIGS:
            ev = eval_rows[c]
            lo, hi = boot[(c, k)]
            lines.append(f"  {c} ({CONFIGS[c]}, a={chosen[c]:g}): efficiency {ev[f'det{k}'].mean():.3f}"
                         f"  [noise bootstrap 68%: {lo:.3f}-{hi:.3f}]  (n={len(ev)})")
        for c in [c for c in CONFIGS if c != "A" and "A" in CONFIGS]:
            m = eval_rows["A"][keys + [f"det{k}"]].merge(eval_rows[c][keys + [f"det{k}"]], on=keys, suffixes=("_A", "_X"))
            gained = int((~m[f"det{k}_A"] & m[f"det{k}_X"]).sum())
            lost = int((m[f"det{k}_A"] & ~m[f"det{k}_X"]).sum())
            lines.append(f"  paired {c} vs A: detected by {c} only {gained}, by A only {lost}")
        for c in CONFIGS:
            ev = eval_rows[c]
            for mc, g in ev.groupby("mchirp"):
                eff = g.groupby("distance")[f"det{k}"].mean()
                per_mass.append({"k": k, "config": c, "mchirp": mc, "d50_kpc": d50(eff.index * 1e3, eff.values),
                                 "eff": float(g[f"det{k}"].mean())})
    print("\n".join(lines))
    pm = pd.DataFrame(per_mass)
    pm.to_csv(args.out / "per_mass.csv", index=False)
    table = pm.pivot_table(index=["k", "mchirp"], columns="config", values="d50_kpc")
    for c in CONFIGS:
        if c != "A" and "A" in CONFIGS:
            table[f"{c}/A"] = table[c] / table["A"]
    print("\nd50 (kpc) per mass, evaluation packs:")
    print(table.round(2).to_string())
    for c, ev in eval_rows.items():
        ev.to_csv(args.out / f"eval_{c}.csv", index=False)
    (args.out / "summary.txt").write_text("chosen a: " + str(chosen) + "\n" + "\n".join(lines) + "\n\n" + table.round(2).to_string() + "\n")

    # Figure 1: efficiency vs distance per mass (FAR ~3% solid, ~1.5% dashed)
    masses = sorted(signals["mchirp"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), constrained_layout=True)
    for ax, mc in zip(axes.flat, masses):
        for c in CONFIGS:
            g = eval_rows[c][np.isclose(eval_rows[c]["mchirp"], mc)]
            for k, ls in ((1, "-"), (0, "--")):
                eff = g.groupby("distance")[f"det{k}"].mean()
                ax.plot(eff.index * 1e3, eff.values, ls, color=COLORS[c], lw=2 if k == 1 else 1.2, marker="o" if k == 1 else None,
                        ms=4, label=f"{c}: {CONFIGS[c]}" if k == 1 else None)
        ax.axhline(0.5, color="#c3c2b7", lw=0.8)
        ax.set_xscale("log"); ax.set_ylim(-0.03, 1.03); ax.grid(alpha=0.3)
        ax.set_title(rf"$\mathcal{{M}}$ = {mc:.3g} M$_\odot$", fontsize=11)
        ax.set_xlabel("distance (kpc)")
    axes[0, 0].set_ylabel("detected fraction"); axes[1, 0].set_ylabel("detected fraction")
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Efficiency on 10 packs (solid: FAR ~3%, dashed: FAR ~1.5%)", fontsize=12)
    fig.savefig(args.out / "fig_efficiency_vs_distance.png", dpi=140)

    # Figure 2: d50 per mass, FAR ~3%
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    for c in CONFIGS:
        g = pm[(pm["k"] == 1) & (pm["config"] == c)].sort_values("mchirp")
        ax.plot(g["mchirp"], g["d50_kpc"], "-o", color=COLORS[c], lw=2, label=f"{c}: {CONFIGS[c]}")
        for mc in g.loc[g["d50_kpc"].isna(), "mchirp"]:  # still >= 50% at the largest distance: lower limit
            dmax = signals.loc[np.isclose(signals["mchirp"], mc), "distance"].max() * 1e3
            ax.plot([mc], [dmax], "^", color=COLORS[c], ms=10)
            ax.annotate(f"> {dmax:.0f}", (mc, dmax), textcoords="offset points", xytext=(6, 4), fontsize=9, color=COLORS[c])
    ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.3)
    ax.set_xlabel(r"chirp mass $\mathcal{M}$ (M$_\odot$)"); ax.set_ylabel(r"$d_{50}$ (kpc)")
    ax.set_title("Distance of 50% detection (FAR ~3%); triangle = lower limit", fontsize=11); ax.legend(fontsize=9)
    fig.savefig(args.out / "fig_d50_vs_mass.png", dpi=140)

    # Figure 3: noise R distribution and thresholds per configuration
    fig, axes = plt.subplots(1, len(CONFIGS), figsize=(4.4 * len(CONFIGS), 3.6), constrained_layout=True, sharey=True)
    for ax, c in zip(axes, CONFIGS):
        nz = selected(noise_raw[c], chosen[c]).groupby("pack")["stat"].max().values
        ev = eval_rows[c]
        # x range around the noise and the threshold (louder injections are piled in the last bin)
        bins = np.linspace(min(nz.min(), 0), max(3 * threshold(nz, 0), nz.max() + 2), 40)
        ax.hist(np.clip(ev["stat"], bins[0], bins[-1]), bins=bins, color=COLORS[c], alpha=0.45, density=True, label="injections (last bin: larger R)")
        ax.hist(nz, bins=bins, histtype="step", color="k", lw=1.6, density=True, label="noise (61 packs)")
        for k, ls in ((1, "-"), (0, "--")):
            ax.axvline(threshold(nz, k), color="k", ls=ls, lw=1.2)
        ax.set_title(f"{c}: {CONFIGS[c]}, a = {chosen[c]:g}", fontsize=11); ax.set_xlabel("R")
    axes[0].set_ylabel("density"); axes[0].legend(fontsize=8)
    fig.suptitle("R of noise (black) and injections; thresholds: solid FAR ~3%, dashed ~1.5%", fontsize=11)
    fig.savefig(args.out / "fig_noise_and_thresholds.png", dpi=140)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
