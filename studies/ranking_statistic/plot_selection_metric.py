#!/usr/bin/env python3
"""[UNDER TEST] Tsft chosen by nsigma vs by NMSE: detection efficiency at fixed FAR.

Three ways of choosing the Tsft of a chunk among the 45 of list 95:

  nsigma          largest nsigma (rule B, v1.1 default)
  NMSE            smallest NMSE among the Tsft with an isolated candidate
                  (D' with K = 45), raw NMSE
  NMSE, D'''      the same, but the NMSE of saturated short blocks (<= 24 steps
                  drifting one remapped bin per step) is raised to >= 1e-3,
                  since noise fits such straight segments with NMSE ~ 0
  R_a (hybrid)    the isolated Tsft of largest R_a = nsigma - a log10 NMSE
                  (D''' NMSE), with the same a as the threshold: a single
                  parameter, chosen on the training folds

The detection statistic is the same for all: R_a = nsigma - a log10(NMSE) of
the chosen Tsft, with the linear threshold of studies/ranking_statistic
(a and the threshold fitted on 2 of 3 pack folds, applied to the third), so the
efficiencies and false alarms are out of fold.

    python studies/ranking_statistic/plot_selection_metric.py [--tsft-results results/tsft_selection]
        [--out results/ranking_statistic]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pbh_viterbi.search.selection as selection  # noqa: E402
from analyse import A_GRID, INJ_FOLDS, N_FOLDS, cv_job, save, style_axes  # noqa: E402
from pbh_viterbi.search.ranking import clopper_pearson, pack_folds, threshold_for_far  # noqa: E402

FARS = (0.03, 0.01)
METRICS = {  # label: (rule, TOP_K, colour, marker, line style)
    "Tsft by nσ (B, v1.1)": ("B_max_nsigma_95", selection.TOP_K, "#eb6834", "s", "-"),
    "Tsft by NMSE (D''', K = 45)": ("D_topk_nmse_sat_95", 45, "#6250d6", "X", "-"),
    "Tsft by raw NMSE (no saturated-block fix)": ("D_topk_nmse_95", 45, "#6b6a64", "o", "--"),
    "Tsft by R_a = nσ − a log10 NMSE (hybrid)": ("hybrid", None, "#008300", "P", "-"),
}
HYBRID = "Tsft by R_a = nσ − a log10 NMSE (hybrid)"


def hybrid_cv(noise_tab, inj_tab, params, folds, far):
    """Out-of-fold detections of the hybrid rule: a (selection and threshold) chosen on the training folds."""
    sn, si = {}, {}
    for a in A_GRID:
        n = selection.select_by_ranking(noise_tab, a).sort_values("pack").reset_index(drop=True)
        i = (selection.select_by_ranking(inj_tab, a).drop(columns=["mchirp", "distance"], errors="ignore")
             .merge(params, on=["pack", "signal_id"]).sort_values(["pack", "signal_id"]).reset_index(drop=True))
        sn[a], si[a] = n["stat"].to_numpy(), i["stat"].to_numpy()
    inj = i[["pack", "signal_id", "mchirp", "distance"]].copy()
    n_fold = n["pack"].map(folds).to_numpy()
    i_fold = inj["pack"].map(INJ_FOLDS).to_numpy()
    det, false, chosen = np.zeros(len(inj), bool), 0, []
    for fold in range(N_FOLDS):
        trn, tst, tri, tsi = n_fold != fold, n_fold == fold, i_fold != fold, i_fold == fold
        max_false = int(np.floor(far * trn.sum() + 1e-9))
        best = max(A_GRID, key=lambda a: np.mean(si[a][tri] > threshold_for_far(sn[a][trn], max_false)))
        thr = threshold_for_far(sn[best][trn], max_false)
        det[tsi] = si[best][tsi] > thr
        false += int(np.sum(sn[best][tst] > thr))
        chosen.append(best)
    return inj.assign(det=det), false, len(n), chosen
SLICES = {r"light ($\mathcal{M} \leq 1.6\times10^{-3}$ M☉)": (0, 200), "intermediate": (200, 400), r"heavy ($\mathcal{M} \geq 1.8\times10^{-2}$ M☉)": (400, 600)}
INK, MUTED = "#1f1f1e", "#6b6a64"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tsft-results", type=Path, default=ROOT / "results" / "tsft_selection")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "ranking_statistic")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    collected = args.tsft_results / "collected"

    noise_tab = selection.prepare(pd.read_csv(collected / "per_tsft_noise.csv"))
    raw = pd.read_csv(collected / "per_tsft_injected.csv")
    params = raw[["pack", "signal_id", "mchirp", "distance"]].drop_duplicates(["pack", "signal_id"])
    inj_tab = selection.prepare(raw.drop(columns=["mchirp", "distance"]))
    folds = pack_folds(noise_tab["pack"].unique(), N_FOLDS, seed=1)

    results, rows = {}, []
    for label, (rule, top_k, *_style) in METRICS.items():
        if rule == "hybrid":
            for far in FARS:
                det, false, n_noise, chosen = hybrid_cv(noise_tab, inj_tab, params, folds, far)
                print(f"hybrid FAR {far:.0%}: a chosen per fold {chosen}")
                results[(label, far)] = (det, false, n_noise)
                rows.append(dict(metric=label, far=far, recovered=int(det["det"].sum()), injections=len(det),
                                 efficiency=det["det"].mean(), false_positives=false, noise=n_noise))
            continue
        selection.TOP_K = top_k
        n = selection.apply_rule(rule, noise_tab, noise_tab).sort_values("pack").reset_index(drop=True)
        i = (selection.apply_rule(rule, inj_tab, noise_tab).drop(columns=["mchirp", "distance"], errors="ignore")
             .merge(params, on=["pack", "signal_id"]).sort_values(["pack", "signal_id"]).reset_index(drop=True))
        n["fold"], i["fold"] = n["pack"].map(folds), i["pack"].map(INJ_FOLDS)
        for far in FARS:
            res = cv_job(("linear", label, far, n, i))
            results[(label, far)] = (i.assign(det=res["oof_det"]), res["oof_false"], len(n))
            rows.append(dict(metric=label, far=far, recovered=int(res["oof_det"].sum()), injections=len(i),
                             efficiency=res["oof_det"].mean(), false_positives=res["oof_false"], noise=len(n)))
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out / "selection_metric.csv", index=False)
    print(summary.round(3).to_string(index=False))

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelcolor": INK, "text.color": INK})

    # Figure 1: efficiency vs chirp mass, FAR 3% and 1%
    fig, axes = plt.subplots(1, len(FARS), figsize=(12, 4.4), sharey=True)
    for ax, far in zip(axes, FARS):
        for label, (_rule, _k, color, marker, ls) in METRICS.items():
            det, false, n_noise = results[(label, far)]
            g = det.groupby(det["mchirp"].round(8))["det"].agg(["sum", "count"])
            eff = g["sum"] / g["count"]
            ci = np.array([clopper_pearson(int(k), int(c)) for k, c in zip(g["sum"], g["count"])])
            total = det["det"].mean()
            ax.errorbar(g.index, eff, yerr=[eff - ci[:, 0], ci[:, 1] - eff], color=color, marker=marker, ms=5,
                        lw=1.8, ls=ls, elinewidth=0.8, capsize=0,
                        label=f"{label}: {total:.1%} recovered, {false}/{n_noise} noise above")
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.set_xlabel("injected chirp mass (M☉)")
        ax.set_title(f"FAR {far:.0%} (out of fold, linear threshold on R_a = nσ − a log10 NMSE)")
        ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16))
        style_axes(ax)
    axes[0].set_ylabel("detection efficiency (68% CL)")
    fig.suptitle("Choosing the Tsft by NMSE instead of nσ (141 noise packs, 1800 injections)", x=0.01, ha="left")
    save(fig, args.out, "fig_selection_metric_vs_mass")

    # Figure 2: efficiency vs distance per mass range, FAR 1% and 3%
    fig, axes = plt.subplots(len(FARS), len(SLICES), figsize=(14, 7.5), sharex=True, sharey=True)
    for row, far in enumerate(FARS):
        for col, (name, (lo, hi)) in enumerate(SLICES.items()):
            ax = axes[row, col]
            for label, (_rule, _k, color, marker, ls) in METRICS.items():
                det = results[(label, far)][0]
                det = det[(det["signal_id"] >= lo) & (det["signal_id"] < hi)]
                g = det.groupby((det["distance"] * 1e3).round(3))["det"].mean()
                ax.plot(g.index, g.values, color=color, marker=marker, ms=4, lw=1.6, ls=ls,
                        label=f"{label} ({det['det'].mean():.1%})")
            ax.set_xscale("log")
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(f"{name}, FAR {far:.0%}", fontsize=9)
            ax.legend(frameon=False, fontsize=7, loc="lower left")
            style_axes(ax)
        axes[row, 0].set_ylabel("detection efficiency")
    for ax in axes[-1]:
        ax.set_xlabel("injected distance (kpc)")
    save(fig, args.out, "fig_selection_metric_vs_distance")


if __name__ == "__main__":
    main()
