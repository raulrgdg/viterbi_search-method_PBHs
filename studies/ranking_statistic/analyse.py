#!/usr/bin/env python3
"""[UNDER TEST] Threshold review: generalisable thresholds and the gain of D' across the ranking space.

Uses the per-Tsft tables of the Tsft-selection study (studies/tsft_selection,
collected/*.csv) and the rules of pbh_viterbi.search.selection. Three analyses:

1. Ranking-space scan. For R_a = nsigma - a log10(NMSE), a from 0 (nsigma
   alone) to infinity (NMSE alone), the threshold is set on the noise packs
   only (at most floor(FAR N) above it), so nothing is fitted to the
   injections. Efficiency of each rule vs a, and the paired difference
   D' - B with a pack bootstrap interval and McNemar test: does D' help
   everywhere, or only where NMSE dominates?

2. Cross-validation of the threshold construction. Noise packs and injection
   packs are split in 3 folds (an injection fold holds one pack of each signal
   slice); each threshold is fitted on 2 folds and applied to the third. Pooled
   out-of-fold efficiency and FAR (Clopper-Pearson 68% interval) for:
     paper      polynomial in log10 NMSE (analysis/compute_threshold.py)
     monotone   piecewise linear, non-decreasing, flat outside the noise range
     linear     R_a with a chosen on the training folds, threshold from noise
   For each, the in-sample (fit = test) numbers show the optimism.

3. The (NMSE, nsigma) plane: noise and injection triggers of B and D', the
   short-block noise fits, and where D' gains or loses detections.

Outputs in <out>/: scan.csv, cv.csv, summary.md, fig_*.png/.pdf.

    python studies/ranking_statistic/analyse.py [--tsft-results results/tsft_selection]
        [--out results/ranking_statistic] [--processes 7]
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))

from pbh_viterbi.search.ranking import (  # noqa: E402
    MonotoneThreshold,
    block_steps,
    clopper_pearson,
    linear_stat,
    pack_folds,
    threshold_for_far,
)
from pbh_viterbi.search.selection import apply_rule, prepare  # noqa: E402

RULES = {"A": "A_max_nsigma_paper", "B": "B_max_nsigma_95", "D'": "D_topk_nmse_95", "D''": "D_topk_nmse_len_95"}
COLORS = {"A": "#2a78d6", "B": "#eb6834", "D'": "#008300", "D''": "#e34948"}
MARKERS = {"A": "o", "B": "s", "D'": "P", "D''": "*"}
METHOD_COLORS = {"paper": "#6250d6", "monotone": "#1baf7a", "linear": "#eda100"}
FARS = (0.03, 0.01)
A_GRID = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, np.inf)
SLICES = {"light": (0, 200), "intermediate": (200, 400), "heavy": (400, 600)}
INJ_FOLDS = {1: 0, 13: 0, 25: 0, 2: 1, 14: 1, 26: 1, 3: 2, 15: 2, 27: 2}
N_FOLDS = 3
N_BOOT = 1000
DP, DPP = "D'", "D''"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def style_axes(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def save(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=160, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def stat_a(df, a):
    """R_a of the selected triggers; a = inf -> -log10 NMSE."""
    nsigma = df["nsigma"].where(df["isolated"]).to_numpy(float)
    nmse = df["nmse"].where(df["isolated"]).to_numpy(float)
    if np.isinf(a):
        return linear_stat(np.zeros_like(nsigma), nmse, 1.0) + np.where(np.isfinite(nsigma), 0.0, -np.inf)
    return linear_stat(nsigma, nmse, a)


def detect_linear(noise, inj, a, far):
    max_false = int(np.floor(far * len(noise) + 1e-9))
    thr = threshold_for_far(stat_a(noise, a), max_false)
    return stat_a(inj, a) > thr, int(np.sum(stat_a(noise, a) > thr))


def paired(det_x, det_y):
    gain, loss = int(np.sum(det_x & ~det_y)), int(np.sum(~det_x & det_y))
    p = binomtest(gain, gain + loss).pvalue if gain + loss else 1.0
    return gain, loss, p


# ----------------------------------------------------------------------------
# Threshold constructions (fit on training triggers, return a detector)
# ----------------------------------------------------------------------------

def fit_paper(train_noise, train_inj, far, n_noise):
    """Paper polynomial: best degree 0-4 by differential evolution (analysis/compute_threshold.py)."""
    from compute_threshold import FIT_MAX_NMSE, MAX_DEGREE, better, fit_degree

    def xy(df):
        ok = df["isolated"] & np.isfinite(df["stat"]) & (df["nmse"] > 0) & (df["nmse"] <= FIT_MAX_NMSE)
        return np.log10(df.loc[ok, "nmse"].to_numpy()), df.loc[ok, "stat"].to_numpy()

    xs, ys = xy(train_inj)
    xn, yn = xy(train_noise)
    y_all = np.concatenate([ys, yn])
    y_span = max(float(np.ptp(y_all)), 1.0)
    best = None
    for degree in range(MAX_DEGREE + 1):
        result = fit_degree(degree, xs, ys, xn, yn, n_noise, far, y_span)
        if result["far"] <= far and better(result, best):
            best = result
    coeffs = best["coeffs"]

    def detector(df):
        ok = df["isolated"] & np.isfinite(df["stat"]) & (df["nmse"] > 0)
        out = np.zeros(len(df), bool)
        out[ok.to_numpy()] = df.loc[ok, "stat"].to_numpy() >= np.polyval(coeffs, np.log10(df.loc[ok, "nmse"]))
        return out

    return detector


def fit_monotone(train_noise, train_inj, far, n_noise):
    """Monotone, flat-outside threshold with knots at quantiles of the training noise log10 NMSE."""
    ok = train_noise["isolated"] & np.isfinite(train_noise["stat"])
    x = np.log10(train_noise.loc[ok, "nmse"].to_numpy())
    knots = np.unique(np.quantile(x, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    max_false = int(np.floor(far * n_noise + 1e-9))
    th = MonotoneThreshold(knots).fit(train_noise["stat"].where(train_noise["isolated"]),
                                      train_noise["nmse"], train_inj["stat"].where(train_inj["isolated"]),
                                      train_inj["nmse"], max_false)
    return lambda df: th.passes(df["stat"].where(df["isolated"]), df["nmse"])


def fit_linear(train_noise, train_inj, far, n_noise):
    """R_a with the a of A_GRID that recovers most training injections; threshold from the training noise."""
    max_false = int(np.floor(far * n_noise + 1e-9))
    best = None
    for a in A_GRID:
        thr = threshold_for_far(stat_a(train_noise, a), max_false)
        n_det = int(np.sum(stat_a(train_inj, a) > thr))
        if best is None or n_det > best[0]:
            best = (n_det, a, thr)
    _, a, thr = best
    return lambda df: stat_a(df, a) > thr


FITTERS = {"paper": fit_paper, "monotone": fit_monotone, "linear": fit_linear}


def cv_job(job):
    """One (method, rule, FAR) cross-validation: out-of-fold and in-sample detections."""
    method, rule, far, noise, inj = job
    fitter = FITTERS[method]
    det_inj = np.zeros(len(inj), bool)
    false = 0
    for fold in range(N_FOLDS):
        tr_n, te_n = noise[noise["fold"] != fold], noise[noise["fold"] == fold]
        tr_i, te_i = inj[inj["fold"] != fold], inj[inj["fold"] == fold]
        detector = fitter(tr_n, tr_i, far, len(tr_n))
        det_inj[(inj["fold"] == fold).to_numpy()] = detector(te_i)
        false += int(np.sum(detector(te_n)))
    full = fitter(noise, inj, far, len(noise))
    return dict(method=method, rule=rule, far=far, oof_det=det_inj, oof_false=false,
                in_det=full(inj), in_false=int(np.sum(full(noise))))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tsft-results", type=Path, default=ROOT / "results" / "tsft_selection")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "ranking_statistic")
    parser.add_argument("--processes", type=int, default=7)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    collected = args.tsft_results / "collected"

    noise_tab = prepare(pd.read_csv(collected / "per_tsft_noise.csv"))
    inj_raw = pd.read_csv(collected / "per_tsft_injected.csv")
    params = inj_raw[["pack", "signal_id", "mchirp", "distance"]].drop_duplicates(["pack", "signal_id"])
    inj_tab = prepare(inj_raw.drop(columns=["mchirp", "distance"]))
    noise_folds = pack_folds(noise_tab["pack"].unique(), N_FOLDS, seed=1)

    sel = {}
    for short, rule in RULES.items():
        n = apply_rule(rule, noise_tab, noise_tab)
        i = apply_rule(rule, inj_tab, noise_tab).drop(columns=["mchirp", "distance"], errors="ignore")
        i = i.merge(params, on=["pack", "signal_id"], how="left").sort_values(["pack", "signal_id"])
        n = n.sort_values("pack")
        n["fold"] = n["pack"].map(noise_folds)
        i["fold"] = i["pack"].map(INJ_FOLDS)
        i["slice"] = pd.cut(i["signal_id"], [0, 200, 400, 600], right=False, labels=list(SLICES))
        n["steps"] = block_steps(n["block_start_s"], n["block_end_s"], n["tsft"])
        i["steps"] = block_steps(i["block_start_s"], i["block_end_s"], i["tsft"])
        sel[short] = (n.reset_index(drop=True), i.reset_index(drop=True))
    n_noise, n_inj = len(sel["B"][0]), len(sel["B"][1])
    print(f"{n_noise} noise packs, {n_inj} injections")

    # ---- 1. ranking-space scan -------------------------------------------------------------
    rng = np.random.default_rng(0)
    inj_packs = np.array(sorted(sel["B"][1]["pack"].unique()))
    noise_packs = np.array(sorted(sel["B"][0]["pack"].unique()))
    boot_noise = rng.integers(0, len(noise_packs), (N_BOOT, len(noise_packs)))
    boot_inj = rng.integers(0, len(inj_packs), (N_BOOT, len(inj_packs)))
    scan_rows = []
    for far in FARS:
        for a in A_GRID:
            dets = {r: detect_linear(*sel[r], a, far) for r in RULES}
            for r, (det, nfalse) in dets.items():
                row = dict(far=far, a=a, rule=r, noise_above=nfalse, efficiency=det.mean())
                for name in SLICES:
                    row[f"eff_{name}"] = det[(sel[r][1]["slice"] == name).to_numpy()].mean()
                scan_rows.append(row)
            # paired D' - B, pack bootstrap (noise and injection packs resampled; threshold recomputed)
            max_false = int(np.floor(far * n_noise + 1e-9))
            boot_eff = {}
            for r in ("B", "D'", "D''", "A"):
                n, i = sel[r]
                zn = stat_a(n, a)  # one noise trigger per pack, ordered as noise_packs
                zi = stat_a(i, a)
                rows_of = [np.flatnonzero(i["pack"].to_numpy() == pk) for pk in inj_packs]
                effs = np.empty(N_BOOT)
                for b in range(N_BOOT):
                    thr = threshold_for_far(zn[boot_noise[b]], max_false)
                    idx = np.concatenate([rows_of[k] for k in boot_inj[b]])
                    effs[b] = np.mean(zi[idx] > thr)
                boot_eff[r] = effs
            for other in ("D'", "D''", "A"):
                gain, loss, p = paired(dets[other][0], dets["B"][0])
                lo, hi = np.percentile(boot_eff[other] - boot_eff["B"], [16, 84])
                scan_rows.append(dict(far=far, a=a, rule=f"{other}-B", gain=gain, loss=loss, mcnemar_p=p,
                                      efficiency=dets[other][0].mean() - dets["B"][0].mean(), ci_lo=lo, ci_hi=hi))
    scan = pd.DataFrame(scan_rows)
    scan.to_csv(out / "scan.csv", index=False)

    # ---- 2. cross-validation of the threshold constructions --------------------------------
    jobs = [(m, r, far, sel[r][0], sel[r][1]) for m in FITTERS for r in ("B", "D'", "D''") for far in FARS]
    with ProcessPoolExecutor(max_workers=args.processes) as pool:
        cv = list(pool.map(cv_job, jobs))
    cv_rows = []
    for res in cv:
        lo, hi = clopper_pearson(res["oof_false"], n_noise)
        cv_rows.append(dict(method=res["method"], rule=res["rule"], far=res["far"],
                            oof_efficiency=res["oof_det"].mean(), oof_false=res["oof_false"],
                            oof_far=res["oof_false"] / n_noise, oof_far_lo=lo, oof_far_hi=hi,
                            in_efficiency=res["in_det"].mean(), in_false=res["in_false"]))
    cvdf = pd.DataFrame(cv_rows)
    cvdf.to_csv(out / "cv.csv", index=False)
    cv_pairs = []
    for method in FITTERS:
        for far in FARS:
            get = {r["rule"]: r for r in cv if r["method"] == method and r["far"] == far}
            for other in ("D'", "D''"):
                gain, loss, p = paired(get[other]["oof_det"], get["B"]["oof_det"])
                heavy = (sel["B"][1]["slice"] == "heavy").to_numpy()
                hg, hl, hp = paired(get[other]["oof_det"][heavy], get["B"]["oof_det"][heavy])
                cv_pairs.append(dict(method=method, far=far, pair=f"{other} vs B", gain=gain, loss=loss, p=p,
                                     heavy_gain=hg, heavy_loss=hl, heavy_p=hp))
    cv_pairs = pd.DataFrame(cv_pairs)
    cv_pairs.to_csv(out / "cv_paired.csv", index=False)

    # ---- figures -------------------------------------------------------------------------
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelcolor": INK, "text.color": INK})
    a_pos = {a: k for k, a in enumerate(A_GRID)}
    a_labels = [("∞" if np.isinf(a) else f"{a:g}") for a in A_GRID]

    # Fig 1: efficiency vs a, and paired difference
    fig, axes = plt.subplots(2, len(FARS), figsize=(12, 7), sharex=True)
    for j, far in enumerate(FARS):
        ax = axes[0, j]
        for r in RULES:
            s = scan[(scan["far"] == far) & (scan["rule"] == r)]
            ax.plot(s["a"].map(a_pos), s["efficiency"], color=COLORS[r], marker=MARKERS[r], lw=1.8, ms=6,
                    label=r)
        ax.set_title(f"FAR {far:.0%}: threshold from noise only (R_a = nσ − a log10 NMSE)")
        ax.set_ylabel("efficiency (all injections)")
        ax.legend(frameon=False, fontsize=8)
        style_axes(ax)
        ax = axes[1, j]
        ax.axhline(0, color=MUTED, lw=0.8)
        for other in ("D'", "D''", "A"):
            s = scan[(scan["far"] == far) & (scan["rule"] == f"{other}-B")]
            x = s["a"].map(a_pos).to_numpy() + {"D'": -0.12, "D''": 0.0, "A": 0.12}[other]
            ax.errorbar(x, s["efficiency"], yerr=[np.clip(s["efficiency"] - s["ci_lo"], 0, None), np.clip(s["ci_hi"] - s["efficiency"], 0, None)],
                        color=COLORS[other], marker=MARKERS[other], ms=6, lw=0, elinewidth=1.4, capsize=2,
                        label=f"{other} − B (68% pack bootstrap)")
            for xx, (_, rr) in zip(x, s.iterrows()):
                if rr["mcnemar_p"] < 0.01 and other == "D'":
                    ax.annotate("*", (xx, rr["ci_hi"]), ha="center", va="bottom", color=INK)
        ax.set_ylabel("efficiency difference vs B")
        ax.set_xlabel("weight a of −log10 NMSE in the ranking statistic (0: nσ only, ∞: NMSE only)")
        ax.set_xticks(range(len(A_GRID)), a_labels)
        ax.legend(frameon=False, fontsize=8)
        style_axes(ax)
    fig.text(0.01, 0.005, "* McNemar p < 0.01 for D' vs B", color=MUTED, fontsize=8)
    save(fig, out, "fig_ranking_scan")

    # Fig 2: cross-validation, out-of-fold vs in-sample
    fig, axes = plt.subplots(1, len(FARS), figsize=(12, 4.2), sharey=True)
    for ax, far in zip(axes, FARS):
        c = cvdf[cvdf["far"] == far]
        labels = []
        for k, (_, row) in enumerate(c.iterrows()):
            color = METHOD_COLORS[row["method"]]
            ax.bar(k, row["oof_efficiency"], color=color, width=0.7)
            ax.plot([k - 0.35, k + 0.35], [row["in_efficiency"]] * 2, color=INK, lw=1.5)
            ax.annotate(f"{row['oof_false']}/{n_noise}", (k, 0.02), ha="center", color="white", fontsize=7,
                        rotation=90, va="bottom")
            labels.append(f"{row['method']}\n{row['rule']}")
        ax.set_xticks(range(len(labels)), labels, fontsize=7)
        ax.set_title(f"FAR target {far:.0%}: out-of-fold efficiency (bars), in-sample (lines); "
                     "label = out-of-fold noise above", fontsize=8)
        style_axes(ax)
    axes[0].set_ylabel("efficiency (all injections)")
    save(fig, out, "fig_cv")

    # Fig 3: the plane, B and D'
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    noise_pt = noise_tab[noise_tab["isolated"]]
    short_blocks = noise_pt[noise_pt["block_steps"] < 32]
    det_b = detect_linear(*sel["B"], 1.0, 0.03)[0]
    det_d = detect_linear(*sel["D'"], 1.0, 0.03)[0]
    for ax, r in zip(axes, ("B", "D'")):
        n, i = sel[r]
        ax.scatter(np.log10(short_blocks["nmse"]), short_blocks["nsigma"], s=6, color=MUTED, alpha=0.25,
                   marker="x", label="noise, any Tsft, block < 32 steps")
        ax.scatter(np.log10(i["nmse"]), i["nsigma"], s=5, color=COLORS[r], alpha=0.35, label=f"injections ({r})")
        ax.scatter(np.log10(n["nmse"]), n["nsigma"], s=18, color=INK, label=f"noise triggers ({r})")
        ax.set_xlabel("log10 NMSE")
        ax.set_title(f"rule {r}: selected triggers")
        ax.set_yscale("symlog", linthresh=3)
        style_axes(ax)
    i_d = sel["D'"][1]
    gained, lost = det_d & ~det_b, ~det_d & det_b
    axes[1].scatter(np.log10(i_d.loc[gained, "nmse"]), i_d.loc[gained, "nsigma"], s=22, facecolor="none",
                    edgecolor="#2a78d6", lw=1, label="detected by D' only (R_1, FAR 3%)")
    axes[1].scatter(np.log10(i_d.loc[lost, "nmse"]), i_d.loc[lost, "nsigma"], s=22, facecolor="none",
                    edgecolor="#e34948", lw=1, label="detected by B only")
    axes[0].set_ylabel("nσ (symlog)")
    for ax in axes:
        ax.legend(frameon=False, fontsize=7, loc="upper right")
    save(fig, out, "fig_plane")

    # Fig 4: block length vs NMSE
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(noise_pt["block_steps"], np.log10(noise_pt["nmse"]), s=4, color=MUTED, alpha=0.3,
               label="noise, every Tsft")
    for r in ("B", "D'"):
        i = sel[r][1]
        ax.scatter(i["steps"], np.log10(i["nmse"]), s=5, color=COLORS[r], alpha=0.4, label=f"injections, {r}")
    ax.axvline(32, color=INK, lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("isolated block length (SFT steps)")
    ax.set_ylabel("log10 NMSE")
    ax.set_title("Small NMSE of noise comes from short blocks")
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)
    save(fig, out, "fig_block_length")

    # ---- summary ---------------------------------------------------------------------------
    lines = ["# Threshold review and D' across the ranking space [UNDER TEST]", "",
             f"{n_noise} noise packs, {n_inj} injections; thresholds at FAR 3% and 1%.", "",
             "## 1. Ranking-space scan (threshold from noise only)", "",
             "| FAR | a | eff A | eff B | eff D' | eff D'' | D'−B (68% CI) | D' gain/loss vs B | McNemar p |",
             "|---|---|---|---|---|---|---|---|---|"]
    for far in FARS:
        for a in A_GRID:
            s = scan[(scan["far"] == far) & (scan["a"] == a)].set_index("rule")
            d = s.loc["D'-B"]
            lines.append(f"| {far:.0%} | {'∞' if np.isinf(a) else f'{a:g}'} | {s.loc['A', 'efficiency']:.3f} | "
                         f"{s.loc['B', 'efficiency']:.3f} | {s.loc[DP, 'efficiency']:.3f} | "
                         f"{s.loc[DPP, 'efficiency']:.3f} | {d['efficiency']:+.3f} "
                         f"[{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}] | {int(d['gain'])}/{int(d['loss'])} | "
                         f"{d['mcnemar_p']:.2g} |")
    lines += ["", "## 2. Cross-validation of the threshold construction (3 folds over packs)", "",
              "| FAR | method | rule | out-of-fold eff | in-sample eff | out-of-fold noise above | FAR 68% CI |",
              "|---|---|---|---|---|---|---|"]
    for _, r in cvdf.iterrows():
        lines.append(f"| {r['far']:.0%} | {r['method']} | {r['rule']} | {r['oof_efficiency']:.3f} | "
                     f"{r['in_efficiency']:.3f} | {r['oof_false']}/{n_noise} | "
                     f"[{r['oof_far_lo']:.3f}, {r['oof_far_hi']:.3f}] |")
    lines += ["", "Paired out-of-fold comparison:", "",
              "| FAR | method | pair | gain/loss | p | heavy gain/loss | heavy p |", "|---|---|---|---|---|---|---|"]
    for _, r in cv_pairs.iterrows():
        lines.append(f"| {r['far']:.0%} | {r['method']} | {r['pair']} | {r['gain']}/{r['loss']} | {r['p']:.2g} | "
                     f"{r['heavy_gain']}/{r['heavy_loss']} | {r['heavy_p']:.2g} |")
    lines += ["", "## Figures", "",
              "* `fig_ranking_scan`: efficiency vs a per rule, and D'−B, D''−B, A−B with bootstrap intervals.",
              "* `fig_cv`: out-of-fold vs in-sample efficiency and out-of-fold false alarms per construction.",
              "* `fig_plane`: (log10 NMSE, nσ) of the selected triggers, short-block noise fits, D' gains/losses.",
              "* `fig_block_length`: block length vs NMSE for noise (every Tsft) and injections."]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
