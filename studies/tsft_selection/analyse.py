#!/usr/bin/env python3
"""Compare the Tsft-selection rules at equal false-alarm ratio.

Reads the tables written by collect.py, applies every rule of
pbh_viterbi.search.selection to the noise packs and to the injections, and
compares the detection efficiency of the rules at the same FAR.

Two detection criteria:

  1D     stat > threshold, with the threshold set on the noise packs so that
         at most floor(FAR N) of them pass (stat = nsigma of the selected Tsft,
         or the Gaussian z of its p-value for the C rules; -inf when the
         selected Tsft has no isolated candidate). Gives full ROC curves.
  paper  (--paper-threshold) the (NMSE, stat) polynomial threshold of
         analysis/compute_threshold.py, fitted per rule and FAR, as the final
         analysis would do. Slow (~10-20 min per fit, run in parallel).

Outputs, in <results>/analysis/:
  selections.csv           selected Tsft and statistic per rule and chunk
  efficiency_by_mass.csv   efficiency per rule, FAR, criterion and chirp mass
  summary.csv, summary.md  reached FAR and efficiency per rule (all, per slice)
  roc.csv                  efficiency vs FAR per rule and slice (1D)
  fig_*.png / .pdf         figures (also listed in summary.md)

    python studies/tsft_selection/analyse.py [--results results/tsft_selection] [--paper-threshold]
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pbh_viterbi.config import MCHIRP_GRID, TSFT_LISTS  # noqa: E402
from pbh_viterbi.search.selection import RULES, apply_rule, prepare, threshold_at_far  # noqa: E402
from pbh_viterbi.threshold import is_candidate, load_threshold  # noqa: E402
from pbh_viterbi.tsft import tsft_lim  # noqa: E402

FARS = (0.03, 0.01)
SLICES = {"light (signals 0-199)": (0, 200), "intermediate (200-399)": (200, 400), "heavy (400-599)": (400, 600)}
INJ_PARAMS = ["pack", "signal_id", "mchirp", "distance", "ra", "dec", "pol", "t_to_merger"]

# Reference categorical palette (dataviz skill, light mode), fixed order = RULES order;
# a marker per rule as second encoding so identity never relies on colour alone.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#e34948", "#3a3a38", "#6250d6"]
MARKERS = ["o", "s", "^", "v", "D", "P", "*", "h", "X"]
LABELS = {
    "A_max_nsigma_paper": "A  max nσ, 13 paper Tsft",
    "B_max_nsigma_95": "B  max nσ, 45 Tsft (v1.1)",
    "C_min_p_95": "C  min p-value, 45 Tsft",
    "C_min_p_tail_95": "C' min p-value + tail fit",
    "D_isolated_95": "D  max nσ among isolated",
    "D_topk_nmse_95": "D' top-5 nσ, min NMSE",
    "D_topk_nmse_len_95": "D'' D' with blocks ≥ 32 steps",
    "D_topk_nmse_sat_95": "D''' D', saturated short blocks: NMSE ≥ 1e-3",
    "E_mass_consistent_95": "E  D + Tsft ≤ 1.5 T_lim(M_fit)",
}
STYLE = {rule: dict(color=COLORS[i], marker=MARKERS[i], label=LABELS[rule]) for i, rule in enumerate(RULES)}
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


def slice_of(signal_id):
    for name, (lo, hi) in SLICES.items():
        if lo <= signal_id < hi:
            return name
    return "?"


def d50_logistic(distance, detected):
    """Distance of 50% efficiency from a logistic fit in log10(distance) (NaN if undefined)."""
    x, y = np.log10(np.asarray(distance, float)), np.asarray(detected, float)
    if y.all() or not y.any():
        return np.nan

    def nll(theta):
        p = 1.0 / (1.0 + np.exp(np.clip((x - theta[0]) * theta[1], -50, 50)))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    res = minimize(nll, x0=[np.median(x), 5.0], method="Nelder-Mead")
    return 10 ** res.x[0] if res.success and res.x[1] > 0 else np.nan


def fit_paper_threshold(job):
    rule, far, noise_csv, signal_csv, out_json = job
    cmd = [sys.executable, str(ROOT / "analysis" / "compute_threshold.py"), "--far-limit", str(far),
           "--noise-csv", str(noise_csv), "--signal-csv", str(signal_csv), "--output", str(out_json)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return rule, far, result.returncode, result.stderr[-500:]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "tsft_selection")
    parser.add_argument("--paper-threshold", action="store_true", help="Also fit the paper (NMSE, stat) threshold.")
    parser.add_argument("--processes", type=int, default=8)
    args = parser.parse_args()
    collected, out = args.results / "collected", args.results / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    noise = prepare(pd.read_csv(collected / "per_tsft_noise.csv"))
    injected_raw = pd.read_csv(collected / "per_tsft_injected.csv")
    params = injected_raw[INJ_PARAMS].drop_duplicates(["pack", "signal_id"])
    injected = prepare(injected_raw.drop(columns=[c for c in INJ_PARAMS if c not in ("pack", "signal_id")]))
    n_noise, n_inj = noise["pack"].nunique(), len(params)
    print(f"{n_noise} noise packs, {n_inj} injections in packs {sorted(params['pack'].unique())}")

    # ---- selections -------------------------------------------------------
    sel_noise, sel_inj = {}, {}
    for rule in RULES:
        sel_noise[rule] = apply_rule(rule, noise, noise)
        s = apply_rule(rule, injected, noise).drop(columns=["mchirp", "distance"], errors="ignore")
        sel_inj[rule] = s.merge(params, on=["pack", "signal_id"], how="left")
        sel_inj[rule]["slice"] = sel_inj[rule]["signal_id"].map(slice_of)
    keep = ["rule", "injected", "pack", "signal_id", "tsft", "nsigma", "stat", "isolated", "iso_status", "nmse", "mass"]
    pd.concat([pd.concat([sel_noise[r][keep], sel_inj[r][keep + ["mchirp", "distance"]]]) for r in RULES]).to_csv(
        out / "selections.csv", index=False)

    # ---- 1D thresholds, efficiencies, ROC ----------------------------------
    eff_rows, summary_rows, roc_rows = [], [], []
    for rule in RULES:
        zn, si = sel_noise[rule]["stat"].to_numpy(), sel_inj[rule]
        for k in range(n_noise + 1):
            thr = np.sort(zn)[::-1][k] if k < n_noise else -np.inf
            reached = np.sum(zn > thr)
            det = si["stat"] > thr
            roc_rows.append(dict(rule=rule, far=reached / n_noise, slice="all", efficiency=det.mean()))
            for name in SLICES:
                roc_rows.append(dict(rule=rule, far=reached / n_noise, slice=name,
                                     efficiency=det[si["slice"] == name].mean()))
        for far in FARS:
            thr, reached, n = threshold_at_far(zn, far)
            si = si.assign(detected=si["stat"] > thr)
            summary_rows.append(dict(rule=rule, criterion="1D", far_target=far, far_reached=f"{reached}/{n}",
                                     threshold=thr, efficiency_all=si["detected"].mean(),
                                     **{f"eff_{name.split()[0]}": si.loc[si["slice"] == name, "detected"].mean()
                                        for name in SLICES}))
            for mc, g in si.groupby(si["mchirp"].round(8)):
                eff_rows.append(dict(rule=rule, criterion="1D", far=far, mchirp=mc, n=len(g),
                                     efficiency=g["detected"].mean(), d50_kpc=1e3 * d50_logistic(g["distance"],
                                                                                                 g["detected"])))

    # ---- paper (NMSE, stat) threshold --------------------------------------
    if args.paper_threshold:
        work = out / "paper_threshold"
        work.mkdir(exist_ok=True)
        jobs = []
        for rule in RULES:
            for frame, kind in ((sel_noise[rule], "noise"), (sel_inj[rule], "signal")):
                f = frame.assign(nsigma=frame["stat"].replace(-np.inf, np.nan))
                f[["pack", "signal_id", "nsigma", "nmse"]].to_csv(work / f"{rule}_{kind}.csv", index=False)
            for far in FARS:
                jobs.append((rule, far, work / f"{rule}_noise.csv", work / f"{rule}_signal.csv",
                             work / f"{rule}_far{far}.json"))
        with ProcessPoolExecutor(max_workers=args.processes) as pool:
            for rule, far, code, err in pool.map(fit_paper_threshold, jobs):
                print(f"paper threshold {rule} FAR {far}: {'ok' if code == 0 else 'FAILED ' + err}")
        for rule in RULES:
            for far in FARS:
                path = work / f"{rule}_far{far}.json"
                if not path.exists():
                    continue
                th = load_threshold(path)
                sn, si = sel_noise[rule], sel_inj[rule]
                nd = is_candidate(th, sn["nmse"], sn["stat"].replace(-np.inf, np.nan))
                si = si.assign(detected=is_candidate(th, si["nmse"], si["stat"].replace(-np.inf, np.nan)))
                summary_rows.append(dict(rule=rule, criterion="paper", far_target=far,
                                         far_reached=f"{int(nd.sum())}/{len(sn)}", threshold=np.nan,
                                         efficiency_all=si["detected"].mean(),
                                         **{f"eff_{name.split()[0]}": si.loc[si["slice"] == name, "detected"].mean()
                                            for name in SLICES}))
                for mc, g in si.groupby(si["mchirp"].round(8)):
                    eff_rows.append(dict(rule=rule, criterion="paper", far=far, mchirp=mc, n=len(g),
                                         efficiency=g["detected"].mean(),
                                         d50_kpc=1e3 * d50_logistic(g["distance"], g["detected"])))

    eff, summary, roc = pd.DataFrame(eff_rows), pd.DataFrame(summary_rows), pd.DataFrame(roc_rows)
    eff.to_csv(out / "efficiency_by_mass.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    roc.to_csv(out / "roc.csv", index=False)

    # ---- figures -------------------------------------------------------------
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelcolor": INK, "text.color": INK})
    figs = []

    # ROC curves per slice
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    for ax, name in zip(axes, ["all"] + list(SLICES)):
        for rule in RULES:
            r = roc[(roc["rule"] == rule) & (roc["slice"] == name)].groupby("far")["efficiency"].max()
            r = r[r.index > 0]
            st = STYLE[rule]
            ax.plot(r.index, r.values, color=st["color"], lw=2, drawstyle="steps-post", label=st["label"])
        for far in FARS:
            ax.axvline(far, color=MUTED, lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xlim(1 / n_noise * 0.8, 1)
        ax.set_title(f"{name} injections")
        ax.set_xlabel("false-alarm ratio (noise packs above threshold)")
        style_axes(ax)
    axes[0].set_ylabel("detection efficiency")
    axes[-1].legend(loc="lower right", fontsize=7, frameon=False)
    fig.suptitle(f"ROC, 1D criterion ({n_noise} noise packs, {n_inj} injections)", x=0.01, ha="left")
    save(fig, out, "fig_roc")
    figs.append(("fig_roc", "Efficiency vs FAR for every rule, all injections and per signal slice (1D)."))

    # Efficiency and d50 vs chirp mass at each FAR
    criteria = [c for c in ("1D", "paper") if c in set(eff["criterion"])]
    for quantity, ylabel, fname in (("efficiency", "detection efficiency", "fig_efficiency_vs_mass"),
                                    ("d50_kpc", "distance of 50% efficiency (kpc)", "fig_d50_vs_mass")):
        fig, axes = plt.subplots(len(criteria), len(FARS), figsize=(11, 3.8 * len(criteria)), squeeze=False,
                                 sharex=True)
        for i, crit in enumerate(criteria):
            for j, far in enumerate(FARS):
                ax = axes[i, j]
                for rule in RULES:
                    e = eff[(eff["rule"] == rule) & (eff["far"] == far) & (eff["criterion"] == crit)]
                    st = STYLE[rule]
                    ax.plot(e["mchirp"], e[quantity], color=st["color"], marker=st["marker"], ms=5, lw=1.6,
                            label=st["label"])
                ax.set_xscale("log")
                if quantity == "d50_kpc":
                    ax.set_yscale("log")
                ax.set_title(f"{crit} criterion, FAR {far:.0%}")
                ax.set_ylabel(ylabel)
                style_axes(ax)
            axes[i, 0].legend(fontsize=7, frameon=False)
        for ax in axes[-1]:
            ax.set_xlabel("chirp mass (M☉)")
        save(fig, out, fname)
        figs.append((fname, f"{ylabel.capitalize()} per chirp mass, per rule and FAR."))

    # Selected Tsft of the injections against T_lim of the true mass
    mc_line = np.logspace(np.log10(MCHIRP_GRID.min()), np.log10(MCHIRP_GRID.max()), 200)
    fig, axes = plt.subplots(2, 5, figsize=(19, 7.5))
    tsft_all = TSFT_LISTS["union"]
    for ax, rule in zip(axes.flat, RULES):
        s = sel_inj[rule]
        thr = threshold_at_far(sel_noise[rule]["stat"].to_numpy(), FARS[0])[0]
        det = s["stat"] > thr
        jitter = 10 ** np.random.default_rng(0).uniform(-0.04, 0.04, len(s))
        ax.scatter(s.loc[~det, "mchirp"] * jitter[~det], s.loc[~det, "tsft"], s=6, color=MUTED, alpha=0.35,
                   label="not detected")
        ax.scatter(s.loc[det, "mchirp"] * jitter[det], s.loc[det, "tsft"], s=6, color=STYLE[rule]["color"],
                   alpha=0.6, label=f"detected (FAR {FARS[0]:.0%})")
        ax.plot(mc_line, tsft_lim(mc_line), color=INK, lw=1, ls="--", label="T_lim(M)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(MCHIRP_GRID.min() / 1.5, MCHIRP_GRID.max() * 1.5)
        ax.set_ylim(min(tsft_all) / 1.3, max(tsft_all) * 1.3)
        ax.set_title(LABELS[rule], fontsize=9)
        style_axes(ax)
    ax = axes.flat[-1]
    tsft_counts = pd.DataFrame({LABELS[r][:4].strip(): sel_noise[r]["tsft"].value_counts() for r in RULES}).fillna(0)
    long_share = {k: tsft_counts.loc[tsft_counts.index >= 150, k].sum() / n_noise for k in tsft_counts}
    ax.bar(range(len(long_share)), list(long_share.values()), color=COLORS[:len(long_share)], width=0.7)
    ax.set_xticks(range(len(long_share)), list(long_share.keys()))
    ax.set_title("noise packs whose selected Tsft ≥ 150 s", fontsize=9)
    ax.set_yscale("linear")
    ax.set_ylabel("fraction of noise packs")
    style_axes(ax)
    axes[0, 0].legend(fontsize=7, frameon=False, loc="upper right")
    for a in axes[-1, :]:
        a.set_xlabel("injected chirp mass (M☉)")
    for a in axes[:, 0]:
        a.set_ylabel("selected Tsft (s)")
    save(fig, out, "fig_selected_tsft")
    figs.append(("fig_selected_tsft", "Selected Tsft vs injected chirp mass per rule (dashed: T_lim); last panel: "
                                      "share of noise packs selecting Tsft >= 150 s."))

    # Noise background per Tsft: does the tail grow with Tsft?
    g = noise[noise["tsft"].isin({round(t, 6) for t in TSFT_LISTS["95"]})].groupby("tsft")["nsigma"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for q, label, color, marker in ((g.quantile(0.99), "99th percentile", COLORS[0], "o"),
                                    (g.max(), "maximum", COLORS[1], "s"),
                                    (g.apply(lambda x: (x > 2).mean() * 10), "10 × fraction above 2", COLORS[2], "^")):
        ax.plot(q.index, q.values, color=color, marker=marker, ms=4, lw=1.6, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Tsft (s)")
    ax.set_ylabel("noise nσ")
    ax.set_title(f"Noise nσ tail per Tsft ({n_noise} packs, list 95)")
    ax.legend(fontsize=8, frameon=False)
    style_axes(ax)
    save(fig, out, "fig_noise_tail")
    figs.append(("fig_noise_tail", "Tail of the noise nσ distribution per Tsft."))

    # ---- summary ---------------------------------------------------------------
    lines = ["# Tsft-selection study: summary", "",
             f"{n_noise} noise packs, {n_inj} injections (packs {sorted(params['pack'].unique())}).", "",
             "| rule | criterion | FAR target | FAR reached | efficiency | light | intermediate | heavy |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        lines.append(f"| {LABELS[r['rule']]} | {r['criterion']} | {r['far_target']:.0%} | {r['far_reached']} | "
                     f"{r['efficiency_all']:.3f} | {r['eff_light']:.3f} | {r['eff_intermediate']:.3f} | "
                     f"{r['eff_heavy']:.3f} |")
    lines += ["", "## Figures", ""] + [f"* `{name}.png`: {text}" for name, text in figs]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "run_info.json").write_text(json.dumps({"rules": list(RULES), "fars": FARS, "n_noise": int(n_noise),
                                                   "n_injections": int(n_inj)}, indent=1), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
