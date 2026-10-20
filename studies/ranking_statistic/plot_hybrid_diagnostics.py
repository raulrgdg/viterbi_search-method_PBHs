#!/usr/bin/env python3
"""[UNDER TEST] ROC and selected-Tsft diagnostics of the hybrid selection with NMSE_eff.

Four ways of choosing the Tsft, all ranked with the same detection statistic
R_a = nsigma - a log10(NMSE_eff) (a = config.RANKING_NMSE_WEIGHT, NMSE_eff =
NMSE + 1/(4 D^2), the quantisation limit of the fit):

  A        largest nsigma over the 13 Tsft of the paper
  B        largest nsigma over the 45 Tsft of list 95 (v1.1)
  NMSE_eff smallest NMSE_eff among the isolated Tsft of list 95
  hybrid   largest R_a among the isolated Tsft of list 95 (search default)

The threshold is set on the 141 noise packs only (in sample: nothing is fitted
to the injections, a is fixed). Outputs in <out>/:

  fig_hybrid_roc.png/.pdf           efficiency vs FAR, all injections and per signal slice
  fig_hybrid_selected_tsft.png/.pdf selected Tsft vs injected chirp mass with T_lim, and the
                                    Tsft chosen in the noise packs
  hybrid_roc.csv

    python studies/ranking_statistic/plot_hybrid_diagnostics.py [--tsft-results results/tsft_selection]
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyse import save, style_axes  # noqa: E402
from pbh_viterbi.config import MCHIRP_GRID, RANKING_NMSE_WEIGHT, TSFT_LISTS  # noqa: E402
from pbh_viterbi.search import selection  # noqa: E402
from pbh_viterbi.search.ranking import linear_stat  # noqa: E402
from pbh_viterbi.tsft import tsft_lim  # noqa: E402

FARS = (0.03, 0.01)
SLICES = {"light (signals 0-199)": (0, 200), "intermediate (200-399)": (200, 400), "heavy (400-599)": (400, 600)}
RULES = {  # label: (colour, marker)
    "A: max nσ, 13 paper Tsft": ("#2a78d6", "o"),
    "B: max nσ, 45 Tsft (v1.1)": ("#eb6834", "s"),
    "min NMSE_eff, 45 Tsft": ("#6250d6", "X"),
    "hybrid: max R_a, 45 Tsft (default)": ("#008300", "P"),
}
INK, MUTED = "#1f1f1e", "#6b6a64"


def select(label, df):
    """Selected row per chunk for one rule, with nmse replaced by NMSE_eff and ``stat`` = R_a."""
    if label.startswith("A"):
        sel = selection.apply_rule("A_max_nsigma_paper", df)
    elif label.startswith("B"):
        sel = selection.apply_rule("B_max_nsigma_95", df)
    elif label.startswith("min NMSE_eff"):
        sel = selection.select_by_ranking(df, np.inf)
    else:
        sel = selection.select_by_ranking(df, RANKING_NMSE_WEIGHT)
    sel["isolated"] = sel["isolated"].astype("boolean").fillna(False).astype(bool)
    if label[0] in "AB":  # max-nsigma rules: NMSE_eff of the selected block
        sel["nmse"] = selection.nmse_eff(sel)
    sel["stat"] = linear_stat(sel["nsigma"].where(sel["isolated"]), sel["nmse"].where(sel["isolated"]),
                              RANKING_NMSE_WEIGHT)
    return sel


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

    sel = {}
    for label in RULES:
        n = select(label, noise_tab)
        i = select(label, inj_tab).drop(columns=["mchirp", "distance"], errors="ignore").merge(params,
                                                                                              on=["pack", "signal_id"])
        i["slice"] = pd.cut(i["signal_id"], [0, 200, 400, 600], right=False, labels=list(SLICES))
        sel[label] = (n, i)
    n_noise, n_inj = len(sel["B: max nσ, 45 Tsft (v1.1)"][0]), len(params)

    # ---- ROC ------------------------------------------------------------------------------
    rows = []
    for label, (n, i) in sel.items():
        zn = np.sort(n["stat"].to_numpy())[::-1]
        for k in range(n_noise):
            if not np.isfinite(zn[k]):
                break
            above = i["stat"].to_numpy() >= np.nextafter(zn[k], np.inf)  # k noise packs strictly above
            rows.append(dict(rule=label, far=k / n_noise, slice="all", efficiency=above.mean()))
            for name in SLICES:
                rows.append(dict(rule=label, far=k / n_noise, slice=name,
                                 efficiency=above[(i["slice"] == name).to_numpy()].mean()))
    roc = pd.DataFrame(rows)
    roc.to_csv(args.out / "hybrid_roc.csv", index=False)

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelcolor": INK, "text.color": INK})
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)
    for ax, name in zip(axes, ["all"] + list(SLICES)):
        for label, (color, _marker) in RULES.items():
            r = roc[(roc["rule"] == label) & (roc["slice"] == name)]
            r = r[r["far"] > 0]
            ax.plot(r["far"], r["efficiency"], color=color, lw=2, drawstyle="steps-post", label=label)
        for far in FARS:
            ax.axvline(far, color=MUTED, lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xlim(0.8 / n_noise, 1)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{name} injections")
        ax.set_xlabel("false-alarm ratio (noise packs above threshold)")
        style_axes(ax)
    axes[0].set_ylabel("detection efficiency")
    axes[0].legend(loc="upper left", fontsize=7.5, frameon=False)
    fig.suptitle(f"ROC with R_a = nσ − {RANKING_NMSE_WEIGHT:g} log10 NMSE_eff, NMSE_eff = NMSE + 1/(4D²) "
                 f"({n_noise} noise packs, {n_inj} injections; dotted: FAR 3% and 1%)", x=0.01, ha="left")
    save(fig, args.out, "fig_hybrid_roc")

    # ---- selected Tsft ----------------------------------------------------------------------
    mc_line = np.logspace(np.log10(MCHIRP_GRID.min()), np.log10(MCHIRP_GRID.max()), 200)
    tsft_all = TSFT_LISTS["union"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (label, (color, _marker)) in zip(axes.flat, RULES.items()):
        n, i = sel[label]
        zn = np.sort(n["stat"].to_numpy())[::-1]
        thr = np.nextafter(zn[int(np.floor(FARS[0] * n_noise + 1e-9))], np.inf)
        det = (i["stat"] >= thr).to_numpy()
        jitter = 10 ** np.random.default_rng(0).uniform(-0.04, 0.04, len(i))
        ax.scatter(i.loc[~det, "mchirp"] * jitter[~det], i.loc[~det, "tsft"], s=6, color=MUTED, alpha=0.35,
                   label="not detected")
        ax.scatter(i.loc[det, "mchirp"] * jitter[det], i.loc[det, "tsft"], s=6, color=color, alpha=0.6,
                   label=f"detected, FAR {FARS[0]:.0%} ({det.mean():.1%})")
        ax.plot(mc_line, tsft_lim(mc_line), color=INK, lw=1, ls="--", label=r"$T_\mathrm{lim}(\mathcal{M})$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(MCHIRP_GRID.min() / 1.5, MCHIRP_GRID.max() * 1.5)
        ax.set_ylim(min(tsft_all) / 1.3, max(tsft_all) * 1.3)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("injected chirp mass (M☉)")
        ax.set_ylabel("selected Tsft (s)")
        ax.legend(fontsize=7, frameon=False, loc="lower left")
        style_axes(ax)

    ax = axes.flat[4]  # Tsft selected in the noise packs
    bins = np.logspace(np.log10(2.0), np.log10(500.0), 25)
    for label, (color, _marker) in RULES.items():
        ax.hist(sel[label][0]["tsft"], bins=bins, histtype="step", lw=1.8, color=color, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Tsft selected in the noise packs (s)")
    ax.set_ylabel("noise packs")
    ax.set_title("Tsft chosen by noise", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper center")
    style_axes(ax)

    ax = axes.flat[5]  # share of noise packs selecting a long Tsft
    shares = [np.mean(sel[label][0]["tsft"] >= 150) for label in RULES]
    ax.bar(range(len(RULES)), shares, color=[c for c, _ in RULES.values()], width=0.65)
    ax.set_xticks(range(len(RULES)), ["A", "B", "min\nNMSE_eff", "hybrid"])
    for k, v in enumerate(shares):
        ax.annotate(f"{v:.0%}", (k, v), ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("fraction of noise packs")
    ax.set_title("noise packs whose selected Tsft ≥ 150 s", fontsize=9)
    style_axes(ax)
    fig.subplots_adjust(hspace=0.38, wspace=0.28)
    fig.suptitle("Selected Tsft: injections (dashed: tracking limit T_lim) and noise", x=0.01, ha="left")
    save(fig, args.out, "fig_hybrid_selected_tsft")

    for far in FARS:
        print(f"FAR {far:.0%}:", ", ".join(
            f"{label.split(':')[0]} {roc[(roc['rule'] == label) & (roc['slice'] == 'all') & (roc['far'] <= far + 1e-9)]['efficiency'].max():.3f}"
            for label in RULES))


if __name__ == "__main__":
    main()
