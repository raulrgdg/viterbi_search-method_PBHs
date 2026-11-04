#!/usr/bin/env python3
"""Summary of an injection campaign against the paper (v1.0) results.

For the new campaign (hybrid Tsft selection, linear threshold R_a >= t, one
threshold JSON per FAR) and for the paper campaign (paper_data, polynomial
threshold), per chirp mass: recovered fraction and distance reach d_L,95%
(pbh_viterbi.sensitivity, Wilson bounds). Writes a CSV, a Markdown table and two
figures (recovered fraction and d95 vs chirp mass).

    python analysis/campaign_summary.py --campaign-csv results/search/injected_campaign.csv \\
        --threshold 3=results/search/threshold_far3.json --threshold 1=results/search/threshold_far1.json \\
        --out results/search/campaign_summary
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import _setup  # noqa: F401,E402  (adds src/ to sys.path)
from pbh_viterbi.paths import PAPER_DATA_DIR  # noqa: E402
from pbh_viterbi.sensitivity import distance_reach  # noqa: E402
from pbh_viterbi.threshold import is_candidate, load_threshold, nmse_values  # noqa: E402

PAPER_CSV = PAPER_DATA_DIR / "search_results" / "injected_search_O3b_H1_90packs.csv"
PAPER_THRESHOLD = PAPER_DATA_DIR / "calibration" / "far_threshold.json"
COLORS = {"paper": "#6b6a64", "3": "#2a78d6", "1": "#eb6834"}
SLICES = {"light": (0, 1.6e-3), "intermediate": (1.6e-3, 1.78e-2), "heavy": (1.78e-2, 1.0)}


def recovered(df, threshold_path):
    threshold = load_threshold(threshold_path)
    nmse = pd.to_numeric(nmse_values(threshold, df), errors="coerce")
    return is_candidate(threshold, nmse, pd.to_numeric(df["nsigma"], errors="coerce"))


def per_mass(df, rec, label):
    df = df.assign(rec=rec, mc=df["mchirp"].round(8))
    reach = distance_reach(df["mchirp"], df["distance"], rec).assign(mc=lambda r: r["mchirp"].round(8))
    out = df.groupby("mc").agg(n=("rec", "size"), recovered=("rec", "mean")).reset_index()
    out = out.merge(reach[["mc", "d95", "d95_low", "d95_high"]], on="mc", how="left")
    out[["d95", "d95_low", "d95_high"]] *= 1e3  # Mpc -> kpc
    return out.assign(set=label)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign-csv", type=Path, required=True)
    parser.add_argument("--threshold", action="append", required=True,
                        help="FAR_PERCENT=path of a linear threshold JSON (repeat for each FAR).")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    campaign = pd.read_csv(args.campaign_csv)
    paper = pd.read_csv(PAPER_CSV).drop_duplicates(["pack", "mchirp", "distance"])
    tables = [per_mass(paper, recovered(paper, PAPER_THRESHOLD), "paper")]
    totals = {"paper": dict(n=len(paper), recovered=float(recovered(paper, PAPER_THRESHOLD).mean()),
                            far="4/108 (polynomial)")}
    for item in args.threshold:
        far, path = item.split("=", 1)
        rec = recovered(campaign, Path(path))
        tables.append(per_mass(campaign, rec, far))
        info = json.loads(Path(path).read_text())
        totals[far] = dict(n=len(campaign), recovered=float(rec.mean()),
                           far=f"{info['false_positives']}/{info['noise_triggers_total']}")
        for name, (lo, hi) in SLICES.items():
            m = (campaign["mchirp"] >= lo) & (campaign["mchirp"] < hi)
            totals[far][name] = float(rec[m.to_numpy()].mean())
    for name, (lo, hi) in SLICES.items():
        m = (paper["mchirp"] >= lo) & (paper["mchirp"] < hi)
        totals["paper"][name] = float(recovered(paper, PAPER_THRESHOLD)[m.to_numpy()].mean())
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(args.out / "per_mass.csv", index=False)
    (args.out / "totals.json").write_text(json.dumps(totals, indent=2) + "\n", encoding="utf-8")

    wide = table.pivot(index="mc", columns="set", values=["recovered", "d95"])
    sets = ["paper"] + [t.split("=")[0] for t in args.threshold]
    lines = ["| 𝓜 (M☉) | " + " | ".join(f"recovered {s}" for s in sets) + " | " + " | ".join(f"d95 {s} (kpc)" for s in sets)
             + " |", "|---" * (1 + 2 * len(sets)) + "|"]
    for mc, r in wide.iterrows():
        lines.append(f"| {mc:.3g} | " + " | ".join(f"{r[('recovered', s)]:.2f}" for s in sets) + " | "
                     + " | ".join("—" if not np.isfinite(r[('d95', s)]) else f"{r[('d95', s)]:.3g}" for s in sets) + " |")
    lines += ["", "| set | recovered (all) | light | intermediate | heavy | noise above |", "|---|---|---|---|---|---|"]
    for s in sets:
        t = totals[s]
        lines.append(f"| {s} | {t['recovered']:.3f} | {t['light']:.3f} | {t['intermediate']:.3f} | {t['heavy']:.3f} | "
                     f"{t['far']} |")
    (args.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    labels = {"paper": "paper v1.0 (polynomial, FAR 3.7%)", "3": "v1.2 hybrid, FAR 3%", "1": "v1.2 hybrid, FAR 1%"}
    for quantity, ylabel, fname in (("recovered", "recovered fraction", "fig_recovered_vs_mass"),
                                    ("d95", r"$d_{L,95\%}$ (kpc)", "fig_d95_vs_mass")):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for s in sets:
            t = table[table["set"] == s]
            ax.plot(t["mc"], t[quantity], marker="o", ms=4, lw=1.8, color=COLORS.get(s, "#008300"),
                    ls="--" if s == "paper" else "-", label=labels.get(s, s))
            if quantity == "d95":
                ax.fill_between(t["mc"], t["d95_low"], t["d95_high"], color=COLORS.get(s, "#008300"), alpha=0.12)
        ax.set_xscale("log")
        if quantity == "d95":
            ax.set_yscale("log")
        ax.set_xlabel(r"chirp mass $\mathcal{M}$ (M$_\odot$)")
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#e4e3de")
        ax.legend(frameon=False, fontsize=8)
        fig.savefig(args.out / f"{fname}.png", dpi=160, bbox_inches="tight")
        fig.savefig(args.out / f"{fname}.pdf", bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
