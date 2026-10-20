#!/usr/bin/env python3
"""[UNDER TEST] Ranking plane (NMSE_eff, nsigma) of the hybrid Tsft selection with its linear threshold.

Selects the Tsft of every noise pack and injection of the Tsft-selection study
with the search rule (largest R_a = nsigma - a log10 NMSE_eff, a =
config.RANKING_NMSE_WEIGHT), sets the linear threshold at FAR 3% on the noise
(analysis/compute_threshold_linear.py) and draws Fig. 5 of the paper for it
(figures/fig5_trigger_plane.py).

    python studies/ranking_statistic/plot_plane.py [--tsft-results results/tsft_selection]
        [--out results/ranking_statistic]
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pbh_viterbi.config import RANKING_NMSE_WEIGHT  # noqa: E402
from pbh_viterbi.search import selection  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tsft-results", type=Path, default=ROOT / "results" / "tsft_selection")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "ranking_statistic")
    parser.add_argument("--far", type=float, default=0.03)
    args = parser.parse_args()
    work = args.out / "plane"
    work.mkdir(parents=True, exist_ok=True)
    collected = args.tsft_results / "collected"

    noise = selection.prepare(pd.read_csv(collected / "per_tsft_noise.csv"))
    raw = pd.read_csv(collected / "per_tsft_injected.csv")
    params = raw[["pack", "signal_id", "mchirp", "distance"]].drop_duplicates(["pack", "signal_id"])
    inj = selection.prepare(raw.drop(columns=["mchirp", "distance"]))
    for df, name, extra in ((noise, "noise", None), (inj, "injected", params)):
        sel = selection.select_by_ranking(df, RANKING_NMSE_WEIGHT)
        sel["nmse_eff"] = sel["nmse"]  # select_by_ranking returns NMSE_eff in the nmse column
        if extra is not None:
            sel = sel.drop(columns=["mchirp", "distance"], errors="ignore").merge(extra, on=["pack", "signal_id"])
        sel.to_csv(work / f"{name}.csv", index=False)

    threshold = work / f"threshold_far{args.far:g}.json"
    run = [sys.executable]
    subprocess.run(run + [str(ROOT / "analysis" / "compute_threshold_linear.py"), "--far-limit", str(args.far),
                          "--noise-csv", str(work / "noise.csv"), "--signal-csv", str(work / "injected.csv"),
                          "--output", str(threshold)], check=True)
    subprocess.run(run + [str(ROOT / "figures" / "fig5_trigger_plane.py"), "--noise-csv", str(work / "noise.csv"),
                          "--signal-csv", str(work / "injected.csv"), "--threshold", str(threshold),
                          "--output", str(args.out / "fig_plane_hybrid.png")], check=True)


if __name__ == "__main__":
    main()
