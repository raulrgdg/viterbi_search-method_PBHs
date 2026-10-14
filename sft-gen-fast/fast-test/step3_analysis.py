#!/usr/bin/env python3
"""Step 3: effect of adding Tsft = 1 s on the Mc = 0.085 Msun injections.

Merges the step 2 job files into results/injections_14tsft.csv and writes, per
distance (results/summary_by_distance.csv):
  * median nsigma and NMSE of the recovered signal with 13 and 14 Tsft;
  * how often Tsft = 1 s is selected and how often the result changes;
  * fraction with status candidate_found (provisional linear threshold) with 13
    and 14 Tsft, and the same fraction in the paper campaign as a check.
No FAR threshold is recomputed.
"""

import numpy as np
import pandas as pd
from common import REPO, RESULTS

PAPER = REPO / "paper_data" / "search_results" / "injected_search_O3b_H1_90packs.csv"


def main():
    df = pd.concat([pd.read_csv(p) for p in sorted(RESULTS.glob("injections_14tsft.job*.csv"))], ignore_index=True)
    df = df.sort_values(["pack", "signal_id"])
    df.to_csv(RESULTS / "injections_14tsft.csv", index=False)

    paper = pd.read_csv(PAPER)
    paper = paper[np.isclose(paper.mchirp, 0.085, rtol=1e-3)]
    paper_frac = paper.groupby(paper.distance.round(10)).candidate.mean()

    rows = []
    for distance, g in df.groupby(df.distance.round(10)):
        rows.append({
            "distance": distance,
            "n": len(g),
            "median_nsigma_13": g.nsigma_13.median(),
            "median_nsigma_14": g.nsigma_14.median(),
            "median_nmse_13": g.nmse_13.median(),
            "median_nmse_14": g.nmse_14.median(),
            "frac_tsft1_selected": (g.tsft_14 == 1).mean(),
            "frac_result_changed": (~g.same_result).mean(),
            "frac_candidate_13": (g.status_13 == "candidate_found").mean(),
            "frac_candidate_14": (g.status_14 == "candidate_found").mean(),
            "frac_candidate_paper": paper_frac.get(distance, np.nan),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "summary_by_distance.csv", index=False)
    pd.set_option("display.width", 200)
    print(summary.round(4).to_string(index=False))
    print(f"\nTsft = 1 s selected in {(df.tsft_14 == 1).sum()}/{len(df)} injections; "
          f"result changed in {(~df.same_result).sum()}/{len(df)}")
    print("Selected Tsft with 13:", df.tsft_13.value_counts().sort_index().to_dict())
    print("Selected Tsft with 14:", df.tsft_14.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
