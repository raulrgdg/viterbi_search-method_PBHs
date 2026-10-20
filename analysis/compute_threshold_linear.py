#!/usr/bin/env python3
"""[UNDER TEST] Linear detection threshold of the hybrid Tsft selection, set on noise only.

A trigger is a candidate when

    R_a = nsigma - a log10(NMSE_eff) >= t,

i.e. nsigma >= t + a log10(NMSE_eff), the degree-1 case of the threshold model
of pbh_viterbi.threshold. a is fixed (config.RANKING_NMSE_WEIGHT, chosen on the
calibration set of studies/ranking_statistic), so only t is set here, from the
noise triggers alone: at most floor(FAR N) of the N noise chunks are above it.
Noise chunks without an isolated candidate are never above it. The FAR can
only take the values k/N; the output gives the reached k/N and its
Clopper-Pearson intervals.

    python analysis/compute_threshold_linear.py --far-limit 0.03 \\
        --noise-csv results/search/search_results_noise.csv \\
        [--signal-csv results/search/injected_campaign.csv] --output results/search/threshold_far3.json

--far-limit has no default: it is a choice of the analysis. --signal-csv is only
used to report the recovered fraction, not to set the threshold.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _setup  # noqa: F401  (adds src/ to sys.path)
from pbh_viterbi.config import RANKING_NMSE_WEIGHT
from pbh_viterbi.paths import SEARCH_RESULTS_DIR
from pbh_viterbi.search.ranking import clopper_pearson, linear_stat
from pbh_viterbi.threshold import is_candidate, load_threshold, nmse_values


def ranking(df, a):
    """R_a of each row from nsigma and nmse_eff (-inf without an isolated candidate)."""
    for column in ("nsigma", "nmse_eff"):
        if column not in df:
            raise SystemExit(f"Column {column!r} missing: these results do not come from the hybrid selection.")
    return linear_stat(pd.to_numeric(df["nsigma"], errors="coerce"), pd.to_numeric(df["nmse_eff"], errors="coerce"), a)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--noise-csv", type=Path, required=True)
    parser.add_argument("--far-limit", type=float, required=True,
                        help="Maximum fraction of noise chunks above the threshold (no default: choose it).")
    parser.add_argument("--signal-csv", type=Path, help="Injected results, only to report the recovered fraction.")
    parser.add_argument("--a", type=float, default=RANKING_NMSE_WEIGHT,
                        help="Weight of -log10(NMSE_eff) (default: config.RANKING_NMSE_WEIGHT).")
    parser.add_argument("--output", type=Path, default=SEARCH_RESULTS_DIR / "threshold_linear.json")
    args = parser.parse_args()

    noise = pd.read_csv(args.noise_csv)
    stat = np.sort(ranking(noise, args.a))[::-1]
    n = len(stat)
    max_false = int(np.floor(args.far_limit * n + 1e-9))
    # Smallest t with at most max_false noise chunks at or above it (candidates satisfy R_a >= t).
    t = float(np.nextafter(stat[max_false], np.inf)) if max_false < n and np.isfinite(stat[max_false]) else None
    if t is None:
        raise SystemExit("Fewer than floor(FAR N) + 1 noise chunks with an isolated candidate: FAR not reachable.")
    k = int(np.sum(stat >= t))
    report = {
        "model": "nsigma_threshold = a log10(nmse_eff) + t  (R_a = nsigma - a log10 nmse_eff >= t)",
        "basis": "log10_nmse",
        "nmse_column": "nmse_eff",
        "degree": 1,
        "coefficients": [args.a, t],
        "intercept": 0.0,
        "a": args.a,
        "t": t,
        "far_limit": args.far_limit,
        "noise_triggers_total": n,
        "false_positives": k,
        "far": k / n,
        "far_ci68": clopper_pearson(k, n, 0.68),
        "far_ci90": clopper_pearson(k, n, 0.90),
        "noise_csv": Path(args.noise_csv).name,
    }
    threshold = {"basis": "log10_nmse", "coefficients": np.array([args.a, t]), "intercept": 0.0,
                 "log_center": 0.0, "log_scale": 1.0, "nmse_column": "nmse_eff"}
    assert int(is_candidate(threshold, nmse_values(threshold, noise), noise["nsigma"]).sum()) == k
    if args.signal_csv:
        signal = pd.read_csv(args.signal_csv)
        recovered = is_candidate(threshold, pd.to_numeric(nmse_values(threshold, signal), errors="coerce"),
                                 pd.to_numeric(signal["nsigma"], errors="coerce"))
        report.update(signal_triggers=len(signal), signals_recovered=int(recovered.sum()),
                      signal_csv=Path(args.signal_csv).name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    load_threshold(args.output)  # round-trip check
    print(f"a = {args.a:g}, t = {t:.4f}: {k}/{n} noise chunks above (FAR {k / n:.4f}, "
          f"68% CI [{report['far_ci68'][0]:.4f}, {report['far_ci68'][1]:.4f}])"
          + (f"; {report['signals_recovered']}/{report['signal_triggers']} signals recovered" if args.signal_csv else ""))
    print(f"Threshold written to {args.output}")


if __name__ == "__main__":
    main()
