#!/usr/bin/env python3
"""Detection threshold in the (NMSE, nsigma) plane at a fixed false-alarm ratio (Sec. III B).

The threshold is a polynomial in log10(NMSE), nsigma >= polyval(c, log10 NMSE).
For each degree 0..4 the coefficients are found with differential evolution,
maximising the number of injected triggers above the curve subject to

  * at most a fraction --far-limit of the noise triggers above it,
  * the curve not decreasing for log10(NMSE) in [-2, 1],
  * a small penalty on the distance of the curve above the noise triggers.

Triggers with NMSE > 10 are ignored in the fit. Among the degrees that meet
the FAR limit, the one recovering most signals wins (ties: lower FAR, then
lower degree).

The paper threshold (paper_data/calibration/far_threshold.json) is reproduced with

    python analysis/compute_threshold.py

which allows 4 of the 108 noise triggers above the curve (--far-limit 0.04,
i.e. FAR = 3.7%). Takes ~10-20 min.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

import _setup  # noqa: F401  (adds src/ to sys.path)
from pbh_viterbi.paths import PAPER_DATA_DIR, SEARCH_RESULTS_DIR
from pbh_viterbi.threshold import is_candidate

FIT_MAX_NMSE = 10.0
MAX_DEGREE = 4
MONOTONIC_RANGE = (-2.0, 1.0)  # log10(NMSE) range where the curve must not decrease
PENALTY_SCALE = 1e9
GAP_WEIGHT = 2.0
MONOTONIC_WEIGHT = 1e6
DE_OPTIONS = dict(maxiter=5000, popsize=35, polish=True, seed=0, tol=1e-7, mutation=(0.5, 1.4), recombination=0.85)


def load_triggers(path):
    df = pd.read_csv(path)
    for column in ("nmse", "nsigma"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def fit_degree(degree, x_signal, y_signal, x_noise, y_noise, n_noise_total, far_limit, y_span):
    x_grid = np.linspace(*MONOTONIC_RANGE, 200)

    def objective(coeffs):
        noise_threshold = np.polyval(coeffs, x_noise)
        false_positives = np.count_nonzero(y_noise >= noise_threshold)
        true_positives = np.count_nonzero(y_signal >= np.polyval(coeffs, x_signal))
        slopes = np.diff(np.polyval(coeffs, x_grid)) / np.diff(x_grid)
        monotonic_penalty = MONOTONIC_WEIGHT * np.sum(np.clip(-slopes, 0, None))
        gap_penalty = GAP_WEIGHT * np.mean(np.maximum(noise_threshold - y_noise, 0))
        far_penalty = PENALTY_SCALE * max(0.0, false_positives / n_noise_total - far_limit)
        return -true_positives + far_penalty + monotonic_penalty + gap_penalty

    result = differential_evolution(objective, [(-y_span, y_span)] * (degree + 1), **DE_OPTIONS)
    coeffs = result.x
    false_positives = int(np.count_nonzero(y_noise >= np.polyval(coeffs, x_noise)))
    true_positives = int(np.count_nonzero(y_signal >= np.polyval(coeffs, x_signal)))
    return {"degree": degree, "coeffs": coeffs, "far": false_positives / n_noise_total,
            "false_positives": false_positives, "true_positives": true_positives}


def better(candidate, best):
    if best is None or candidate["true_positives"] > best["true_positives"]:
        return True
    if candidate["true_positives"] < best["true_positives"]:
        return False
    if candidate["far"] < best["far"]:
        return True
    return bool(np.isclose(candidate["far"], best["far"]) and candidate["degree"] < best["degree"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--noise-csv", type=Path,
                        default=PAPER_DATA_DIR / "search_results" / "noise_search_O3b_H1_108packs.csv")
    parser.add_argument("--signal-csv", type=Path,
                        default=PAPER_DATA_DIR / "search_results" / "injected_search_O3b_H1_90packs.csv")
    parser.add_argument("--far-limit", type=float, default=0.04)
    parser.add_argument("--output", type=Path, default=SEARCH_RESULTS_DIR / "far_threshold.json")
    args = parser.parse_args()

    noise = load_triggers(args.noise_csv)
    signal = load_triggers(args.signal_csv)
    n_noise_total = len(noise)  # every analysed noise chunk, including those without a valid NMSE

    signal = signal.dropna(subset=["nmse", "nsigma"])
    noise = noise.dropna(subset=["nmse", "nsigma"])
    signal = signal[signal["nmse"] > 0]
    noise = noise[noise["nmse"] > 0]
    signal_fit = signal[signal["nmse"] <= FIT_MAX_NMSE]
    noise_fit = noise[noise["nmse"] <= FIT_MAX_NMSE]

    y_all = np.concatenate([signal["nsigma"].to_numpy(), noise["nsigma"].to_numpy()])
    y_span = max(float(y_all.max() - y_all.min()), 1.0)
    data = (np.log10(signal_fit["nmse"].to_numpy()), signal_fit["nsigma"].to_numpy(),
            np.log10(noise_fit["nmse"].to_numpy()), noise_fit["nsigma"].to_numpy())

    best = None
    for degree in range(MAX_DEGREE + 1):
        result = fit_degree(degree, *data, n_noise_total, args.far_limit, y_span)
        print(f"degree {degree}: FAR = {result['far']:.4f}, recovered = {result['true_positives']}", flush=True)
        if result["far"] <= args.far_limit and better(result, best):
            best = result
    if best is None:
        raise SystemExit(f"No threshold with FAR <= {args.far_limit} was found.")

    threshold = {"basis": "log10_nmse", "coefficients": best["coeffs"], "intercept": 0.0,
                 "log_center": 0.0, "log_scale": 1.0}
    recovered = int(is_candidate(threshold, signal["nmse"], signal["nsigma"]).sum())
    report = {
        "model": "nsigma_threshold = polyval(coefficients, log10(nmse))",
        "basis": "log10_nmse",
        "degree": best["degree"],
        "coefficients": [float(c) for c in best["coeffs"]],
        "intercept": 0.0,
        "far_limit": args.far_limit,
        "noise_triggers_total": n_noise_total,
        "false_positives": best["false_positives"],
        "far": best["far"],
        "signal_triggers": int(len(signal)),
        "signals_recovered": recovered,
        "noise_csv": Path(args.noise_csv).name,
        "signal_csv": Path(args.signal_csv).name,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Best degree {best['degree']}: {best['false_positives']}/{n_noise_total} noise triggers above "
          f"(FAR {best['far']:.4f}), {recovered}/{len(signal)} signals recovered")
    print(f"Threshold written to {args.output}")


if __name__ == "__main__":
    main()
