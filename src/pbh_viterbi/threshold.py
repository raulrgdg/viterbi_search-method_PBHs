"""Detection threshold in the (NMSE, nsigma) plane (Sec. III B, Fig. 5).

A trigger is a candidate when

    nsigma >= polyval(coefficients, basis(NMSE)) + intercept

where basis(NMSE) is

    "log10_nmse"       log10(NMSE)                      (paper threshold)
    "scaled_log_nmse"  (log10 NMSE - log_center) / log_scale
    "nmse"             NMSE

The paper threshold is produced by analysis/compute_threshold.py and stored as
JSON in paper_data/calibration/far_threshold.json.
"""

import json
from pathlib import Path

import numpy as np


def basis_values(nmse, basis="log10_nmse", log_center=0.0, log_scale=1.0):
    nmse = np.asarray(nmse, dtype=float)
    if basis == "log10_nmse":
        return np.log10(nmse)
    if basis == "scaled_log_nmse":
        return (np.log10(nmse) - log_center) / log_scale
    if basis == "nmse":
        return nmse
    raise ValueError(f"Unknown polynomial basis: {basis}")


def load_threshold(path):
    """Read a threshold JSON (coefficients, intercept, basis)."""
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "coefficients": np.asarray(report["coefficients"], dtype=float),
        "intercept": float(report.get("intercept", 0.0)),
        "basis": report.get("basis", "nmse"),
        "log_center": float(report.get("log_center", 0.0)),
        "log_scale": float(report.get("log_scale", 1.0)),
    }


def threshold_curve(threshold, nmse):
    """nsigma threshold at the given NMSE values."""
    x = basis_values(nmse, threshold["basis"], threshold["log_center"], threshold["log_scale"])
    return np.polyval(threshold["coefficients"], x) + threshold["intercept"]


def is_candidate(threshold, nmse, nsigma):
    """Boolean mask of triggers on or above the threshold; triggers without a valid NMSE are rejected."""
    nmse = np.asarray(nmse, dtype=float)
    nsigma = np.asarray(nsigma, dtype=float)
    valid = np.isfinite(nmse) & np.isfinite(nsigma) & (nmse > 0)
    mask = np.zeros(nmse.shape, dtype=bool)
    mask[valid] = nsigma[valid] >= threshold_curve(threshold, nmse[valid])
    return mask
