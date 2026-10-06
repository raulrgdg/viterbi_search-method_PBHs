"""Distance reach of the injected search, d_L,95% (Sec. III B, Figs. 6-7).

For each chirp mass, the injections are ordered by distance and the cumulative
recovered fraction (recovered / injected up to distance d) is computed. The
distance reach is the largest distance before that fraction first drops
below 95%. The 1-sigma uncertainty uses the Wilson score interval of the
cumulative fraction: applying the same rule to its lower (upper) bound gives
the lower (upper) bound of the reach.
"""

import numpy as np
import pandas as pd

RECOVERY_FRACTION = 0.95
WILSON_Z = 1.0  # ~68% interval


def wilson_interval(k, n, z=WILSON_Z):
    """Wilson score interval of a binomial proportion k/n."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    phat = k / n
    denom = 1 + z ** 2 / n
    center = (phat + z ** 2 / (2 * n)) / denom
    halfwidth = (z / denom) * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2))
    return center - halfwidth, center + halfwidth


def first_crossing(distances, fraction, threshold=RECOVERY_FRACTION):
    """Largest distance such that ``fraction >= threshold`` at every distance up to it.

    Returns None if the fraction is already below threshold at the closest distance.
    """
    below = np.flatnonzero(np.asarray(fraction) < threshold)
    if below.size == 0:
        return distances[-1]
    if below[0] == 0:
        return None
    return distances[below[0] - 1]


def distance_reach(mchirp, distance, recovered, threshold=RECOVERY_FRACTION, z=WILSON_Z):
    """d_L,95% per chirp mass with its Wilson bounds.

    Inputs are per-injection arrays. Returns a DataFrame with columns
    ``mchirp``, ``d95``, ``d95_low``, ``d95_high`` (same units as ``distance``);
    chirp masses that never reach the threshold are omitted.
    """
    df = pd.DataFrame({"mchirp": mchirp, "distance": distance, "recovered": np.asarray(recovered, dtype=bool)})
    rows = []
    for mc in sorted(df["mchirp"].unique()):
        grp = (df[df["mchirp"] == mc].groupby("distance")["recovered"]
               .agg(detected="sum", total="count").reset_index().sort_values("distance"))
        d = grp["distance"].to_numpy()
        k_cum = grp["detected"].cumsum().to_numpy()
        n_cum = grp["total"].cumsum().to_numpy()
        frac_low, frac_high = wilson_interval(k_cum, n_cum, z)

        d95 = first_crossing(d, k_cum / n_cum, threshold)
        if d95 is None:
            continue
        upper = first_crossing(d, frac_high, threshold)  # optimistic fraction -> larger distance
        lower = first_crossing(d, frac_low, threshold)   # pessimistic fraction -> smaller distance
        rows.append({
            "mchirp": mc,
            "d95": d95,
            "d95_low": d[0] if lower is None else lower,
            "d95_high": d95 if upper is None else upper,
        })
    return pd.DataFrame(rows)


def pchip_curve(x, y, n=400):
    """Shape-preserving interpolation through (x, y) in log-log space."""
    from scipy.interpolate import PchipInterpolator

    log_x, log_y = np.log10(x), np.log10(y)
    order = np.argsort(log_x)
    interpolator = PchipInterpolator(log_x[order], log_y[order])
    x_dense = np.logspace(log_x.min(), log_x.max(), n)
    return x_dense, 10 ** interpolator(np.log10(x_dense))
