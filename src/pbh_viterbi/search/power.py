"""Power accumulated along Viterbi tracks: the nsigma statistic and window screening."""

import logging

import numpy as np
import pandas as pd

from pbh_viterbi.config import DOMINANCE_RATIO, NSIGMA_SELECTION_THRESHOLD, SIGNIFICANT_BLOCK_Z_THRESHOLD

log = logging.getLogger(__name__)


def as_track(track):
    track = np.asarray(track, dtype=float)
    if track.ndim != 1:
        raise ValueError(f"The track must be 1D, got shape {track.shape}")
    return track


def as_power_map(power, n_time):
    """Return the map as [time, frequency], cut to ``n_time`` time steps."""
    power = np.asarray(power, dtype=float)
    if power.ndim != 2:
        raise ValueError(f"The power map must be 2D, got shape {power.shape}")
    if power.shape[0] != n_time and power.shape[1] == n_time:
        power = power.T
    if power.shape[0] < n_time:
        raise ValueError(f"The power map has {power.shape[0]} time steps but the track has {n_time}")
    return power[:n_time]


def window_powers(track_index, power, n_windows):
    """Power of the track in each of ``n_windows`` contiguous windows.

    Returns ``(starts, ends, powers)``; ``powers.sum()`` is the total track power,
    i.e. rho^2_tot of Eq. (15) up to normalisation.
    """
    track_index = as_track(track_index)
    power = as_power_map(power, len(track_index))
    if not 0 < n_windows <= len(track_index):
        raise ValueError("n_windows must be in [1, len(track)]")
    edges = np.linspace(0, len(track_index), n_windows + 1, dtype=int)
    starts, ends = edges[:-1], edges[1:]

    idx = np.clip(np.rint(track_index).astype(np.int64, copy=False), 0, power.shape[1] - 1)
    point_power = power[np.arange(len(idx)), idx]
    return starts, ends, np.add.reduceat(point_power, starts)


def total_track_power(track_index, power):
    """Total power accumulated along a track."""
    return np.sum(window_powers(track_index, power, 1)[2])


def load_noise_background(path):
    """Read the per-Tsft mean and std of the noise track power.

    Returns two dicts ``{tsft: mean}`` and ``{tsft: std}``.
    """
    df = pd.read_csv(path)
    df["tsft"] = df["tsft"].astype(int)
    means = dict(zip(df["tsft"], df["mean_total_power"], strict=False))
    stds = dict(zip(df["tsft"], df["std_total_power"], strict=False))
    return means, stds


def select_best_tsft(tsft_results, noise_means, noise_stds, threshold=NSIGMA_SELECTION_THRESHOLD):
    """Pick the Tsft whose Viterbi track is most significant, Eq. (17).

    nsigma = (P_track - mean_noise) / std_noise is computed for every Tsft and
    the largest value is returned as ``(tsft, nsigma, above_threshold)``.
    """
    best_above = -np.inf
    best_overall = -np.inf
    opt_tsft, opt_nsigma, found_above = 0, -np.inf, False

    for result in tsft_results:
        tsft = int(result["tsft"])
        nsigma = (total_track_power(result["track_index"], result["power"]) - noise_means[tsft]) / noise_stds[tsft]
        log.debug("tsft=%d s: nsigma=%.4f", tsft, nsigma)

        if nsigma > best_overall:
            best_overall = nsigma
            opt_tsft, opt_nsigma = tsft, nsigma
        if nsigma > threshold and nsigma > best_above:
            found_above = True
            best_above = nsigma
            opt_tsft, opt_nsigma = tsft, nsigma

    return opt_tsft, opt_nsigma, found_above


def dominant_window(fractions, best_idx, z_threshold=SIGNIFICANT_BLOCK_Z_THRESHOLD):
    """Whether window ``best_idx`` is the loudest and a robust outlier (median/MAD z-score)."""
    fractions = np.asarray(fractions, dtype=float)
    median = np.median(fractions)
    robust_std = 1.4826 * np.median(np.abs(fractions - median))
    score = (fractions[int(best_idx)] - median) / robust_std
    flag = np.isclose(fractions[int(best_idx)], np.max(fractions)) and score > z_threshold
    return flag, score


def select_top_windows(starts, ends, fractions, n_top, k):
    """Windows whose power fraction exceeds median + k * MAD (stage 1 of the isolation).

    Up to ``n_top`` windows are returned as (start, end, fraction), loudest first;
    only the loudest one is kept if it carries ``DOMINANCE_RATIO`` times the
    power of the second. ``flag`` tells whether the loudest window is a strong
    outlier, typical of short, high-mass signals.
    """
    if n_top <= 0:
        raise ValueError("n_top must be > 0")
    if k < 0:
        raise ValueError("k must be >= 0")

    starts = np.asarray(starts, dtype=int)
    ends = np.asarray(ends, dtype=int)
    fractions = np.asarray(fractions, dtype=float)
    if len(fractions) == 0:
        return [], False

    median = np.median(fractions)
    threshold = median + k * np.median(np.abs(fractions - median))
    windows = [
        (int(starts[i]), int(ends[i]), float(fractions[i]), int(i))
        for i in range(len(fractions))
        if fractions[i] > threshold
    ]
    if not windows:
        return [], False

    windows.sort(key=lambda w: w[2], reverse=True)
    flag, score = dominant_window(fractions, windows[0][3])
    log.debug("Loudest window %d: robust z=%.3f, dominant=%s", windows[0][3], score, flag)

    if n_top >= 2 and len(windows) >= 2 and windows[0][2] >= DOMINANCE_RATIO * windows[1][2]:
        return [windows[0][:3]], flag
    return [w[:3] for w in windows[:n_top]], flag


def screen_windows(track_index, power, n_windows, n_top, k):
    """Split the track into windows and keep the ones carrying most of its power."""
    starts, ends, powers = window_powers(track_index, power, n_windows)
    total = np.sum(powers)
    fractions = np.zeros_like(powers) if total <= 0 else powers / total
    return select_top_windows(starts, ends, fractions, n_top=n_top, k=k)
