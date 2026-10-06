"""Fit of Viterbi tracks against the inspiral model in the (t, f^-8/3) map (Sec. II D).

In the remapped coordinate y = f^-8/3 an inspiral is a straight line whose
slope depends only on the chirp mass, Eq. (5). A track segment is fitted by
scanning a grid of chirp masses, and the normalised mean square error (NMSE,
Eq. 22) of the best slope measures how inspiral-like the segment is.
"""

import logging

import numpy as np

from pbh_viterbi.config import FIT_MASS_MAX, FIT_MASS_MIN, FIT_MASS_SAMPLES

log = logging.getLogger(__name__)

GM_SUN = 1.32712442099e20  # m^3 s^-2
LIGHT_SPEED = 299792458.0  # m s^-1


def slope_of_mass(mchirp):
    """d(f^-8/3)/dt of an inspiral with chirp mass ``mchirp`` (Msun), Eqs. (5)-(6).

    Negative masses map to the opposite slope so that fits can detect tracks
    evolving in the wrong direction.
    """
    mchirp = np.asarray(mchirp, dtype=float)
    k = (96 / 5) * (np.pi ** (8 / 3)) * ((GM_SUN / LIGHT_SPEED ** 3) ** (5 / 3))
    m53 = np.sign(mchirp) * (np.abs(mchirp) ** (5 / 3))
    return (-8 / 3) * k * m53


def split_track_windows(track, n_windows):
    """Edges of ``n_windows`` contiguous, nearly equal windows covering the track."""
    if n_windows <= 0:
        raise ValueError("n_windows must be > 0")
    if n_windows > len(track):
        raise ValueError("n_windows cannot exceed the track length")
    edges = np.linspace(0, len(track), n_windows + 1, dtype=int)
    return edges[:-1], edges[1:]


def mass_grid(n_samples=FIT_MASS_SAMPLES, m_min=FIT_MASS_MIN, m_max=FIT_MASS_MAX, include_negative=True):
    """Log-uniform chirp-mass grid in [m_min, m_max], optionally mirrored to negative masses."""
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")
    if m_min <= 0 or m_max <= 0 or m_min >= m_max:
        raise ValueError("Require 0 < m_min < m_max")
    positive = np.logspace(np.log10(m_min), np.log10(m_max), num=n_samples)
    if not include_negative:
        return positive
    return np.concatenate((-positive[::-1], positive))


def fit_slope_windows(track, tsft, n_windows, mass_samples):
    """Best-fitting chirp mass of each window of a track.

    Within each window the track, shifted to start at zero, is compared with
    lines of slope ``slope_of_mass(m)`` for every candidate mass ``m``.

    Returns ``(starts, ends, best_slope, best_mass, best_nmse)``, one entry per window.
    """
    starts, ends = split_track_windows(track, n_windows)
    y = np.asarray(track, dtype=float)

    candidate_mass = np.asarray(mass_samples, dtype=float)
    candidate_slope = slope_of_mass(candidate_mass)
    best_slope = np.zeros(len(starts), dtype=float)
    best_mass = np.zeros(len(starts), dtype=float)
    best_nmse = np.zeros(len(starts), dtype=float)

    for i, (s, e) in enumerate(zip(starts, ends)):
        xw = np.arange(e - s, dtype=float) * tsft
        yw = y[s:e] - y[s]

        residuals = yw[None, :] - (candidate_slope[:, None] * xw[None, :])
        sse = np.sum(residuals * residuals, axis=1)

        j = int(np.argmin(sse))
        best_slope[i] = candidate_slope[j]
        best_mass[i] = candidate_mass[j]
        best_nmse[i] = sse[j] / (float(np.sum(yw * yw)) + 1e-30)

    return starts, ends, best_slope, best_mass, best_nmse


def _blocks_to_indices(candidate_blocks, tsft, n_samples, blocks_in_time):
    """Convert (start, end, ratio) blocks into clipped, non-empty sample-index intervals."""
    blocks_idx = []
    for block in candidate_blocks:
        start, end = block[:2]
        if blocks_in_time:
            start_idx, end_idx = int(np.rint(float(start) / tsft)), int(np.rint(float(end) / tsft))
        else:
            start_idx, end_idx = int(start), int(end)
        start_idx = max(0, start_idx)
        end_idx = min(n_samples, end_idx)
        if end_idx > start_idx:
            blocks_idx.append((start_idx, end_idx))
    return blocks_idx


def _fit_block(block_track, tsft, n_windows_per_block, mass_samples, force_n_windows=None):
    """fit_slope_windows on one block, with the window count bounded by the block length."""
    n_windows = int(n_windows_per_block)
    if n_windows <= 0:
        raise ValueError("n_windows_per_block must be > 0")
    n_windows = min(n_windows, len(block_track))
    if force_n_windows is not None:
        n_windows = min(len(block_track), int(force_n_windows))
    return fit_slope_windows(block_track, tsft, n_windows, mass_samples)


def fit_candidate_blocks(track, tsft, candidate_blocks, flag, n_windows_per_block, mass_samples=None,
                         blocks_in_time=True):
    """Fit each candidate block and drop the sub-windows with a negative best mass.

    If ``flag`` is set (a dominant power window was found), every block is
    split into 8 sub-windows regardless of ``n_windows_per_block``.

    Returns a list with one dict per kept block: ``block_start``, ``block_end``,
    and per sub-window ``starts``, ``ends`` (global indices), ``best_slope``,
    ``best_mass`` and ``best_nmse``.
    """
    y = np.asarray(track, dtype=float)
    if mass_samples is None:
        mass_samples = mass_grid(include_negative=True)

    kept_blocks = []
    for s, e in _blocks_to_indices(candidate_blocks, tsft, len(y), blocks_in_time):
        block_track = y[s:e]
        starts_l, ends_l, best_slope, best_mass, best_nmse = _fit_block(
            block_track, tsft, n_windows_per_block, mass_samples, force_n_windows=8 if flag else None
        )

        valid = best_mass >= 0
        if not np.any(valid):
            log.debug("Block [%d, %d) discarded: every sub-window has a negative mass.", s, e)
            continue

        kept_blocks.append({
            "block_start": s,
            "block_end": e,
            "starts": (starts_l + s)[valid],
            "ends": (ends_l + s)[valid],
            "best_slope": best_slope[valid],
            "best_mass": best_mass[valid],
            "best_nmse": best_nmse[valid],
        })
    return kept_blocks


def _true_runs(mask):
    """Inclusive (first, last) index pairs of consecutive True runs."""
    runs = []
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def fit_significant_blocks(track, tsft, candidate_blocks, n_windows_per_block, mass_samples=None,
                           blocks_in_time=True):
    """Fit blocks dominated by a short, loud signal (high-mass, fast-evolving inspirals).

    Each block is split into sub-windows. If only some of them have a positive
    best mass, every run of consecutive positive sub-windows is refitted as a
    single window and the run with the lowest NMSE is kept.
    """
    y = np.asarray(track, dtype=float)
    if mass_samples is None:
        mass_samples = mass_grid(include_negative=True)

    kept_blocks = []
    for s, e in _blocks_to_indices(candidate_blocks, tsft, len(y), blocks_in_time):
        block_track = y[s:e]
        starts_l, ends_l, best_slope, best_mass, best_nmse = _fit_block(
            block_track, tsft, n_windows_per_block, mass_samples
        )

        valid = best_mass >= 0
        if not np.any(valid):
            log.debug("Block [%d, %d) discarded: every sub-window has a negative mass.", s, e)
            continue

        if np.all(valid):
            kept_blocks.append({
                "block_start": s,
                "block_end": e,
                "starts": starts_l + s,
                "ends": ends_l + s,
                "best_slope": best_slope,
                "best_mass": best_mass,
                "best_nmse": best_nmse,
            })
            continue

        refits = []
        for i0, i1 in _true_runs(valid):
            local_start, local_end = int(starts_l[i0]), int(ends_l[i1])
            if local_end <= local_start:
                continue
            _gs, _ge, g_slope, g_mass, g_nmse = fit_slope_windows(
                block_track[local_start:local_end], tsft, 1, mass_samples
            )
            if len(g_mass) == 0 or g_mass[0] < 0:
                continue
            refits.append({
                "block_start": s,
                "block_end": e,
                "starts": np.array([local_start + s]),
                "ends": np.array([local_end + s]),
                "best_slope": np.array([g_slope[0]]),
                "best_mass": np.array([g_mass[0]]),
                "best_nmse": np.array([g_nmse[0]]),
            })

        if refits:
            kept_blocks.append(refits[int(np.argmin([r["best_nmse"][0] for r in refits]))])
        else:
            log.debug("Block [%d, %d) discarded after refitting its positive-mass runs.", s, e)

    return kept_blocks


def expand_block(
    optimal_block,
    track,
    tsft,
    expansion_window=0.2,
    trim_expansion_window=0.1,
    local_mass_frac=0.5,
    local_mass_points=200,
    nmse_expand_factor=10.0,
    nmse_expand_floor=0.02,
    nmse_expand_cap=0.30,
    nmse_nref=64,
    nmse_len_alpha=0.5,
    nmse_len_floor=0.01,
    nmse_penalty_eps=1e-3,
    nmse_norm_window_s=1024.0,
    reference_window_s=1024.0,
    reference_mass_points=400,
    trim_max_frac=0.30,
    accepted_windows=None,
):
    """Grow and trim the selected block to isolate the full signal (stage 3, Fig. 4).

    1. Expansion: starting from each window of ``optimal_block``, windows of
       ``expansion_window`` times its size are added to the left and right while
       their fit, restricted to masses within ``local_mass_frac`` of the
       neighbour's, keeps the NMSE below a length-dependent threshold.
    2. Trimming: up to ``trim_max_frac`` of the original block is removed from
       each edge while doing so lowers the NMSE of the block, fitted with masses
       around a reference mass measured at the block centre.
    3. Final fit: the resulting block is fitted with the full positive mass grid.
       The reported NMSE is divided by a length factor computed from the first
       ``nmse_norm_window_s`` seconds, which penalises short candidates.

    If ``accepted_windows`` is a list, the (start, end) of every window accepted
    during the expansion is appended to it (used to draw Fig. 4).

    Returns a dict with the block bounds, the fitted windows, ``final_mass``,
    ``final_slope``, ``final_nmse_raw`` and the penalised ``final_nmse``, or
    ``None`` if no valid window remains.
    """
    y = np.asarray(track, dtype=float)
    n_total = len(y)

    starts = np.asarray(optimal_block["starts"], dtype=int)
    ends = np.asarray(optimal_block["ends"], dtype=int)
    best_mass = np.asarray(optimal_block["best_mass"], dtype=float)
    best_slope = np.asarray(optimal_block["best_slope"], dtype=float)
    best_nmse = np.asarray(
        optimal_block.get("best_nmse", np.full(best_mass.shape, np.inf, dtype=float)), dtype=float
    )

    def nmse_threshold_for_window(base_threshold, window_n):
        len_factor = min(1.0, (max(1, int(window_n)) / float(nmse_nref)) ** float(nmse_len_alpha))
        return float(base_threshold) * max(float(nmse_len_floor), float(len_factor))

    def local_mass_grid(center_mass):
        m0 = float(center_mass)
        return np.linspace(max(1e-30, m0 * (1.0 - local_mass_frac)), m0 * (1.0 + local_mass_frac),
                           int(local_mass_points))

    def quick_reference_mass(block_start, block_end):
        block_center = (int(block_start) + int(block_end)) // 2
        ref_samples = int(max(1, round(float(reference_window_s) / float(tsft))))
        ref_samples = min(ref_samples, max(1, int(block_end) - int(block_start)))
        ref_start = max(int(block_start), block_center - (ref_samples // 2))
        ref_end = min(int(block_end), ref_start + ref_samples)
        ref_start = max(int(block_start), ref_end - ref_samples)
        ref_grid = mass_grid(int(reference_mass_points), FIT_MASS_MIN, FIT_MASS_MAX, include_negative=False)
        _rs, _re, _slope, ref_mass, _nmse = fit_slope_windows(y[ref_start:ref_end], tsft, 1, ref_grid)
        return float(ref_mass[0])

    def block_nmse(block_start, block_end):
        if int(block_end) <= int(block_start):
            return np.inf
        _s, _e, _slope, _mass, nmse = fit_slope_windows(y[int(block_start):int(block_end)], tsft, 1, trim_mass_grid)
        return float(nmse[0])

    original_windows = [
        (int(ws), int(we), float(bs), float(bm), float(bn))
        for ws, we, bs, bm, bn in zip(starts, ends, best_slope, best_mass, best_nmse)
    ]
    windows_all = list(original_windows)

    initial_block_start = int(np.min(starts))
    initial_block_end = int(np.max(ends))
    reference_mass = quick_reference_mass(initial_block_start, initial_block_end)
    trim_mass_grid = np.linspace(
        max(1e-30, reference_mass * (1.0 - local_mass_frac)),
        reference_mass * (1.0 + local_mass_frac),
        int(local_mass_points),
    )

    # 1. Expansion around every seed window.
    for ws, we, _bs, bm, bn in original_windows:
        if np.isfinite(bn):
            nmse_threshold_base = float(np.clip(bn * nmse_expand_factor, nmse_expand_floor, nmse_expand_cap))
        else:
            nmse_threshold_base = float(nmse_expand_cap)
        if bm < 0:
            continue

        half_size = max(1, int(np.rint(expansion_window * (we - ws))))

        left_end = ws
        left_grid = local_mass_grid(bm)
        while True:
            left_start = left_end - half_size
            if left_start < 0:
                break
            _s, _e, left_slope, left_mass, left_nmse = fit_slope_windows(y[left_start:left_end], tsft, 1, left_grid)
            threshold = nmse_threshold_for_window(nmse_threshold_base, max(1, left_end - left_start))
            if left_mass[0] > 0 and left_nmse[0] < threshold:
                windows_all.append((int(left_start), int(left_end), float(left_slope[0]), float(left_mass[0]),
                                    float(left_nmse[0])))
                if accepted_windows is not None:
                    accepted_windows.append((int(left_start), int(left_end)))
                left_grid = local_mass_grid(left_mass[0])
                left_end = left_start
            else:
                break

        right_start = we
        right_grid = local_mass_grid(bm)
        while True:
            right_end = right_start + half_size
            if right_end > n_total:
                break
            _s, _e, right_slope, right_mass, right_nmse = fit_slope_windows(y[right_start:right_end], tsft, 1,
                                                                           right_grid)
            threshold = nmse_threshold_for_window(nmse_threshold_base, max(1, right_end - right_start))
            if 1e-5 < right_mass[0] < 1e-1 and right_nmse[0] < threshold:
                windows_all.append((int(right_start), int(right_end), float(right_slope[0]), float(right_mass[0]),
                                    float(right_nmse[0])))
                if accepted_windows is not None:
                    accepted_windows.append((int(right_start), int(right_end)))
                right_grid = local_mass_grid(right_mass[0])
                right_start = right_end
            else:
                break

    # 2. Trimming of the original block edges.
    left_seed = int(np.argmin(starts))
    right_seed = int(np.argmax(ends))
    left_trim_size = max(1, int(np.rint(trim_expansion_window * max(1, int(ends[left_seed]) - int(starts[left_seed])))))
    right_trim_size = max(1, int(np.rint(trim_expansion_window * max(1, int(ends[right_seed]) - int(starts[right_seed])))))
    max_trim = max(0, int(np.floor(trim_max_frac * max(1, initial_block_end - initial_block_start))))

    trimmed_start = initial_block_start
    trimmed_end = initial_block_end

    current_start, current_end = trimmed_start, trimmed_end
    while current_start + left_trim_size <= current_end:
        if current_start - initial_block_start + left_trim_size > max_trim:
            break
        trim_end = current_start + left_trim_size
        if block_nmse(trim_end, current_end) < block_nmse(current_start, current_end):
            current_start = trim_end
            trimmed_start = current_start
        else:
            break

    current_start, current_end = trimmed_start, trimmed_end
    while current_end - right_trim_size >= current_start:
        if initial_block_end - current_end + right_trim_size > max_trim:
            break
        trim_start = current_end - right_trim_size
        if block_nmse(current_start, trim_start) < block_nmse(current_start, current_end):
            current_end = trim_start
            trimmed_end = current_end
        else:
            break

    def full_block_fit(block_start, block_end):
        grid = mass_grid(int(max(reference_mass_points, local_mass_points)), FIT_MASS_MIN, FIT_MASS_MAX,
                         include_negative=False)
        _s, _e, slope, mass, nmse = fit_slope_windows(y[block_start:block_end], tsft, 1, grid)
        return float(slope[0]), float(mass[0]), float(nmse[0])

    if trimmed_start > initial_block_start or trimmed_end < initial_block_end:
        windows_all = [w for w in windows_all if w[0] >= trimmed_start and w[1] <= trimmed_end]
        if trimmed_end > trimmed_start:
            windows_all.append((int(trimmed_start), int(trimmed_end), *full_block_fit(trimmed_start, trimmed_end)))

    # Keep the minimum-NMSE fit of every distinct (start, end) window.
    by_window = {}
    for ws, we, bs, bm, bn in windows_all:
        if (ws, we) not in by_window or bn < by_window[(ws, we)][2]:
            by_window[(ws, we)] = (bs, bm, bn)

    if not by_window and trimmed_end > trimmed_start:
        by_window[(int(trimmed_start), int(trimmed_end))] = full_block_fit(trimmed_start, trimmed_end)

    if not by_window:
        for ws, we, bs, bm, bn in original_windows:
            if (ws, we) not in by_window or bn < by_window[(ws, we)][2]:
                by_window[(ws, we)] = (bs, bm, bn)

    if not by_window:
        return None

    # 3. Final fit of the isolated block.
    keys = sorted(by_window)
    starts_arr = np.array([k[0] for k in keys], dtype=int)
    ends_arr = np.array([k[1] for k in keys], dtype=int)
    block_start = int(np.min(starts_arr))
    block_end = int(np.max(ends_arr))

    final_grid = mass_grid(FIT_MASS_SAMPLES, 1e-5, 1e-1, include_negative=False)
    _fs, _fe, final_slope, final_mass, final_nmse_raw = fit_slope_windows(y[block_start:block_end], tsft, 1,
                                                                           final_grid)
    final_nmse_raw = float(final_nmse_raw[0])

    norm_window_samples = int(max(1, round(float(nmse_norm_window_s) / float(tsft))))
    final_n_points = int(max(1, min(block_end, block_start + norm_window_samples) - block_start))
    final_len_factor = min(1.0, (final_n_points / float(nmse_nref)) ** float(nmse_len_alpha))
    final_len_factor = max(float(nmse_len_floor), float(final_len_factor))
    final_nmse = final_nmse_raw / max(float(nmse_penalty_eps), float(final_len_factor))

    return {
        "block_start": block_start,
        "block_end": block_end,
        "starts": starts_arr,
        "ends": ends_arr,
        "best_slope": np.array([by_window[k][0] for k in keys], dtype=float),
        "best_mass": np.array([by_window[k][1] for k in keys], dtype=float),
        "best_nmse": np.array([by_window[k][2] for k in keys], dtype=float),
        "final_slope": float(final_slope[0]),
        "final_mass": float(final_mass[0]),
        "final_nmse_raw": final_nmse_raw,
        "final_nmse": final_nmse,
        "final_n_points": final_n_points,
        "final_n_points_total": int(max(1, block_end - block_start)),
        "final_norm_window_samples": norm_window_samples,
        "final_len_factor": final_len_factor,
    }
