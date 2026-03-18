import numpy as np
from pathlib import Path

MU_SOLAR = 1.32712442099e20
LIGHT_SPEED = 299792458.0
DEBUG_FIT = True
DEFAULT_MASS_GRID_SAMPLES = 20000
DEFAULT_MASS_MIN = 1e-5
DEFAULT_MASS_MAX = 1e-1


def slope_of_mass(M):
    """Compute physically-motivated slope as a function of chirp mass."""
    M = np.asarray(M, dtype=float)
    k = (96 / 5) * (np.pi ** (8 / 3)) * ((MU_SOLAR / LIGHT_SPEED ** 3) ** (5 / 3))
    # Preserve sign for negative mass samples while keeping real-valued power.
    m53 = np.sign(M) * (np.abs(M) ** (5 / 3))
    slope = (-8 / 3) * k * m53
    return slope


def load_track_txt(path_txt):
    """Load a 1D frequency track from a text file."""
    return np.loadtxt(path_txt)


def split_track_windows(track, n_windows):
    """Split a track into `n_windows` non-overlapping windows."""
    if n_windows <= 0:
        raise ValueError("n_windows debe ser > 0")
    n = len(track)
    if n_windows > n:
        raise ValueError("n_windows no puede ser mayor que la longitud del track")

    edges = np.linspace(0, n, n_windows + 1, dtype=int)
    return edges[:-1], edges[1:]


def mass_grid_samples(n_samples, m_min, m_max, include_negative=True):
    """Build a log-uniform mass grid in ``[m_min, m_max]``."""
    print("usando fit_v2")
    if n_samples <= 0:
        raise ValueError("n_samples debe ser > 0")
    if m_min <= 0 or m_max <= 0 or m_min >= m_max:
        raise ValueError("Se requiere 0 < m_min < m_max")

    pos_grid = np.logspace(np.log10(m_min), np.log10(m_max), num=n_samples)
    if not include_negative:
        return pos_grid
    neg_grid = -pos_grid[::-1]
    return np.concatenate((neg_grid, pos_grid))


def _default_mass_grid(include_negative=True):
    """Build the standard mass grid used by this fitting module."""
    return mass_grid_samples(
        n_samples=DEFAULT_MASS_GRID_SAMPLES,
        m_min=DEFAULT_MASS_MIN,
        m_max=DEFAULT_MASS_MAX,
        include_negative=include_negative,
    )


def _candidate_blocks_to_indices(candidate_blocks, tsft, n_samples, blocks_in_time):
    """Convert candidate blocks into clipped sample-index intervals."""
    blocks_idx = []
    significant_block = False

    for block in candidate_blocks:
        start, end, _ratio = block[:3]
        if blocks_in_time:
            start_idx = int(np.rint(float(start) / tsft))
            end_idx = int(np.rint(float(end) / tsft))
        else:
            start_idx = int(start)
            end_idx = int(end)

        if len(candidate_blocks) == 1:
            significant_block = True

        start_idx = max(0, start_idx)
        end_idx = min(n_samples, end_idx)
        if end_idx <= start_idx:
            continue
        blocks_idx.append((start_idx, end_idx))

    return blocks_idx, significant_block


def _fit_block_windows(block_track, tsft, n_windows_per_block, mass_samples, force_n_windows=None):
    """Run windowed slope fitting on a block with bounded window count."""
    n_windows = int(n_windows_per_block)
    if n_windows <= 0:
        raise ValueError("n_windows_per_block debe ser > 0")
    if n_windows > len(block_track):
        n_windows = len(block_track)
    if force_n_windows is not None:
        n_windows = min(len(block_track), int(force_n_windows))

    return fit_slope_windows(
        track=block_track,
        tsft=tsft,
        n_windows=n_windows,
        mass_samples=mass_samples,
    )


def slope_candidates_from_mass(mass_samples):
    """Map mass samples into slope candidates deterministically."""
    mass_samples = np.asarray(mass_samples, dtype=float)
    return slope_of_mass(mass_samples)


def fit_slope_windows(track, tsft, n_windows, mass_samples):
    """Fit local linear slope per window using a precomputed mass grid.

    Args:
        track: 1D frequency track.
        tsft: Sampling step in seconds.
        n_windows: Number of windows.
        mass_samples: Candidate mass values for slope mapping.

    Returns:
        Tuple ``(starts, ends, best_slope, best_mass, best_nmse)``.
    """
    starts, ends = split_track_windows(track, n_windows)
    y = np.asarray(track, dtype=float)

    candidate_mass = np.asarray(mass_samples, dtype=float)
    candidate_slope = slope_candidates_from_mass(candidate_mass)
    best_slope = np.zeros(len(starts), dtype=float)
    best_mass = np.zeros(len(starts), dtype=float)   
    best_nmse = np.zeros(len(starts), dtype=float)

    for i, (s, e) in enumerate(zip(starts, ends)):
        xw = np.arange(e - s, dtype=float) * tsft
        yw = y[s:e] - y[s]

        # Least-squares objective over candidate slopes.
        r = yw[None, :] - (candidate_slope[:, None] * xw[None, :])
        S = np.sum(r * r, axis=1)

        j = int(np.argmin(S))
        sy2 = float(np.sum(yw * yw))
        denom = sy2 + 1e-30

        best_slope[i] = candidate_slope[j]
        best_mass[i] = candidate_mass[j]
        best_nmse[i] = S[j] / denom

    return starts, ends, best_slope, best_mass, best_nmse

def fit_slope_candidate_blocks(
    track,
    tsft,
    candidate_blocks,
    flag,
    n_windows_per_block,
    mass_samples=None,
    blocks_in_time=True,
):
    """
    Ejecuta fit_slope_windows sobre cada bloque candidato y conserva
    solo los bloques con masas ajustadas no negativas. !!

    Parámetros:
      - candidate_blocks: iterable de pares (start, end). Si blocks_in_time=True, en segundos.
                          Si blocks_in_time=False, en índices de muestra.
      - mass_samples: masas candidatas para mapear a pendientes físicas.
      - n_windows_per_block: cantidad de subventanas por bloque.

    Retorna:
      Lista de dicts (uno por bloque conservado), con:
        block_start, block_end, starts, ends, best_slope, best_mass, best_nmse
      Los índices starts/ends son globales sobre el track original.
    """

    y = np.asarray(track, dtype=float)
    n = len(y)
    if mass_samples is None:
        mass_samples = _default_mass_grid(include_negative=True)

    blocks_idx, significant_block = _candidate_blocks_to_indices(
        candidate_blocks,
        tsft=tsft,
        n_samples=n,
        blocks_in_time=blocks_in_time,
    )

    kept_blocks = []
    for s, e in blocks_idx:
        block_track = y[s:e]
        if len(block_track) == 0:
            continue

        starts_l, ends_l, best_slope, best_mass, best_nmse = _fit_block_windows(
            block_track=block_track,
            tsft=tsft,
            n_windows_per_block=n_windows_per_block,
            mass_samples=mass_samples,
            force_n_windows=8 if flag else None,
        )

        # Descarta solo las subventanas con masa negativa y conserva las demás.
        valid_mask = best_mass >= 0
        if not np.any(valid_mask):
            print("!!! Bloque descartado: todas sus subventanas tienen masa negativa")
            continue
        if not np.all(valid_mask):
            print("!!! Bloque parcialmente filtrado: se descartan subventanas con masa negativa")

        kept_blocks.append({
            "block_start": s,
            "block_end": e,
            "starts": (starts_l + s)[valid_mask],
            "ends": (ends_l + s)[valid_mask],
            "best_slope": best_slope[valid_mask],
            "best_mass": best_mass[valid_mask],
            "best_nmse": best_nmse[valid_mask],
        })
    return kept_blocks, significant_block



def fit_slope_candidate_significant__blocks(
    track,
    tsft,
    candidate_blocks,
    n_windows_per_block,
    mass_samples=None,
    blocks_in_time=True,
):
    """Run per-block slope fitting and keep only non-negative mass solutions.

    Args:
        track: Full input track.
        tsft: Sampling step in seconds.
        candidate_blocks: Iterable of tuples ``(start, end, ratio)``.
        n_windows_per_block: Number of fit windows inside each candidate block.
        mass_samples: Optional custom mass grid.
        blocks_in_time: Whether block bounds are in seconds.

    Returns:
        Tuple ``(kept_blocks, significant_block)``.
    """

    def _true_runs(mask):
        """Return inclusive index ranges for consecutive True runs."""
        runs = []
        i = 0
        nmask = len(mask)
        while i < nmask:
            if mask[i]:
                j = i
                while j + 1 < nmask and mask[j + 1]:
                    j += 1
                runs.append((i, j))
                i = j + 1
            else:
                i += 1
        return runs

    y = np.asarray(track, dtype=float)
    n = len(y)

    if mass_samples is None:
        mass_samples = _default_mass_grid(include_negative=True)

    blocks_idx, significant_block = _candidate_blocks_to_indices(
        candidate_blocks,
        tsft=tsft,
        n_samples=n,
        blocks_in_time=blocks_in_time,
    )

    kept_blocks = []

    for s, e in blocks_idx:
        block_track = y[s:e]
        if len(block_track) == 0:
            continue

        starts_l, ends_l, best_slope, best_mass, best_nmse = _fit_block_windows(
            block_track=block_track,
            tsft=tsft,
            n_windows_per_block=n_windows_per_block,
            mass_samples=mass_samples,
        )

        starts_l = np.asarray(starts_l)
        ends_l = np.asarray(ends_l)
        best_slope = np.asarray(best_slope)
        best_mass = np.asarray(best_mass)
        best_nmse = np.asarray(best_nmse)

        valid_mask = best_mass >= 0

        if not np.any(valid_mask):
            if DEBUG_FIT:
                print("Discarded block: all window masses are negative.")
            continue

        if np.all(valid_mask):
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

        if DEBUG_FIT:
            print(valid_mask)
            print("Partially valid block: refitting contiguous valid groups.")

        true_groups = _true_runs(valid_mask)

        refit_candidates = []

        for i0, i1 in true_groups:
            local_start = int(starts_l[i0])
            local_end = int(ends_l[i1])
            if DEBUG_FIT:
                print("Refit group:", local_start * tsft, local_end * tsft)

            if local_end <= local_start:
                continue

            group_track = block_track[local_start:local_end]
            if len(group_track) == 0:
                continue

            g_starts, g_ends, g_slope, g_mass, g_nmse = fit_slope_windows(
                track=group_track,
                tsft=tsft,
                n_windows=1,
                mass_samples=mass_samples,
            )
            if DEBUG_FIT:
                print("Refit NMSE:", g_nmse)

            g_starts = np.asarray(g_starts)
            g_ends = np.asarray(g_ends)
            g_slope = np.asarray(g_slope)
            g_mass = np.asarray(g_mass)
            g_nmse = np.asarray(g_nmse)

            if len(g_mass) == 0 or g_mass[0] < 0:
                continue

            refit_candidates.append({
                "block_start": s,
                "block_end": e,
                "starts": np.array([local_start + s]),
                "ends": np.array([local_end + s]),
                "best_slope": np.array([g_slope[0]]),
                "best_mass": np.array([g_mass[0]]),
                "best_nmse": np.array([g_nmse[0]]),
            })

        if not refit_candidates:
            if DEBUG_FIT:
                print("Discarded block after refit: no valid subgroup survived.")
            continue

        # Keep the best (minimum NMSE) refit candidate.
        best_idx = int(np.argmin([cand["best_nmse"][0] for cand in refit_candidates]))
        kept_blocks.append(refit_candidates[best_idx])

    return kept_blocks, significant_block

def expansion_block(
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
):
    """Expand an optimal block left/right with local constrained refits.

    Args:
        optimal_block: Block dictionary from previous fit stage.
        track: Full input track.
        tsft: Sampling step in seconds.
        expansion_window: Relative expansion size per iteration.
        min_expansion_samples: Minimum samples per expansion window.
        local_mass_frac: Local mass search span around center mass.
        local_mass_points: Number of local mass samples.
        nmse_expand_factor: Expansion threshold multiplier.
        nmse_expand_floor: Minimum expansion threshold.
        nmse_expand_cap: Maximum expansion threshold.
        nmse_nref: Reference sample count for NMSE length correction.
        nmse_len_alpha: Exponent in length correction term.
        nmse_len_floor: Floor for length correction term.
        nmse_penalty_eps: Numerical floor for denominator safety.
        nmse_norm_window_s: Duration used for normalized NMSE window.
        reference_window_s: Duration used to estimate the reference mass.
        reference_mass_points: Mass grid size for the quick reference fit.
        trim_max_frac: Maximum fraction of the original block trim per side.

    Returns:
        Expanded block dictionary or ``None`` if expansion fails.
    """
    y = np.asarray(track, dtype=float)
    n_total = len(y)
    
    starts = np.asarray(optimal_block["starts"], dtype=int)
    ends = np.asarray(optimal_block["ends"], dtype=int)
    best_mass = np.asarray(optimal_block["best_mass"], dtype=float)
    best_slope = np.asarray(optimal_block["best_slope"], dtype=float)
    best_nmse = np.asarray(
        optimal_block.get("best_nmse", np.full(best_mass.shape, np.inf, dtype=float)),
        dtype=float,
    )

    def nmse_threshold_for_window(base_threshold, window_n):
        window_n = max(1, int(window_n))
        len_factor = min(1.0, (window_n / float(nmse_nref)) ** float(nmse_len_alpha))
        len_factor = max(float(nmse_len_floor), float(len_factor))
        return float(base_threshold) * float(len_factor)

    def quick_reference_mass(block_start, block_end):
        block_center = (int(block_start) + int(block_end)) // 2
        ref_window_samples = int(max(1, round(float(reference_window_s) / float(tsft))))
        ref_window_samples = min(ref_window_samples, max(1, int(block_end) - int(block_start)))
        ref_start = max(int(block_start), block_center - (ref_window_samples // 2))
        ref_end = min(int(block_end), ref_start + ref_window_samples)
        ref_start = max(int(block_start), ref_end - ref_window_samples)

        ref_mass_grid = mass_grid_samples(
            n_samples=int(reference_mass_points),
            m_min=DEFAULT_MASS_MIN,
            m_max=DEFAULT_MASS_MAX,
            include_negative=False,
        )
        _rs, _re, _ref_slope, ref_mass, _ref_nmse = fit_slope_windows(
            track=y[ref_start:ref_end],
            tsft=tsft,
            n_windows=1,
            mass_samples=ref_mass_grid,
        )
        return float(ref_mass[0])

    def block_nmse_with_reference(block_start, block_end):
        if int(block_end) <= int(block_start):
            return np.inf
        _st, _en, _blk_slope, _blk_mass, blk_nmse = fit_slope_windows(
            track=y[int(block_start):int(block_end)],
            tsft=tsft,
            n_windows=1,
            mass_samples=trim_mass_grid,
        )
        return float(blk_nmse[0])

    original_windows = []
    for ws, we, bs, bm, bn in zip(starts, ends, best_slope, best_mass, best_nmse):
        original_windows.append((int(ws), int(we), float(bs), float(bm), float(bn)))

    windows_all = list(original_windows)
    seeds = list(original_windows)
    if DEBUG_FIT:
        print("Seeds:", seeds)
    initial_block_start = int(np.min(starts))
    initial_block_end = int(np.max(ends))
    reference_mass = quick_reference_mass(initial_block_start, initial_block_end)
    trim_mass_grid = np.linspace(
        max(1e-30, reference_mass * (1.0 - local_mass_frac)),
        reference_mass * (1.0 + local_mass_frac),
        int(local_mass_points),
    )

    for ws, we, _bs, bm, bn in seeds:
        if np.isfinite(bn):
            nmse_threshold_base = float(np.clip(bn * nmse_expand_factor, nmse_expand_floor, nmse_expand_cap))
        else:
            nmse_threshold_base = float(nmse_expand_cap)
        if (bm < 0):
            if DEBUG_FIT:
                print("Optimal block has negative mass; skipping expansion.")
            continue

        base_size = we - ws
        half_size_current = max(1, int(np.rint(expansion_window * base_size)))
        half_size = half_size_current

        def local_mass_grid(center_mass):
            m0 = float(center_mass)
            m_min_local = max(1e-30, m0 * (1.0 - local_mass_frac))
            m_max_local = m0 * (1.0 + local_mass_frac)
            return np.linspace(m_min_local, m_max_local, int(local_mass_points))

        left_end = ws
        left_grid = local_mass_grid(bm)
        while True:
            left_start = left_end - half_size
            if left_start < 0:
                break
            _st, _en, left_slope, left_mass, left_nmse = fit_slope_windows(
                track=y[left_start:left_end],
                tsft=tsft,
                n_windows=1,
                mass_samples=left_grid,
            )
            left_n = max(1, left_end - left_start)
            left_nmse_threshold = nmse_threshold_for_window(nmse_threshold_base, left_n)
            if DEBUG_FIT:
                print("Left NMSE:", left_nmse)
            if left_mass[0] > 0 and left_nmse[0] < left_nmse_threshold:
                windows_all.append(
                    (
                        int(left_start),
                        int(left_end),
                        float(left_slope[0]),
                        float(left_mass[0]),
                        float(left_nmse[0]),
                    )
                )
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
            _st, _en, right_slope, right_mass, right_nmse = fit_slope_windows(
                track=y[right_start:right_end],
                tsft=tsft,
                n_windows=1,
                mass_samples=right_grid,
            )
            right_n = max(1, right_end - right_start)
            right_nmse_threshold = nmse_threshold_for_window(nmse_threshold_base, right_n)
            if DEBUG_FIT:
                print("Right mass:", right_mass[0])
                print("Right NMSE:", right_nmse)
            if right_mass[0] > 0 and right_mass[0] < 1e-1 and right_mass[0] > 1e-5 and right_nmse[0] < right_nmse_threshold:
                windows_all.append(
                    (
                        int(right_start),
                        int(right_end),
                        float(right_slope[0]),
                        float(right_mass[0]),
                        float(right_nmse[0]),
                    )
                )
                right_grid = local_mass_grid(right_mass[0])
                right_start = right_end
            else:
                break

    left_seed_idx = int(np.argmin(starts))
    right_seed_idx = int(np.argmax(ends))
    

    left_trim_half_size = max(1, int(np.rint(trim_expansion_window * max(1, int(ends[left_seed_idx]) - int(starts[left_seed_idx])))))

    right_trim_half_size = max(1, int(np.rint(trim_expansion_window * max(1, int(ends[right_seed_idx]) - int(starts[right_seed_idx])))))

    left_expanded_globally = any(w[0] < initial_block_start for w in windows_all)
    right_expanded_globally = any(w[1] > initial_block_end for w in windows_all)
    trimmed_block_start = initial_block_start
    trimmed_block_end = initial_block_end
    max_left_trim = max(0, int(np.floor(trim_max_frac * max(1, initial_block_end - initial_block_start))))
    max_right_trim = max(0, int(np.floor(trim_max_frac * max(1, initial_block_end - initial_block_start))))

    current_start = trimmed_block_start
    current_end = trimmed_block_end
    while current_start + left_trim_half_size <= current_end:
        if current_start - initial_block_start + left_trim_half_size > max_left_trim:
            break
        trim_start = current_start
        trim_end = current_start + left_trim_half_size
        nmse_with_window = block_nmse_with_reference(current_start, current_end)
        nmse_without_window = block_nmse_with_reference(trim_end, current_end)
        if DEBUG_FIT:
            print("Left block NMSE with window:", nmse_with_window)
            print("Left block NMSE without window:", nmse_without_window)
        if nmse_without_window < nmse_with_window:
            current_start = trim_end
            trimmed_block_start = current_start
        else:
            break

    current_start = trimmed_block_start
    current_end = trimmed_block_end
    while current_end - right_trim_half_size >= current_start:
        if initial_block_end - current_end + right_trim_half_size > max_right_trim:
            break
        trim_start = current_end - right_trim_half_size
        trim_end = current_end
        nmse_with_window = block_nmse_with_reference(current_start, current_end)
        nmse_without_window = block_nmse_with_reference(current_start, trim_start)
        if DEBUG_FIT:
            print("Right block NMSE with window:", nmse_with_window)
            print("Right block NMSE without window:", nmse_without_window)
        if nmse_without_window < nmse_with_window:
            current_end = trim_start
            trimmed_block_end = current_end
        else:
            break

    if trimmed_block_start > initial_block_start or trimmed_block_end < initial_block_end:
        windows_all = [
            w for w in windows_all
            if w[0] >= trimmed_block_start and w[1] <= trimmed_block_end
        ]
        if trimmed_block_end > trimmed_block_start:
            trimmed_mass_grid = mass_grid_samples(
                n_samples=int(max(reference_mass_points, local_mass_points)),
                m_min=DEFAULT_MASS_MIN,
                m_max=DEFAULT_MASS_MAX,
                include_negative=False,
            )
            _ts, _te, trimmed_slope_arr, trimmed_mass_arr, trimmed_nmse_arr = fit_slope_windows(
                track=y[trimmed_block_start:trimmed_block_end],
                tsft=tsft,
                n_windows=1,
                mass_samples=trimmed_mass_grid,
            )
            windows_all.append(
                (
                    int(trimmed_block_start),
                    int(trimmed_block_end),
                    float(trimmed_slope_arr[0]),
                    float(trimmed_mass_arr[0]),
                    float(trimmed_nmse_arr[0]),
                )
            )

    # Deduplicate windows and keep the minimum-NMSE entry per (start, end).
    by_window = {}
    for ws, we, bs, bm, bn in windows_all:
        key = (ws, we)
        if key not in by_window or bn < by_window[key][2]:
            by_window[key] = (bs, bm, bn)

    if not by_window and trimmed_block_end > trimmed_block_start:
        trimmed_mass_grid = mass_grid_samples(
            n_samples=int(max(reference_mass_points, local_mass_points)),
            m_min=DEFAULT_MASS_MIN,
            m_max=DEFAULT_MASS_MAX,
            include_negative=False,
        )
        _ts, _te, trimmed_slope_arr, trimmed_mass_arr, trimmed_nmse_arr = fit_slope_windows(
            track=y[trimmed_block_start:trimmed_block_end],
            tsft=tsft,
            n_windows=1,
            mass_samples=trimmed_mass_grid,
        )
        by_window[(int(trimmed_block_start), int(trimmed_block_end))] = (
            float(trimmed_slope_arr[0]),
            float(trimmed_mass_arr[0]),
            float(trimmed_nmse_arr[0]),
        )

    if not by_window:
        for ws, we, bs, bm, bn in original_windows:
            key = (ws, we)
            if key not in by_window or bn < by_window[key][2]:
                by_window[key] = (bs, bm, bn)

    if not by_window:
        return None

    sorted_keys = sorted(by_window)
    starts_arr = np.array([k[0] for k in sorted_keys], dtype=int)
    ends_arr = np.array([k[1] for k in sorted_keys], dtype=int)
    slope_arr = np.array([by_window[k][0] for k in sorted_keys], dtype=float)
    mass_arr = np.array([by_window[k][1] for k in sorted_keys], dtype=float)
    nmse_arr = np.array([by_window[k][2] for k in sorted_keys], dtype=float)
    block_start = int(np.min(starts_arr))
    block_end = int(np.max(ends_arr))

    m_min_final, m_max_final = 1e-5, 1e-1

    final_mass_grid = mass_grid_samples(
        n_samples=DEFAULT_MASS_GRID_SAMPLES,
        m_min=m_min_final,
        m_max=m_max_final,
        include_negative=False,
    )
    _fs, _fe, final_slope_arr, final_mass_arr, final_nmse_arr = fit_slope_windows(
        track=y[block_start:block_end],
        tsft=tsft,
        n_windows=1,
        mass_samples=final_mass_grid,
    )
    final_slope = float(final_slope_arr[0])
    final_mass = float(final_mass_arr[0])
    final_nmse_raw = float(final_nmse_arr[0])

    final_n_points_total = int(max(1, block_end - block_start))
    # Normalize NMSE using only an initial time window in sample space.
    norm_window_samples = int(max(1, round(float(nmse_norm_window_s) / float(tsft))))
    norm_window_end = block_start + norm_window_samples
    final_n_points = int(max(1, min(block_end, norm_window_end) - block_start))

    final_len_factor = min(1.0, (final_n_points / float(nmse_nref)) ** float(nmse_len_alpha))
    final_len_factor = max(float(nmse_len_floor), float(final_len_factor))
    final_len_factor_safe = max(float(nmse_penalty_eps), float(final_len_factor))
    final_nmse = final_nmse_raw / final_len_factor_safe

    return {
        "block_start": block_start,
        "block_end": block_end,
        "starts": starts_arr,
        "ends": ends_arr,
        "best_slope": slope_arr,
        "best_mass": mass_arr,
        "best_nmse": nmse_arr,
        "final_slope": final_slope,
        "final_mass": final_mass,
        "final_nmse_raw": final_nmse_raw,
        "final_nmse": final_nmse,
        "final_n_points": final_n_points,
        "final_n_points_total": final_n_points_total,
        "final_norm_window_samples": norm_window_samples,
        "final_len_factor": final_len_factor,
    }


if __name__ == "__main__":
    path_track = str(
        Path(__file__).resolve().parents[1]
        / "outputs"
        / "data"
        / "tracks-frequencies_remote"
        / "signal"
        / "track-freqs_Tsft-16_pack-3_mc-1e-02_dl-0_110.txt"
    )
    tsft = 16
    n_windows = 16
    n_mass_grid = 20000
    m_min = 1e-5
    m_max = 1e-1
    include_negative = True

    track = load_track_txt(path_track)
    mass_samples = mass_grid_samples(
        n_samples=n_mass_grid,
        m_min=m_min,
        m_max=m_max,
        include_negative=include_negative,
    )
    track= track[1280:1536]
    starts, ends, best_slope, best_mass, best_nmse = fit_slope_windows(
        track=track,
        tsft=tsft,
        n_windows=1,
        mass_samples=mass_samples,
    )

    print(f"Ajuste por ventana con grid logspace: y_local = m*x_local, tsft={tsft}s")
    print(
        f"Bounds de masa: [{m_min:.1e}, {m_max:.1e}], "
        f"grid_points={len(mass_samples)}, include_negative={include_negative}"
    )
    for s, e, m, M, nmse in zip(starts, ends, best_slope, best_mass, best_nmse):
        t0 = s * tsft
        t1 = e * tsft
        print(f"[{t0}:{t1}] m_best={m:.6e}  M_best={M:.6e}  NMSE={nmse:.6e}")
