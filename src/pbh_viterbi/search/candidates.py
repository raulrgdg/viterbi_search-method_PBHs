"""Candidate isolation and ranking (Sec. III A).

Given the Viterbi tracks of one data chunk for every Tsft, the search

1. picks the Tsft with the most significant track (nsigma, Eq. 17);
2. splits that track into 8 windows and keeps the two with the largest power
   fraction;
3. fits them against the inspiral model and keeps the window with the lowest
   NMSE (Eqs. 22-23);
4. expands and trims that window to isolate the whole signal and fits it again.

The result is a trigger (nsigma, NMSE) plus the chirp mass of the best fit
(Eq. 25). The ``candidate`` flag uses a provisional linear decision line; the
paper classification applies the FAR = 3% polynomial threshold afterwards
(analysis/compute_threshold.py).
"""

import logging

import numpy as np

from pbh_viterbi.config import (
    N_POWER_WINDOWS,
    NMSE_LEN_ALPHA,
    NMSE_NREF,
    POWER_THRESHOLD_K,
    PROVISIONAL_THRESHOLD_INTERCEPT,
    PROVISIONAL_THRESHOLD_SLOPE,
    TOP_N_BLOCKS,
)
from pbh_viterbi.search.fitting import expand_block, fit_candidate_blocks, fit_significant_blocks
from pbh_viterbi.search.power import as_track, screen_windows, select_best_tsft

log = logging.getLogger(__name__)

# Possible values of the ``status`` column, in pipeline order.
STATUS_NO_POWER_WINDOW = "failed_second_power_check"
STATUS_NO_FIT = "failed_third_fit_check"
STATUS_NO_OPTIMAL_BLOCK = "failed_optimal_block_selection"
STATUS_NO_EXPANSION = "failed_block_expansion"
STATUS_BELOW_THRESHOLD = "rejected_linear_threshold"
STATUS_CANDIDATE = "candidate_found"


def provisional_nsigma_threshold(nmse):
    return PROVISIONAL_THRESHOLD_SLOPE * nmse + PROVISIONAL_THRESHOLD_INTERCEPT


def select_optimal_block(blocks, tsft):
    """Block with the lowest NMSE; within 5% of it, the longer block wins."""
    best_nmse = np.inf
    optimal_block = None
    optimal_length = 0.0

    for block in blocks:
        nmse_values = np.asarray(block.get("best_nmse", []), dtype=float)
        if nmse_values.size == 0:
            continue
        current_nmse = float(np.min(nmse_values))
        if not np.isfinite(current_nmse):
            continue
        length = block["block_end"] * tsft - block["block_start"] * tsft
        if current_nmse < best_nmse or (
            np.isclose(current_nmse, best_nmse, rtol=0.05, atol=1e-6) and length > optimal_length
        ):
            optimal_block, best_nmse, optimal_length = block, current_nmse, length

    return optimal_block


def isolate_candidate(track_index, track_freq, power, tsft, accepted_windows=None):
    """Run stages 2-4 on the track of one Tsft.

    Returns ``(status, top_windows, optimal_block, expanded_block)``; the blocks
    are ``None`` when the corresponding stage fails.
    """
    top_windows, flag = screen_windows(track_index, power, N_POWER_WINDOWS, TOP_N_BLOCKS, POWER_THRESHOLD_K)
    if not top_windows:
        return STATUS_NO_POWER_WINDOW, top_windows, None, None

    track = as_track(track_freq)
    if flag:
        # A dominant window points to a short, loud (high-mass) signal: refit it in 8 sub-windows.
        blocks = fit_significant_blocks(track, tsft, top_windows, n_windows_per_block=8, blocks_in_time=False)
    else:
        blocks = fit_candidate_blocks(track, tsft, top_windows, flag, n_windows_per_block=1, blocks_in_time=False)
    if not blocks:
        return STATUS_NO_FIT, top_windows, None, None

    optimal_block = select_optimal_block(blocks, tsft)
    if optimal_block is None:
        return STATUS_NO_OPTIMAL_BLOCK, top_windows, None, None

    expanded = expand_block(optimal_block, track, tsft=tsft, nmse_nref=NMSE_NREF, nmse_len_alpha=NMSE_LEN_ALPHA,
                            accepted_windows=accepted_windows)
    if expanded is None:
        return STATUS_NO_EXPANSION, top_windows, optimal_block, None
    return None, top_windows, optimal_block, expanded


def search_candidate(tsft_results, noise_means, noise_stds):
    """Rank one data chunk given its per-Tsft Viterbi products.

    ``tsft_results`` is a list of dicts as returned by
    ``pbh_viterbi.sft.maps.process_tsft``. Returns a dict with ``candidate``,
    ``nmse``, ``nsigma``, ``mass`` (chirp-mass estimate, Msun), ``tsft`` and
    ``status``; ``nmse``/``mass`` are None when the isolation fails.
    """
    opt_tsft, opt_nsigma, _ = select_best_tsft(tsft_results, noise_means, noise_stds)
    selected = next((r for r in tsft_results if int(r["tsft"]) == int(opt_tsft)), None)
    if selected is None:
        raise ValueError(f"Optimal tsft {opt_tsft} not found in tsft_results")
    log.info("Selected tsft=%d s with nsigma=%.4f", opt_tsft, opt_nsigma)

    result = {"candidate": False, "nmse": None, "nsigma": opt_nsigma, "mass": None, "tsft": opt_tsft}
    status, _windows, _optimal, expanded = isolate_candidate(
        selected["track_index"], selected["track_freq"], selected["power"], opt_tsft
    )
    if status is not None:
        log.info("No candidate: %s", status)
        return {**result, "status": status}

    nmse = float(expanded.get("final_nmse", np.inf))
    if not np.isfinite(nmse):
        nmse_values = np.asarray(expanded.get("best_nmse", []), dtype=float)
        nmse = float(np.min(nmse_values)) if nmse_values.size > 0 else np.inf
    mass = float(expanded.get("final_mass", expanded["best_mass"][0]))

    passed = opt_nsigma >= provisional_nsigma_threshold(nmse)
    status = STATUS_CANDIDATE if passed else STATUS_BELOW_THRESHOLD
    log.info("nsigma=%.4f nmse=%.4e mass=%.4e -> %s", opt_nsigma, nmse, mass, status)
    return {**result, "candidate": bool(passed), "nmse": nmse, "mass": mass, "status": status}
