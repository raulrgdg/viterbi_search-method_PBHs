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
import time

import numpy as np

from pbh_viterbi.config import (
    N_POWER_WINDOWS,
    NMSE_LEN_ALPHA,
    NMSE_NREF,
    POWER_THRESHOLD_K,
    PROVISIONAL_RANKING_THRESHOLD,
    PROVISIONAL_THRESHOLD_INTERCEPT,
    PROVISIONAL_THRESHOLD_SLOPE,
    RANKING_NMSE_WEIGHT,
    SAT_MAX_STEPS,
    SAT_NMSE_FLOOR,
    SAT_RATE_TOL,
    TOP_N_BLOCKS,
    TSFT_SELECTION,
)
from pbh_viterbi.search.fitting import expand_block, fit_candidate_blocks, fit_significant_blocks
from pbh_viterbi.search.power import as_track, screen_windows, select_best_tsft, total_track_power, tsft_key
from pbh_viterbi.search.ranking import effective_nmse, linear_stat

log = logging.getLogger(__name__)

# Possible values of the ``status`` column, in pipeline order.
STATUS_NO_POWER_WINDOW = "failed_second_power_check"
STATUS_NO_FIT = "failed_third_fit_check"
STATUS_NO_OPTIMAL_BLOCK = "failed_optimal_block_selection"
STATUS_NO_EXPANSION = "failed_block_expansion"
STATUS_NO_ISOLATED = "no_isolated_tsft"  # hybrid selection: no Tsft gives an isolated candidate
STATUS_BELOW_RANKING = "rejected_ranking_threshold"  # hybrid selection: R_a below the provisional threshold
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


def search_candidate(tsft_results, noise_means, noise_stds, method=None):
    """Rank one data chunk given its per-Tsft Viterbi products.

    ``tsft_results`` is a list of dicts as returned by
    ``pbh_viterbi.sft.maps.track_product``. ``method`` (default
    config.TSFT_SELECTION) chooses the Tsft:

      "hybrid"      [UNDER TEST] isolate a candidate on every Tsft and keep the
                    one of largest R_a = nsigma - a log10(NMSE_eff)
      "max_nsigma"  the Tsft of largest nsigma, then isolate (v1.0, v1.1)

    Returns a dict with ``candidate``, ``nmse``, ``nsigma``, ``mass``
    (chirp-mass estimate, Msun), ``tsft``, ``status``, ``ranking_stat`` (R_a),
    ``nmse_eff`` and ``selection``; ``nmse``/``mass`` are None when the
    isolation fails. The hybrid method also returns ``per_tsft``, the rows of
    per_tsft_candidates (one per Tsft).
    """
    method = method or TSFT_SELECTION
    if method == "hybrid":
        return _search_hybrid(tsft_results, noise_means, noise_stds)
    if method != "max_nsigma":
        raise ValueError(f"Unknown Tsft selection {method!r}; expected 'hybrid' or 'max_nsigma'")
    return {**_search_max_nsigma(tsft_results, noise_means, noise_stds), "ranking_stat": None, "nmse_eff": None,
            "selection": "max_nsigma"}


def _search_max_nsigma(tsft_results, noise_means, noise_stds):
    """v1.0/v1.1 search: Tsft of largest nsigma, isolation on its track, provisional linear threshold."""
    opt_tsft, opt_nsigma, _ = select_best_tsft(tsft_results, noise_means, noise_stds)
    selected = next((r for r in tsft_results if tsft_key(r["tsft"]) == opt_tsft), None)
    if selected is None:
        raise ValueError(f"Optimal tsft {opt_tsft} not found in tsft_results")
    log.info("Selected tsft=%g s with nsigma=%.4f", opt_tsft, opt_nsigma)

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


# --------------------------------------------------------------------------
# [UNDER TEST] Hybrid Tsft selection (studies/tsft_selection,
# studies/ranking_statistic) and the per-Tsft products it is built on.
# --------------------------------------------------------------------------

def _search_hybrid(tsft_results, noise_means, noise_stds):
    """Isolate on every Tsft; keep the isolated Tsft of largest R_a = nsigma - a log10(NMSE_eff).

    NMSE_eff is the NMSE with saturated short blocks raised to SAT_NMSE_FLOOR
    (search.ranking.effective_nmse). If no Tsft gives an isolated candidate,
    the Tsft of largest nsigma is reported with status ``no_isolated_tsft``.
    """
    rows = per_tsft_candidates(tsft_results, noise_means, noise_stds)
    best, best_stat = None, -np.inf
    for row in rows:
        if row["iso_status"] != "ok" or not (row["nmse"] is not None and np.isfinite(row["nmse"]) and row["nmse"] > 0):
            continue
        row["nmse_eff"] = float(effective_nmse(row["nmse"], row["mass"], row["block_start_s"], row["block_end_s"],
                                               row["tsft"], SAT_MAX_STEPS, SAT_RATE_TOL, SAT_NMSE_FLOOR))
        stat = float(linear_stat([row["nsigma"]], [row["nmse_eff"]], RANKING_NMSE_WEIGHT)[0])
        if stat > best_stat:
            best, best_stat = row, stat
    base = {"selection": "hybrid", "per_tsft": rows}
    if best is None:
        loudest = max(rows, key=lambda r: r["nsigma"])
        log.info("No isolated candidate on any Tsft (largest nsigma %.4f at tsft=%g s)", loudest["nsigma"],
                 loudest["tsft"])
        return {**base, "candidate": False, "nmse": None, "nsigma": loudest["nsigma"], "mass": None,
                "tsft": loudest["tsft"], "status": STATUS_NO_ISOLATED, "ranking_stat": None, "nmse_eff": None}
    passed = best_stat >= PROVISIONAL_RANKING_THRESHOLD
    status = STATUS_CANDIDATE if passed else STATUS_BELOW_RANKING
    log.info("Selected tsft=%g s: nsigma=%.4f nmse=%.4e (eff %.4e) mass=%.4e R_a=%.3f -> %s", best["tsft"],
             best["nsigma"], best["nmse"], best["nmse_eff"], best["mass"], best_stat, status)
    return {**base, "candidate": bool(passed), "nmse": best["nmse"], "nsigma": best["nsigma"], "mass": best["mass"],
            "tsft": best["tsft"], "status": status, "ranking_stat": best_stat, "nmse_eff": best["nmse_eff"]}


def per_tsft_candidates(tsft_results, noise_means, noise_stds):
    """Isolate a candidate on the track of every Tsft, not only the selected one.

    Returns one dict per Tsft with the raw track power (so nsigma can be
    renormalised offline, e.g. leaving the pack out of the background), its
    nsigma, the isolation outcome and, when the isolation succeeds, the NMSE,
    chirp mass and time span (s from the start of the chunk) of the expanded
    block. ``iso_status`` is "ok" or the failure status of isolate_candidate.
    """
    rows = []
    for product in tsft_results:
        tsft = tsft_key(product["tsft"])
        tic = time.perf_counter()
        track_power = float(total_track_power(product["track_index"], product["power"]))
        top_windows, flag = screen_windows(product["track_index"], product["power"], N_POWER_WINDOWS, TOP_N_BLOCKS,
                                           POWER_THRESHOLD_K)
        status, _windows, _optimal, expanded = isolate_candidate(product["track_index"], product["track_freq"],
                                                                 product["power"], tsft)
        row = {
            "tsft": tsft,
            "track_power": track_power,
            "nsigma": (track_power - noise_means[tsft]) / noise_stds[tsft],
            "n_steps": len(product["track_index"]),
            "n_top_windows": len(top_windows),
            "dominant_window": bool(flag),
            "iso_status": status or "ok",
            "nmse": None, "nmse_raw": None, "mass": None, "block_start_s": None, "block_end_s": None,
        }
        if expanded is not None:
            nmse = float(expanded.get("final_nmse", np.inf))
            if not np.isfinite(nmse):
                nmse_values = np.asarray(expanded.get("best_nmse", []), dtype=float)
                nmse = float(np.min(nmse_values)) if nmse_values.size > 0 else np.inf
            row.update(nmse=nmse, nmse_raw=expanded.get("final_nmse_raw"),
                       mass=float(expanded.get("final_mass", expanded["best_mass"][0])),
                       block_start_s=float(expanded["block_start"]) * tsft,
                       block_end_s=float(expanded["block_end"]) * tsft)
        row["iso_seconds"] = round(time.perf_counter() - tic, 3)
        rows.append(row)
    return rows
