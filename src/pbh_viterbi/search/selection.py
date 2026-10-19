"""[UNDER TEST] Rules that pick the Tsft of a chunk from its per-Tsft products.

Part of the Tsft-selection study (studies/tsft_selection). The default search
still uses search.power.select_best_tsft (largest nsigma over the Tsft list);
nothing here is called by the pipeline until a rule is adopted.

Every rule reads a per-Tsft table, one row per chunk and Tsft, as written with
``--per-tsft-output`` (results.PER_TSFT_FIELDNAMES): columns pack, signal_id,
injected, tsft, nsigma, iso_status ("ok" or the failure of the candidate
isolation), nmse and mass. ``apply_rule`` returns one row per chunk with the
selected Tsft and the detection statistic ``stat`` of the rule.

Rules (RULES):

  A_max_nsigma_paper   largest nsigma over the 13 Tsft of the paper
  B_max_nsigma_95      largest nsigma over the 45 Tsft of list 95 (current v1.1)
  C_min_p_95           smallest empirical p-value per Tsft (noise background of
                       the other packs); stat = equivalent Gaussian z, ties
                       (p at the floor 1/N) broken by nsigma
  C_min_p_tail_95      as C, with an exponential tail fitted above the 90th
                       percentile of the background (finer p beyond the
                       largest noise value)
  D_isolated_95        largest nsigma among the Tsft whose isolation succeeds
  D_topk_nmse_95       among the 5 Tsft of largest nsigma whose isolation
                       succeeds, the one with the smallest NMSE
  E_mass_consistent_95 as D, keeping only Tsft <= 1.5 T_lim(fitted chirp mass)
                       (the margin covers the drift tolerated by the Tsft
                       lists, (Tsft/T_lim)^2 <= 1.03, and the error of the
                       fitted mass; a noise track at 293 s fitted as 0.1 Msun,
                       T_lim = 2.6 s, is still removed)

The p-values leave the pack of the chunk out of the background, since the
injections are made into the same packs as the noise background.
"""

import numpy as np
import pandas as pd
from scipy import stats

from pbh_viterbi.config import TSFT_LISTS
from pbh_viterbi.search.power import tsft_key
from pbh_viterbi.tsft import tsft_lim

CHUNK_KEYS = ["injected", "pack", "signal_id"]
TAIL_QUANTILE = 0.90
TOP_K = 5
T_LIM_MARGIN = 1.5


def _tsft_set(name):
    return {tsft_key(t) for t in TSFT_LISTS[name]}


def prepare(per_tsft):
    """Typed copy of a per-Tsft table (missing signal_id -> -1 so it can be a group key)."""
    df = per_tsft.copy()
    df["signal_id"] = pd.to_numeric(df["signal_id"], errors="coerce").fillna(-1).astype(int)
    df["tsft"] = df["tsft"].map(tsft_key)
    for column in ("nsigma", "nmse", "mass"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["isolated"] = (df["iso_status"] == "ok") & np.isfinite(df["nmse"]) & (df["nmse"] > 0)
    return df


def empirical_pvalues(df, noise, tail_quantile=None):
    """p-value of every row's nsigma against the noise nsigma of the same Tsft, leaving its pack out.

    p = (#{background >= x} + 1) / (N + 1). With ``tail_quantile``, values above
    that quantile u of the background get p = P(b > u) * exp(-(x - u) / scale),
    with the exponential scale = mean excess over u (its maximum-likelihood
    estimate), so that p keeps decreasing beyond the largest noise value.
    """
    p = np.full(len(df), np.nan)
    for tsft, rows in df.groupby("tsft").groups.items():
        bg = noise[noise["tsft"] == tsft]
        b, b_pack = bg["nsigma"].to_numpy(float), bg["pack"].to_numpy(int)
        sub = df.loc[rows]
        x, x_pack = sub["nsigma"].to_numpy(float), sub["pack"].to_numpy(int)
        keep = b_pack[None, :] != x_pack[:, None]  # leave the chunk's own pack out
        n = keep.sum(axis=1)
        counts = ((b[None, :] >= x[:, None]) & keep).sum(axis=1)
        p_sub = (counts + 1.0) / (n + 1.0)
        if tail_quantile is not None:
            for pack in np.unique(x_pack):
                sel = x_pack == pack
                b_loo = b[b_pack != pack]
                u = np.quantile(b_loo, tail_quantile)
                excess = b_loo[b_loo > u] - u
                if excess.size == 0 or excess.mean() <= 0:
                    continue
                above = sel & (x > u)
                tail = np.mean(b_loo > u) * np.exp(-(x[above] - u) / excess.mean())
                p_sub[above] = np.minimum(p_sub[above], tail)
        p[df.index.get_indexer(rows)] = p_sub
    return p


def _select(df, tsft_set, score, eligible=None):
    """Row of largest ``score`` per chunk among the Tsft of ``tsft_set`` (and ``eligible`` rows)."""
    mask = df["tsft"].isin(tsft_set)
    if eligible is not None:
        mask &= eligible
    cand = df[mask & np.isfinite(score)]
    idx = score[cand.index].groupby([cand[k] for k in CHUNK_KEYS]).idxmax()
    return df.loc[idx.to_numpy()]


def _top_k_by_nsigma(df, tsft_set, k):
    in_set = df[df["tsft"].isin(tsft_set)]
    rank = in_set.groupby(CHUNK_KEYS)["nsigma"].rank(ascending=False, method="first")
    return df.index.isin(rank[rank <= k].index)


def mass_consistent(df):
    """Tsft no longer than T_LIM_MARGIN times the tracking limit T_lim of the fitted chirp mass."""
    mass = df["mass"].to_numpy(float)
    limit = np.full(len(df), np.nan)
    ok = np.isfinite(mass) & (mass > 0)
    limit[ok] = T_LIM_MARGIN * tsft_lim(mass[ok])
    return pd.Series(np.nan_to_num(df["tsft"].to_numpy(float) <= limit, nan=False).astype(bool), index=df.index)


def apply_rule(name, df, noise=None):
    """Selected row per chunk for rule ``name``, with columns ``rule`` and ``stat``.

    ``df`` and ``noise`` are prepared per-Tsft tables (``prepare``); ``noise``
    (the noise chunks) is needed by the p-value rules. Chunks where no Tsft is
    eligible are returned with stat = -inf. The detection statistic ``stat`` is
    nsigma, or the Gaussian z of the p-value for the C rules; it is -inf when
    the selected Tsft has no isolated candidate (no NMSE), as in the pipeline.
    """
    s95, paper = _tsft_set("95"), _tsft_set("paper")
    nsigma = df["nsigma"]
    if name == "A_max_nsigma_paper":
        sel = _select(df, paper, nsigma)
    elif name == "B_max_nsigma_95":
        sel = _select(df, s95, nsigma)
    elif name in ("C_min_p_95", "C_min_p_tail_95"):
        if noise is None:
            raise ValueError(f"Rule {name} needs the noise per-Tsft table")
        p = empirical_pvalues(df, noise, TAIL_QUANTILE if name == "C_min_p_tail_95" else None)
        df = df.assign(pvalue=p)
        # Smallest p; ties (common at the resolution 1/(N+1)) go to the largest nsigma.
        sel = _select(df, s95, pd.Series(-np.log10(p) + 1e-9 * nsigma.to_numpy(), index=df.index))
    elif name == "D_isolated_95":
        sel = _select(df, s95, nsigma, df["isolated"])
    elif name == "D_topk_nmse_95":
        sel = _select(df, s95, -np.log10(df["nmse"].where(df["isolated"])),
                      df["isolated"] & _top_k_by_nsigma(df, s95, TOP_K))
    elif name == "E_mass_consistent_95":
        sel = _select(df, s95, nsigma, df["isolated"] & mass_consistent(df))
    else:
        raise ValueError(f"Unknown rule {name!r}; expected one of {RULES}")

    sel = sel.copy()
    if "pvalue" in sel:
        # Gaussian z of p; equal p (frequent at the empirical floor) ranked by nsigma, as in the selection.
        sel["stat"] = stats.norm.isf(sel["pvalue"]) + 1e-6 * sel["nsigma"]
    else:
        sel["stat"] = sel["nsigma"]
    sel.loc[~sel["isolated"], "stat"] = -np.inf

    chunks = df[CHUNK_KEYS].drop_duplicates()
    out = chunks.merge(sel, on=CHUNK_KEYS, how="left")
    out["stat"] = out["stat"].fillna(-np.inf)
    out["isolated"] = out["isolated"].astype("boolean").fillna(False).astype(bool)
    out["rule"] = name
    return out.reset_index(drop=True)


RULES = (
    "A_max_nsigma_paper",
    "B_max_nsigma_95",
    "C_min_p_95",
    "C_min_p_tail_95",
    "D_isolated_95",
    "D_topk_nmse_95",
    "E_mass_consistent_95",
)


def threshold_at_far(noise_stat, far):
    """Smallest threshold with at most floor(far * N) noise chunks strictly above it.

    Returns ``(threshold, k, N)``: a chunk is detected when stat > threshold, and
    the reached false-alarm ratio is k / N.
    """
    values = np.sort(np.asarray(noise_stat, dtype=float))[::-1]
    n = len(values)
    k = int(np.floor(far * n + 1e-9))
    threshold = values[k] if k < n else -np.inf
    reached = int(np.sum(values > threshold))
    return threshold, reached, n
