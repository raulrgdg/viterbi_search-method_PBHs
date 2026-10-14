#!/usr/bin/env python3
"""nsigma(Mc, Tsft) map and optimal Tsft list for the remapped (t, f^-8/3) search.

Analogue of Fig. 2 of arXiv:2401.02314 and of tools/optimal_tsft_list.py, with
the tracking condition of the remapped map (plot_tsft_lim.py) instead of the
one-linear-bin condition of Eq. (14):

  * the signal drifts s = (Tsft / T_lim)^2 remapped bins per SFT;
  * for s <= 1 Viterbi follows the whole track and nsigma ~ sqrt(Tsft)
    (Eq. 2.21 of 2401.02314: rho^2 fixed, sigma ~ sqrt(N_SFT));
  * for s > s_max it cannot follow it and nsigma drops to ~0.

Normalised per chirp mass, nsigma(Mc, Tsft) / max = sqrt(Tsft / T_lim) if
s <= s_max, else 0. Measured on 1000 injections (sft-gen-fast/fast-test):
nsigma follows sqrt(Tsft) within ~10% for s <= 1 and keeps < 12% of it for
s >= 1.1; s_max = 1.03 keeps Mc = 0.085 at Tsft = 3 s (s = 1.013), as measured.

The Tsft list is the smallest set such that every chirp mass in the range has
one Tsft with normalised nsigma >= --min-nsigma-normed (integer programme, as
in optimal_tsft_list.py). Also reports the coverage of the paper list.

    python tools/opt_tsft_remapped/tsft_list_remapped.py --min-nsigma-normed 0.9
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import Bounds, LinearConstraint, milp  # noqa: E402

from plot_tsft_lim import TSFT_GRID, tsft_eq14, tsft_lim  # noqa: E402

DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "results" / "plots" / "nsigma_map_remapped.png"


def nsigma_normed(mchirp, tsft, fmin, fmax, s_max=1.03):
    """sqrt(Tsft / T_lim) where Viterbi can follow the signal, 0 elsewhere (arrays broadcast)."""
    t_lim = tsft_lim(mchirp, fmin, fmax)
    s = (tsft / t_lim) ** 2
    return np.where(s <= s_max, np.sqrt(np.minimum(tsft / t_lim, 1.0)), 0.0)


def optimal_list(mchirps, candidates, fmin, fmax, threshold, s_max):
    valid = nsigma_normed(mchirps[:, None], candidates[None, :], fmin, fmax, s_max) >= threshold
    if not valid.any(axis=1).all():
        raise ValueError("Some chirp masses cannot be covered: widen the candidate Tsft range.")
    result = milp(c=np.ones(len(candidates)), integrality=np.ones(len(candidates)),
                  bounds=Bounds(0, 1),
                  constraints=LinearConstraint(valid.astype(float), lb=1, ub=np.inf))
    return candidates[result.x > 0.5]


def coverage(tsft_list, mchirps, fmin, fmax, s_max):
    """Best normalised nsigma available at each chirp mass with a given Tsft list."""
    return np.max(nsigma_normed(mchirps[:, None], np.asarray(tsft_list)[None, :], fmin, fmax, s_max), axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-nsigma-normed", type=float, default=0.9)
    parser.add_argument("--fmin", type=float, default=61.1)
    parser.add_argument("--fmax", type=float, default=126.8)
    parser.add_argument("--mchirp-min", type=float, default=2e-4)
    parser.add_argument("--mchirp-max", type=float, default=0.1)
    parser.add_argument("--s-max", type=float, default=1.03, help="Largest drift (bins/step) Viterbi still follows.")
    parser.add_argument("--tsft-max", type=float, default=None,
                        help="Longest Tsft allowed in the list (default: T_lim of the lightest mass).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    mchirps = np.geomspace(args.mchirp_min, args.mchirp_max, 600)
    t_lo = float(tsft_lim(args.mchirp_max, args.fmin, args.fmax))
    t_hi = args.tsft_max or float(tsft_lim(args.mchirp_min, args.fmin, args.fmax))
    candidates = np.unique(np.round(np.geomspace(t_lo * 0.5, t_hi, 400), 2))
    if args.tsft_max:
        mchirps = mchirps[tsft_lim(mchirps, args.fmin, args.fmax) * args.min_nsigma_normed ** 2 <= args.tsft_max]
    selected = optimal_list(mchirps, candidates, args.fmin, args.fmax, args.min_nsigma_normed, args.s_max)

    paper_cov = coverage(TSFT_GRID, mchirps, args.fmin, args.fmax, args.s_max)
    new_cov = coverage(selected, mchirps, args.fmin, args.fmax, args.s_max)
    print(f"Band {args.fmin:g}-{args.fmax:g} Hz, Mc in [{mchirps[0]:.3g}, {mchirps[-1]:.3g}] Msun")
    print(f"Optimal list for normalised nsigma >= {args.min_nsigma_normed}: {len(selected)} Tsft")
    print("  " + ", ".join(f"{t:.3g}" for t in selected))
    print(f"  worst-case coverage {new_cov.min():.3f}, median {np.median(new_cov):.3f}")
    print(f"Paper list ({len(TSFT_GRID)} Tsft): worst-case coverage {paper_cov.min():.3f} "
          f"(at Mc = {mchirps[np.argmin(paper_cov)]:.3g}), median {np.median(paper_cov):.3f}")

    t_axis = np.geomspace(1.0, max(500.0, t_hi), 500)
    grid = nsigma_normed(mchirps[None, :], t_axis[:, None], args.fmin, args.fmax, args.s_max)
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    mesh = ax.pcolormesh(mchirps, t_axis, grid, shading="auto", cmap="viridis", vmin=0, vmax=1, rasterized=True)
    fig.colorbar(mesh, ax=ax, label=r"$n_\sigma(\mathcal{M}, T_{\rm SFT}) \,/\, \max_{T_{\rm SFT}} n_\sigma$")
    ax.plot(mchirps, tsft_lim(mchirps, args.fmin, args.fmax), color="k", lw=1.8,
            label=r"$T_{\rm SFT}^{\rm lim}$ (remapped map)")
    ax.plot(mchirps, tsft_eq14(mchirps, args.fmax), color="w", ls="--", lw=1.4, label=r"$T_{\rm SFT}^{\rm opt}$, Eq. (14)")
    for i, t in enumerate(selected):
        ax.axhline(t, color="r", lw=0.6, alpha=0.8,
                   label=rf"list for $n_\sigma \geq {args.min_nsigma_normed:g}$ max" if i == 0 else None)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Chirp mass $\mathcal{M}$ [$M_\odot$]")
    ax.set_ylabel(r"$T_{\rm SFT}$ [s]")
    ax.set_title(f"Band {args.fmin:g}–{args.fmax:g} Hz", fontsize=11)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=400)
    print(f"Plot written to {args.output}")


if __name__ == "__main__":
    main()
