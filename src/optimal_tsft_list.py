import argparse
import json

import numpy as np
import pycbc.psd
from scipy.optimize import Bounds, LinearConstraint, fsolve, milp


GMSUN = 1.32712440018e20
C_LIGHT = 299792458
T_SUN = GMSUN / (C_LIGHT**3)


def trapezoid(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x=x)
    return np.trapz(y, x=x)


def fs_from_tsft(tsft, mchirp):
    prefactor = 96.0 * (T_SUN * mchirp) ** (5.0 / 3.0) * (tsft**2) / (5.0 * np.pi)
    return prefactor ** (-3.0 / 11.0) / np.pi


def tsft_from_fs(fs, mchirp):
    return ((5.0 * np.pi / 96.0) ** 0.5) * ((T_SUN * mchirp) ** (-5.0 / 6.0)) * ((np.pi * fs) ** (-11.0 / 6.0))


def nsigma_proxy(tsft, mchirp, freqs, psd):
    fs = fs_from_tsft(tsft, mchirp)
    mask = freqs <= fs
    if not np.any(mask):
        return 0.0
    integrand = 1.0 / (psd[mask] * freqs[mask] ** (7.0 / 3.0))
    return np.sqrt(tsft) * trapezoid(integrand, x=freqs[mask])


def dnsigma_condition(fs, freqs, psd):
    mask = freqs <= fs
    if not np.any(mask):
        return np.inf
    integrand = 1.0 / (psd[mask] * freqs[mask] ** (7.0 / 3.0))
    snr_opt2 = trapezoid(integrand, x=freqs[mask])
    if snr_opt2 <= 0:
        return np.inf
    return 0.5 - (6.0 / 11.0) * integrand[-1] * fs / snr_opt2


def build_psd(flow, fhigh, delta_f):
    flen = int(2 * fhigh / delta_f)
    psd_pycbc = pycbc.psd.aLIGOAdVO3LowT1800545(flen, delta_f, flow / 2)
    freqs_all = np.asarray(list(psd_pycbc.sample_frequencies), dtype=float)
    mask = (freqs_all >= flow) & (freqs_all <= fhigh)
    return freqs_all[mask], np.asarray(list(psd_pycbc), dtype=float)[mask]


def _greedy_cover(valid_matrix):
    n_mchirp = valid_matrix.shape[0]
    uncovered = np.ones(n_mchirp, dtype=bool)
    selected = []

    while np.any(uncovered):
        coverage = np.sum(valid_matrix[uncovered], axis=0)
        best_idx = int(np.argmax(coverage))
        if coverage[best_idx] == 0:
            break
        selected.append(best_idx)
        uncovered &= ~valid_matrix[:, best_idx]

    return sorted(set(selected))


def get_optimal_tsft_list(
    min_nsigma_normed,
    mchirp_min=1e-4,
    mchirp_max=1e-1,
    n_mchirp=500,
    flow=61.1,
    psd_fhigh=4096.0,
    delta_f=0.25,
    fs_guess=128.0,
    n_tsft_candidates=300,
):
    if not (0.0 < min_nsigma_normed <= 1.0):
        raise ValueError("min_nsigma_normed must be in (0, 1].")
    if mchirp_min <= 0.0 or mchirp_max <= mchirp_min:
        raise ValueError("Invalid mchirp range.")
    if psd_fhigh <= flow:
        raise ValueError("psd_fhigh must be larger than flow.")

    freqs, psd = build_psd(flow=flow, fhigh=psd_fhigh, delta_f=delta_f)
    mchirps = np.geomspace(mchirp_min, mchirp_max, n_mchirp)

    fs_opt = float(fsolve(dnsigma_condition, fs_guess, args=(freqs, psd))[0])
    print(f'Optimum f*={fs_opt:.6f}')
    if fs_opt < flow or fs_opt > psd_fhigh:
        raise ValueError(
            f"Optimized f*={fs_opt:.6f} Hz lies outside the PSD range [{flow:.6f}, {psd_fhigh:.6f}] Hz."
        )
    opt_tsfts = tsft_from_fs(fs_opt, mchirps)
    max_nsigma = np.array([nsigma_proxy(opt_tsft, mc, freqs, psd) for opt_tsft, mc in zip(opt_tsfts, mchirps)])

    tsft_min = float(np.min(tsft_from_fs(fs_opt, mchirps)))
    tsft_max = float(np.max(tsft_from_fs(flow, mchirps)))
    tsft_candidates = np.geomspace(tsft_min, tsft_max, n_tsft_candidates)

    nsigma_norm = np.zeros((n_mchirp, n_tsft_candidates), dtype=float)
    for i, mc in enumerate(mchirps):
        values = np.array([nsigma_proxy(tsft, mc, freqs, psd) for tsft in tsft_candidates])
        if max_nsigma[i] > 0.0:
            nsigma_norm[i] = values / max_nsigma[i]

    valid_matrix = nsigma_norm >= min_nsigma_normed

    selected_idx = []
    constraints = LinearConstraint(
        valid_matrix.astype(float),
        lb=np.ones(n_mchirp),
        ub=np.full(n_mchirp, np.inf),
    )
    bounds = Bounds(np.zeros(n_tsft_candidates), np.ones(n_tsft_candidates))
    integrality = np.ones(n_tsft_candidates, dtype=int)

    try:
        result = milp(
            c=np.ones(n_tsft_candidates),
            constraints=constraints,
            bounds=bounds,
            integrality=integrality,
        )
        if result.success and result.x is not None:
            selected_idx = np.where(result.x > 0.5)[0].tolist()
    except Exception:
        selected_idx = []

    if not selected_idx:
        selected_idx = _greedy_cover(valid_matrix)

    return [float(tsft_candidates[idx]) for idx in selected_idx]


def main():
    parser = argparse.ArgumentParser(description="Compute the optimal Tsft list for a minimum normalized nsigma.")
    parser.add_argument("--min-nsigma-normed", type=float, required=True)
    parser.add_argument("--mchirp-min", type=float, default=1e-4)
    parser.add_argument("--mchirp-max", type=float, default=1e-1)
    parser.add_argument("--n-mchirp", type=int, default=500)
    parser.add_argument("--flow", type=float, default=61.1)
    parser.add_argument("--psd-fhigh", type=float, default=4096.0)
    parser.add_argument("--delta-f", type=float, default=0.25)
    parser.add_argument("--fs-guess", type=float, default=128.0)
    parser.add_argument("--n-tsft-candidates", type=int, default=300)
    args = parser.parse_args()

    tsft_list = get_optimal_tsft_list(
        min_nsigma_normed=args.min_nsigma_normed,
        mchirp_min=args.mchirp_min,
        mchirp_max=args.mchirp_max,
        n_mchirp=args.n_mchirp,
        flow=args.flow,
        psd_fhigh=args.psd_fhigh,
        delta_f=args.delta_f,
        fs_guess=args.fs_guess,
        n_tsft_candidates=args.n_tsft_candidates,
    )
    print(json.dumps(tsft_list))


if __name__ == "__main__":
    main()
