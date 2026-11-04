# The 1.1–1.6×10⁻³ M☉ gap: Tsft selection or threshold? (2026-11-04)

**Conclusion: the threshold, not the Tsft selection.** At 𝓜 = 1.1–1.6×10⁻³ M☉ the
v1.2 campaign recovers fewer signals than the paper (26–31% vs 27–36%; d95 3.2–5.6 vs
5.6–10 kpc). The chosen Tsft is the right one, and no other choice among the 45 Tsft would
pass more injections. The paper recovered those signals because its polynomial threshold lets
almost any trigger with NMSE < ~3×10⁻³ through, whatever its nσ. On the v1.2 noise that
polynomial would pass 88 of the 141 noise packs. The paper's efficiency in this mass range was
therefore overestimated; v1.2 is not less sensitive there.

Script: `analysis/low_mass_gap.py` (tables in `results/search/campaign_v12/low_mass_gap/`).
Inputs: v1.2 campaign (`injected_campaign_snr.csv`, `per_tsft/`, `per_tsft_noise.csv`,
`search_results_noise.csv`, `threshold_far3.json`, a = 2, t = 7.407) and the paper campaign
(`paper_data/search_results/injected_search_O3b_H1_90packs.csv`, `far_threshold.json`).

## 1. Where the signals are lost

Only at the frontier distance. Closer, both campaigns recover 100%; farther, both are at the
~3% noise floor (one noisy pack per 30). The median nσ is the same in both campaigns:

| 𝓜 (M☉) | d (kpc) | paper | v1.2 FAR 3% | nσ paper | nσ v1.2 |
|---|---|---|---|---|---|
| 1.12e-3 | 3.16 | 1.00 | 0.93 | 4.87 | 6.21 |
| 1.12e-3 | 5.62 | 0.66 | 0.43 | 1.52 | 1.74 |
| 1.59e-3 | 5.62 | 1.00 | 0.83 | 4.47 | 2.98 |
| 1.59e-3 | 10.0 | 0.69 | 0.47 | 0.75 | 0.88 |
| 1.59e-3 | 11.5 | 0.53 | 0.20 | 0.48 | −0.30 |
| 1.59e-3 | 13.3 | 0.50 | 0.20 | 0.38 | −0.17 |

(The two campaigns use different sky draws, so the comparison is statistical, 30 or 32 packs
per cell.)

## 2. The selected Tsft is right

T_lim = 110 s at 1.12×10⁻³ M☉ and 82 s at 1.59×10⁻³ M☉ (`docs/tsft_optimum.md`).

* The hybrid selection picks the largest Tsft below T_lim (99.5 s and 78.25 s) in 70–90% of
  the injections at the closer frontier distances (40–57% at 11.5–13.3 kpc, where most are
  lost anyway, with the rest mostly on the next shorter Tsft), as the √(Tsft/T_lim) law predicts. The paper list (max 88 s) would
  use 88 s and 63 s, i.e. slightly *less* nσ.
* The next Tsft (112.4 s and 88.1 s) is above T_lim and its nσ drops to noise (≈ −0.3): the
  sharp cut-off of the tracking limit.
* **No selection rule would do better.** At every frontier distance the recovered fraction
  equals the fraction of injections for which *any* of the 45 Tsft passes R_a ≥ t
  (e.g. 0.47 and 0.47 at 10 kpc, 0.20 and 0.20 at 11.5 kpc; `recovered_vs_any_tsft.csv`).
* At those distances the best Tsft gives nσ ≈ 0–1.5 with NMSE_eff ≈ 10⁻³–10⁻²·⁵ (well-fitted
  but weak tracks): R_a ≈ 5–7, just below t = 7.41 (`per_tsft_profile.csv`).

## 3. It is not the number of Tsft (trials factor)

Noise and injections re-selected with fewer Tsft, t re-set on the 141 noise packs
(`trials_factor.csv`):

| Tsft set | FAR | t | total | 1.12e-3 | 1.59e-3 | 2e-4 | 5.6e-4 | 0.0355 |
|---|---|---|---|---|---|---|---|---|
| 45 (list 95) | 3% | 7.41 | 0.594 | 0.257 | 0.308 | 0.058 | 0.191 | 0.954 |
| 23 (every 2nd) | 3% | 7.30 | 0.579 | 0.247 | 0.303 | 0.043 | 0.164 | 0.941 |
| 15 (every 3rd) | 3% | 7.21 | 0.569 | 0.238 | 0.318 | 0.026 | 0.166 | 0.922 |
| 32 (≤ 88 s) | 3% | 7.37 | 0.582 | 0.241 | 0.301 | 0.006 | 0.153 | 0.954 |
| 45 (list 95) | 1% | 8.62 | 0.567 | 0.232 | 0.261 | 0.030 | 0.161 | 0.936 |
| 23 (every 2nd) | 1% | 7.58 | 0.571 | 0.241 | 0.283 | 0.042 | 0.161 | 0.937 |
| 15 (every 3rd) | 1% | 7.42 | 0.563 | 0.234 | 0.293 | 0.026 | 0.164 | 0.920 |
| 32 (≤ 88 s) | 1% | 8.62 | 0.559 | 0.226 | 0.261 | 0.000 | 0.146 | 0.936 |

Fewer Tsft lower t slightly, but the gap stays (0.24–0.32 vs 0.27/0.36 in the paper) and the
total gets worse at FAR 3%. (At FAR 1% the 23-Tsft set is 0.4 points higher in total, within
the one-noise-pack instability of that threshold.)

## 4. The cause: where each threshold cuts

The paper polynomial (nσ ≥ P(log₁₀ NMSE), degree 4, fitted on noise *and* the same
injections):

| log₁₀ NMSE | −1 | −2 | −2.2 | −2.5 | −3 |
|---|---|---|---|---|---|
| nσ ≥ | 4.86 | 2.23 | 0.36 | −4.0 | −17.2 |

Below NMSE ≈ 3×10⁻³ almost every trigger passes. Only 5 of the 87 paper noise triggers
(raw NMSE, 13 Tsft, 127.2 Hz band) had NMSE that small, so the fit could afford it. In the paper,
55–75% of the recoveries at the 1.59×10⁻³ M☉ frontier (10–15 kpc) had **nσ < 1.4**, i.e. they
were accepted for their NMSE alone (`paper_polynomial_on_v12.csv`).

The v1.2 linear threshold at log₁₀ NMSE_eff = −3 needs nσ ≥ 1.41, because the v1.2 noise does
reach that region (noise triggers at NMSE_eff ~10⁻³ with nσ up to 2–4, `fig5_trigger_plane`;
lowest noise NMSE_eff 2×10⁻⁴).

**Direct check:** the paper polynomial applied to the v1.2 triggers.

| 𝓜 (M☉) | d (kpc) | paper | v1.2 linear | v1.2 with paper polynomial |
|---|---|---|---|---|
| 1.12e-3 | 5.62 | 0.66 | 0.43 | 0.93 |
| 1.59e-3 | 10.0 | 0.69 | 0.47 | 0.93 |
| 1.59e-3 | 11.5 | 0.53 | 0.20 | 0.93 |
| 1.59e-3 | 13.3 | 0.50 | 0.20 | 0.87 |

It would recover more than the paper, but **88/141 v1.2 noise packs** (91/141 with the raw
NMSE) pass it, a FAR of ~62%. The same failure to generalise was found out of fold on the study
set (`studies/ranking_statistic`: FAR 5.7% for a 3% target); here it is concentrated in one mass
range.

## 5. Where the Tsft does play a role

Indirectly: the hybrid selection looks for low NMSE_eff on every Tsft, in noise as well, so the
v1.2 noise background reaches NMSE_eff ~2×10⁻⁴, whereas the paper noise (one Tsft chosen by nσ)
rarely went below 6×10⁻³. This is what makes the v1.2 FAR honest, and the threshold is set on
that same noise.

## Caveats

* The paper NMSE is the raw NMSE, with 13 Tsft and a 127.2 Hz band for the noise products
  (`paper_data/README.md`), so the two noise distributions are not strictly comparable.
* Recovering these masses would mean accepting triggers with nσ ≲ 1 at NMSE_eff ~10⁻³. That
  region can only be certified with a much larger noise background (O3a H1, O3b L1), the same
  limit found in `analysis/threshold_shape_comparison.md`.

## For the paper

State that the v1.0 efficiency at 1.1–1.6×10⁻³ M☉ came from a threshold that did not hold
its FAR there, not that v1.2 lost sensitivity: same nσ at the same distance, optimal Tsft
selected, and no Tsft rule or Tsft list recovers the difference under a noise-set threshold.
