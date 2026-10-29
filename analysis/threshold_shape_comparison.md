# Threshold shape: linear vs quadratic vs cubic (2026-10-29)

**Conclusion: keep the linear threshold** `R_a = nσ − a log10(NMSE_eff) ≥ t`
(`compute_threshold_linear.py`, a = 2). Curved boundaries gain at most ~0.5 points of
efficiency, the gain is not supported by the calibration set, and they open regions of the
(NMSE_eff, nσ) plane where 141 noise packs cannot certify the false-alarm ratio.

## Question

In the ranking plane of the v1.2 campaign, signals with low nσ and small NMSE_eff
(NMSE_eff < 2×10⁻⁴, a region the noise never reaches) are missed by the straight line,
mostly at FAR 1% (`fig_missed_signals_plane.png`). Would a quadratic or cubic boundary,
calibrated on injections outside the campaign, recover them?

## Method

* Boundary nσ ≥ P(log10 NMSE_eff), P a polynomial of degree 1, 2 or 3, constrained to be
  non-decreasing in NMSE_eff (a worse fit can never be easier to pass).
* Fitted on the **calibration set** (2000 injections in packs 95, 104, 113, 115, 126, 127,
  128, 131, 134, 140, outside the campaign) and the 141-pack noise search: maximise the
  calibration injections above P with at most ⌊FAR·N⌋ noise packs above it (differential
  evolution). Tsft selection fixed (hybrid, a = 2) for every boundary.
* Out of fold: 3 folds of noise packs, the boundary fitted on 2 and the noise above it counted
  on the third (⌊FAR·N_train⌋ allowed per fold).
* Evaluated on the **campaign** (90 packs, 18000 injections), which takes no part in the fit.

## Results

| FAR | Degree | Noise above | Out-of-fold noise above | Calibration | Campaign | Missed with NMSE_eff < 2e-4 |
|---|---|---|---|---|---|---|
| 3% | 1 (line) | 4/141 | 7/141 | 61.9% | 59.1% | 48 |
| 3% | 2 | 4/141 | 7/141 | 61.9% | 59.2% | 29 |
| 3% | 3 | 4/141 | 7/141 | 62.1% | 59.7% | 0 |
| 1% | 1 (line) | 1/141 | 2/141 | 60.0% | 57.2% | 215 |
| 1% | 2 | 1/141 | 2/141 | 60.1% | 57.5% | 178 |
| 1% | 3 | 1/141 | 2/141 | 60.6% | 57.7% | 148 |

The line fitted here with a free slope gives slope 2.1–2.2, consistent with the calibrated
a = 2; the production threshold (a = 2, t set on noise) gives 59.5% (FAR 3%) and 56.7% (FAR 1%)
on the campaign.

**Simulation with synthetic calibration points** (FAR 1%; real campaign injections added to the
calibration set, campaign recovery counted without them):

| Added to the calibration set | Line | Quadratic | Cubic |
|---|---|---|---|
| nothing | 57.2% | 57.5% | 57.7% |
| +50 missed signals of the left region | 57.5% | 57.6% | 58.0% |
| +150 missed signals of the left region | 57.2% | 57.4% | 57.7% |
| +2000 random injections | 57.1% | 57.5% | 57.9% |
| +6600 random + 50 of the left region | 57.6% | 57.6% | 57.9% |

## Why the linear threshold

1. **Small gain.** Quadratic and cubic recover +0.1 to +0.6 points on the campaign; on the
   calibration set, where the boundary is chosen, the gain is 0 to 12 signals out of 2000.
2. **More injections do not help.** When the curves are forced to open the left region (adding
   its points), they rotate around the highest noise pack (NMSE_eff ≈ 6×10⁻⁴, nσ ≈ 2.7 at
   FAR 1%) and lose at intermediate NMSE what they gain on the left: the total stays within
   57.1–58.0% in every scenario.
3. **The cubic follows the noise outline.** It takes an S shape with a plateau at nσ ≈ 3.5
   along the top of the noise cloud (NMSE_eff 10⁻³–10⁻¹), set by a handful of noise packs.
4. **The regions a curve opens cannot be certified.** No noise pack falls at
   NMSE_eff < 2×10⁻⁴, but with 141 packs this only bounds the noise probability there to
   < 2.1% per chunk (95%, 3/N): accepting that region could add up to ~2% unseen false alarms,
   more than the 1% target. The line keeps a minimum nσ there.
5. **Same false-alarm control, fewer parameters.** All degrees keep the same out-of-fold noise
   counts; the line has one shape parameter (a) with a direct meaning and is the simplest to
   justify.

The limit is the noise background, not the shape: at FAR 1% the threshold is set by one noise
pack. With a larger background (several hundred noise chunks) a curved boundary could be
calibrated and its open regions certified; until then the linear threshold is used, with
FAR 3% as the main result and FAR 1% as secondary, always with its Clopper-Pearson interval.

## False-alarm probability of the linear thresholds

The threshold t is set on the 141 noise packs so that at most k of them are above it. The
fraction seen above, k/N, is optimistic by construction: t is placed just above the
(k+1)-th largest noise trigger. For a new noise chunk, the probability of exceeding t
follows exactly (order statistics, no assumption on the noise distribution) a
Beta(k+1, N−k) distribution, of mean (k+1)/(N+1).

| Threshold (a = 2) | t | Noise above (k/N) | Observed, Clopper-Pearson 68% / 95% | **Expected FAP for new noise**, mean (68% / 95%) | Out of fold (3 folds) |
|---|---|---|---|---|---|
| FAR 3% | 7.407 | 4/141 = 2.84% | [1.49, 5.01]% / [0.78, 7.10]% | **3.5%** ([2.0, 5.0]% / [1.2, 7.1]%) | 4/141 = 2.84% |
| FAR 1% | 8.624 | 1/141 = 0.71% | [0.12, 2.31]% / [0.02, 3.89]% | **1.4%** ([0.5, 2.3]% / [0.2, 3.9]%) | 1/141 = 0.71% |

* The real false-alarm probability per 32768 s chunk is about **3.5% ± 1.5%** for the FAR 3%
  threshold and **1.4% (+0.9/−0.9)%** for the FAR 1% threshold (68% intervals); at 95% they
  can be as large as 7.1% and 3.9%.
* The out-of-fold check (t set on 2/3 of the noise packs, counted on the remaining third)
  reproduces the in-sample counts, so the threshold does not over-fit the noise.
* With 141 noise packs the FAR 1% threshold rests on a single noise trigger: its relative
  uncertainty is large (~±65%). Reporting FAR 3% as the main result and FAR 1% as secondary,
  each with its interval, follows from this.

## Figures (`analysis/figures/threshold_shape/`)

* `fig_missed_signals_plane.png`: recovered and missed campaign injections in the plane, with
  the noise and the linear thresholds at FAR 3% and 1%.
* `fig_threshold_curve_test.png`: line vs quadratic fitted on the calibration set, FAR 3% and 1%.
* `fig_threshold_sim_linear_quadratic.png`, `fig_threshold_sim_quadratic_cubic.png`: boundaries
  fitted in each simulation scenario (FAR 1%).
