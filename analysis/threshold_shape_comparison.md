# Threshold shape: linear vs quadratic vs cubic (2026-10-29)

**Conclusion: keep the linear threshold** `R_a = nσ − a log10(NMSE_eff) ≥ t` (a = 2,
`compute_threshold_linear.py`):

1. **The region a curve would open cannot be certified.** The signals a curve recovers sit at
   NMSE_eff < 2×10⁻⁴, where no noise pack falls. With 0 of 141 packs there, the noise
   probability in that region is only bounded to **< 2.1% per chunk** (95%, rule of three, 3/N).
   Accepting it could add up to ~2% of unseen false alarms, more than the 1% target. The line
   keeps a minimum nσ there.
2. **The gain is small:** +0.1 to +0.6 points on the campaign, with the same out-of-fold noise
   counts for every degree.
3. **More injections do not help:** forced to open that region, the curves rotate around the
   highest noise pack and lose at intermediate NMSE what they gain on the left (total 57.1–58.0%
   in every scenario). The cubic just follows the top of the noise cloud.

The limit is the noise background, not the shape: at FAR 1% the threshold is set by one noise
pack. Curved boundaries can be revisited with a much larger background (O3a H1, O3b L1).

## Method

* Boundary nσ ≥ P(log10 NMSE_eff), P of degree 1, 2 or 3, non-decreasing in NMSE_eff.
* Fitted on the calibration set (2000 injections in 10 packs outside the campaign) and the
  141 noise packs: maximise the injections above P with at most ⌊FAR·N⌋ noise packs above
  (differential evolution). Tsft selection fixed (hybrid, a = 2).
* Out of fold: 3 folds of noise packs. Evaluated on the campaign (18000 injections), which
  takes no part in the fit.

## Results

| FAR | Degree | Noise above (out of fold) | Calibration | Campaign | Missed at NMSE_eff < 2e-4 |
|---|---|---|---|---|---|
| 3% | 1 | 4/141 (7/141) | 61.9% | 59.1% | 48 |
| 3% | 2 | 4/141 (7/141) | 61.9% | 59.2% | 29 |
| 3% | 3 | 4/141 (7/141) | 62.1% | 59.7% | 0 |
| 1% | 1 | 1/141 (2/141) | 60.0% | 57.2% | 215 |
| 1% | 2 | 1/141 (2/141) | 60.1% | 57.5% | 178 |
| 1% | 3 | 1/141 (2/141) | 60.6% | 57.7% | 148 |

The free-slope line gives slope 2.1–2.2, consistent with a = 2. The production threshold
(t set on noise) gives 59.5% (FAR 3%) and 56.7% (FAR 1%).

**Simulation** (FAR 1%; campaign injections added to the calibration set, campaign recovery
counted without them):

| Added to the calibration set | Line | Quadratic | Cubic |
|---|---|---|---|
| nothing | 57.2% | 57.5% | 57.7% |
| +50 missed signals of the left region | 57.5% | 57.6% | 58.0% |
| +150 missed signals of the left region | 57.2% | 57.4% | 57.7% |
| +2000 random injections | 57.1% | 57.5% | 57.9% |
| +6600 random + 50 of the left region | 57.6% | 57.6% | 57.9% |

The curves pivot on the highest noise pack (NMSE_eff ≈ 6×10⁻⁴, nσ ≈ 2.7); the cubic takes an
S shape with a plateau at nσ ≈ 3.5 set by a handful of noise packs.

## False-alarm probability of the linear thresholds

t sits just above the (k+1)-th largest of N noise triggers, so k/N is optimistic. For new
noise the exceedance probability is Beta(k+1, N−k) (order statistics, no assumption on the
noise), of mean (k+1)/(N+1):

| Threshold | t | Noise above | Clopper-Pearson 68% | **Expected FAP, mean (68% / 95%)** |
|---|---|---|---|---|
| FAR 3% | 7.407 | 4/141 | [1.5, 5.0]% | **3.5%** ([2.0, 5.0]% / [1.2, 7.1]%) |
| FAR 1% | 8.624 | 1/141 | [0.1, 2.3]% | **1.4%** ([0.5, 2.3]% / [0.2, 3.9]%) |

Out of fold the counts are the same (4/141, 1/141). The FAR 1% threshold rests on one noise
trigger (~±65%): FAR 3% is the main result, FAR 1% secondary, each with its interval.

## Figures and code

`analysis/figures/threshold_shape/`: `fig_missed_signals_plane.png` (recovered and missed
injections with the linear thresholds), `fig_threshold_curve_test.png` (line vs quadratic),
`fig_threshold_sim_linear_quadratic.png` and `fig_threshold_sim_quadratic_cubic.png`
(simulations, FAR 1%).

Only the degree 1–2 fit is kept as a script (`results/plots/campaign_v12/threshold_curve_test.py`,
not tracked); the cubic fit and the simulations were run interactively.
