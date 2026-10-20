# Threshold review and the Tsft rules across the ranking space [UNDER TEST]

**Status: under test, branch `tsft-selection`.** `analysis/compute_threshold.py`
(the paper threshold) is unchanged; the new tools are in
`src/pbh_viterbi/search/ranking.py` and the `D_topk_nmse_len_95` rule of
`search/selection.py`.

Data: the per-Tsft products of `studies/tsft_selection` (141 noise packs,
1800 injections in packs 1–3, 13–15, 25–27, 79 Tsft).

```bash
python studies/ranking_statistic/analyse.py      # ~10 min, 7 processes; results/ranking_statistic/
```

## Questions

1. Is the threshold construction generalisable, i.e. does its FAR and
   efficiency hold on packs it was not fitted on?
2. Does D' (smallest NMSE among the 5 largest-nσ isolated Tsft) improve on B
   (largest nσ, v1.1) everywhere in the ranking space, or only for one way of
   combining nσ and NMSE?

## Method

* **Ranking-space scan**: R_a = nσ − a log10 NMSE for a = 0 (nσ alone) … ∞
  (NMSE alone); threshold from the noise packs only (at most ⌊FAR·N⌋ above),
  nothing fitted to injections. Paired D'−B with McNemar test and a pack
  bootstrap (noise and injection packs resampled, threshold recomputed).
* **Cross-validation**: 3 folds over noise packs and over injection packs (one
  pack per signal slice per fold); fit on 2, test on the third; pooled
  out-of-fold efficiency and FAR (Clopper-Pearson 68%). Constructions:
  `paper` (polynomial, compute_threshold.py), `monotone` (piecewise linear,
  non-decreasing, flat outside the training-noise NMSE range), `linear` (R_a,
  a chosen on the training folds).
* **D''**: D' counting only blocks of ≥ 32 SFT steps as isolated.

## Results (2026-10-20)

**The noise does reach small NMSE.** Over all Tsft, 22 of 141 noise packs have
a track with NMSE < 1e-7 (down to 1e-9). All 59 noise fits with NMSE < 1e-6
are blocks of 2–24 steps whose drift is exactly one remapped bin per step: the
Viterbi track jumping by its maximum step, i.e. a straight line at T_lim.
Injections with NMSE < 1e-6 have blocks of median 391 steps
(`fig_block_length`).

**1. The paper construction does not generalise; monotone and linear do.**

| FAR target | construction | B out-of-fold (in-sample) | D' out-of-fold (in-sample) | out-of-fold noise above, B / D' |
|---|---|---|---|---|
| 3% | paper | 0.485 (0.488) | 0.487 (0.516) | 5 / **8** of 141 |
| 3% | monotone | 0.474 (0.480) | 0.487 (0.493) | 4 / 4 |
| 3% | linear | 0.484 (0.488) | 0.487 (0.492) | 4 / 5 |
| 1% | paper | 0.471 (0.483) | 0.467 (0.514) | 3 / **5** of 141 |
| 1% | monotone | 0.456 (0.472) | 0.478 (0.481) | 2 / 2 |
| 1% | linear | 0.458 (0.473) | 0.467 (0.472) | 3 / 2 |

The paper polynomial overestimates D' by 3–5 points in-sample and its
out-of-fold FAR is 5.7% (target 3%) and 3.5% (target 1%). The monotone and
linear constructions keep the FAR near target and lose ≤ 1.6 points out of
fold.

**2. D' does not help everywhere** (`fig_ranking_scan`):

* nσ-dominated (a ≤ 0.1): D' = B (−0.8 to +0.2 points).
* balanced (a = 0.2–0.75): **D' > B by +1.7 to +2.8 points at FAR 3%**
  (e.g. +52/−1 injections at a = 0.2, p = 1e-14), mostly heavy masses.
* NMSE-dominated (a ≥ 1.5): **D' < B**, down to −12 points at a = 10. Picking
  the smallest NMSE among 5 Tsft also lowers the NMSE of the noise triggers
  (`fig_plane`: noise at NMSE ~1e-6, nσ ~ 1), so the threshold has to rise.
* D'' (blocks ≥ 32 steps) never loses significantly to B for a ≥ 0.2 and gains
  where NMSE dominates (+3.5 points at a = ∞, FAR 3%), but loses light masses
  (light efficiency 0.135 vs 0.19), whose long-Tsft blocks are short.
* Out of fold, the heavy-mass gain of D' is robust (monotone: +60/−0 at FAR 3%,
  +57/−0 at 1%; linear +51/−0), the total gain is small (+0.3 to +2.2 points).

**3. The 13 paper Tsft are not worse than the 45 of v1.1** once the threshold
is honest: best R_a efficiency at FAR 3% is 0.507 for A (a = 5), 0.488 for B,
0.492 for D' and D''. B only wins at light masses (0.19 vs 0.16); A wins at
heavy masses (0.76 vs 0.67).

## Caveats

* 141 noise packs: FAR resolution 1/141, so FAR 1% is one pack; 9 injection
  packs: the pack bootstrap intervals are wide when NMSE dominates.
* R_a and the monotone curve are two families of thresholds; others (e.g. a
  length-aware NMSE) may rank differently.

## D''': do not trust the NMSE of saturated short blocks (2026-10-20)

**A duration cut does not work.** Block durations of the noise fits with
NMSE < 1e-6 (225–3334 s, median 779 s) overlap those of the heavy signals
with NMSE < 1e-4 (362–3358 s, median 822 s): heavy signals are only in band
for the last minutes before merger. Removing the noise fits needs a cut near
3400 s, which kills the heavy signals (efficiency 0.10); a 400 s cut changes
nothing. A cut in steps (D'') instead kills light signals, whose blocks have
7–14 steps at Tsft ≈ 260 s.

**What separates them is not the shape.** A saturated block (drift of exactly
one remapped bin per step, the largest Viterbi jump) is a straight line that
any fit matches with NMSE ~ 0: 100% of the noise fits with NMSE < 1e-6, but
also 88% of the light signals with NMSE < 1e-4. So D''' keeps these blocks but
does not trust their NMSE: for blocks of ≤ 24 steps (the longest saturated
noise block) with |drift per step − 1| < 1e-3 the NMSE is raised to at least
1e-3, and nσ decides.

| | B | A | D' | D''' |
|---|---|---|---|---|
| best R_a efficiency, FAR 3% (in-sample) | 0.488 | 0.507 | 0.492 | **0.517** |
| linear, out-of-fold, FAR 3% (noise above) | 0.484 (4/141) | 0.504 (5) | 0.487 (5) | **0.515 (4)** |
| linear, out-of-fold, FAR 1% (noise above) | 0.458 (3/141) | 0.496 (2) | 0.467 (2) | **0.515 (1)** |
| paper, out-of-fold, FAR 3% (noise above) | 0.485 (5) | 0.508 (6) | 0.487 (**8**) | **0.518 (5)** |

* D''' − B over the ranking scan: never significantly negative (−0.2 points at
  a = 0, p = 0.45), +1.0 to +1.4 points at a = 0.2–0.75, **+2.9 to +4.6 points
  for a ≥ 1** (McNemar p down to 1e-22); unlike D', it does not collapse when
  NMSE dominates.
* Out of fold (linear threshold, a chosen on the training folds): D''' vs B
  +66/−11 (FAR 3%) and **+108/−5** (FAR 1%), heavy slice +52/−0 and +61/−0;
  D''' vs A: 0.515 vs 0.504 (3%), 0.515 vs 0.496 (1%).
* With D''' the paper polynomial also generalises (out-of-fold 0.518, 5/141),
  since the degenerate noise fits no longer sit at small NMSE.

Caveat: SAT_MAX_STEPS = 24 comes from the noise alone, but the floor 1e-3 was
picked among two values (1e-3, 1e-2) on these injections; the three D'''
parameters should be frozen now and checked on an independent injection set
(e.g. the full campaign).

## Tsft chosen by nσ, by NMSE, or by the ranking statistic (hybrid)

`plot_selection_metric.py` → `fig_selection_metric_vs_mass`,
`fig_selection_metric_vs_distance`, `selection_metric.csv`. All out of fold,
linear threshold on R_a = nσ − a log10 NMSE.

The hybrid chooses, per chunk, the isolated Tsft of largest R_a (NMSE with the
D''' correction), with the **same a as the threshold**, so it adds no
parameter: a = 0 is "largest nσ among isolated Tsft", a → ∞ is "smallest
NMSE" (D''' with K = 45). a is chosen on the training folds (5, 5, 5 at FAR 3%;
5, 3, 5 at FAR 1%).

| Tsft chosen by | FAR 3%: recovered (noise above) | FAR 1% | light 3% / 1% | intermediate | heavy |
|---|---|---|---|---|---|
| largest nσ (B) | 48.4% (4/141) | 45.8% (3/141) | 18.5 / 17.8% | 59.7 / 53.8% | 67.2 / 65.7% |
| smallest NMSE (D''', K = 45) | 57.9% (5/141) | 56.2% (2/141) | 16.5 / 13.5% | 68.7 / 66.7% | 88.7 / 88.5% |
| smallest raw NMSE | 43.3% (4/141) | 40.1% (2/141) | 13.0 / 12.2% | 40.0 / 37.5% | 76.8 / 70.5% |
| **largest R_a (hybrid)** | **59.4% (5/141)** | **57.0% (2/141)** | 18.2 / 16.5% | 68.5 / 66.0% | **91.7 / 88.5%** |

The hybrid recovers the light masses to the level of B at FAR 3% (18.2 vs
18.5%) and most of them at 1% (16.5 vs 17.8%), keeps the gain of the NMSE
choice at intermediate and heavy masses, and has the best total. Nearby light
signals (0.1–0.56 kpc) are still recovered at 87% (FAR 3%) and 83% (1%)
against ~95% for B.

## NMSE_eff from the quantisation of the track (replaces D''', 2026-10-21)

D''' raised the NMSE of saturated short blocks to a fixed floor. It works, but
it needs three tuned values and leaves a column of triggers at the floor in the
ranking plane. The cause has a simpler description: the Viterbi track takes
integer remapped bins. A line rounded to the grid has residuals uniform in
[−½, ½] bin (mean square 1/12), and a ramp of D bins has mean square
displacement D²/3; since NMSE = Σ residual² / Σ displacement²
(`fitting.fit_slope_windows`), no fit can resolve an NMSE below

    NMSE_q = (1/12) / (D²/3) = 1/(4 D²).

The search uses NMSE_eff = NMSE + 1/(4D²), D the drift of the fitted inspiral
across the block (`ranking.quantised_nmse`, `config.NMSE_QUANT_COEFF`). Fitting
quantised ramps gives 0.98–1.4 × 1/(4D²) for D = 10–1000
(`tests/test_ranking.py`). For the longest saturated noise block (D = 24)
1/(4D²) = 4.3e-4, close to the empirical floor 6.6e-4 of D'''; for signal
tracks (D ~ 10²–10³) it is 10⁻⁶–10⁻⁷.

Out of fold, linear threshold (`plot_selection_metric.py`):

| Tsft chosen by | FAR 3% (noise above) | FAR 1% | light / intermediate / heavy (3%) |
|---|---|---|---|
| largest nσ (B, v1.1) | 48.4% (4/141) | 45.8% (3/141) | 18.5 / 59.7 / 67.2% |
| smallest NMSE_eff | 57.3% (3/141) | 55.5% (2/141) | 14.5 / 68.7 / 88.7% |
| **largest R_a, NMSE_eff (hybrid, search default)** | **57.8% (4/141)** | **56.6% (1/141)** | 16.5 / 68.3 / 88.7% |
| largest R_a with the D''' floor (for reference) | 59.4% (5/141) | 57.0% (2/141) | 18.2 / 68.5 / 91.7% |

`fig_plane_hybrid.png` (`plot_plane.py`) shows the plane without the column;
`fig_plane_nmse_eff_stepfloor.png` is the D''' version. A few noise triggers
share an NMSE_eff: exact staircases of the same Tsft and length, whose NMSE is
only the quantisation term.
