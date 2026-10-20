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
