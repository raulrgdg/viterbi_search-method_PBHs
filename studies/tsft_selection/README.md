# Tsft-selection study

The study ran before the hybrid selection became the default (v1.2); its code is
used only with `--per-tsft-output` and the `union` Tsft list.

## Why

In the first v1.1 test campaign (packs 1, 13, 25), the heavy signals of pack 25
(𝓜 = 0.018–0.1 M☉) were recovered up to ~40 kpc, against ~65–90 kpc in v1.0;
75 of its 200 injections ended as `failed_third_fit_check`. All 75 had selected
Tsft = 293.25 s with nσ ≈ 2.2–2.45: the noise of pack 25 alone gives nσ = 2.29
at that Tsft. The selection takes the largest nσ over the Tsft list; a heavy
signal at moderate distance has nσ below that at its (short) Tsft, so a noise
fluctuation of a long Tsft wins, the isolation runs on a noise track and fails.
v1.0 (13 Tsft up to 88 s) did not see this.

Background facts (141 noise packs, `calibration/noise_track_power.csv`):
* the nσ tail does not grow with Tsft (99th percentile 2.4–3.3 for every Tsft,
  skewness 0.4–1.05, largest at 30–180 s, not at 260–422 s);
* long Tsft (≥ 150 s) are strongly correlated with each other (ρ ≈ 0.97) but
  weakly with the short ones (ρ = 0.41 between 2.6 and 422 s): they add a
  nearly independent noise trial to the maximum;
* 49 of the 141 noise packs select a Tsft ≥ 150 s.

## Rules compared (`src/pbh_viterbi/search/selection.py`)

| Rule | Selection over the Tsft | Detection statistic |
|---|---|---|
| A | largest nσ, 13 Tsft of the paper | nσ |
| B | largest nσ, 45 Tsft of list 95 (v1.1 default) | nσ |
| C | smallest empirical p-value per Tsft (noise of the other packs) | Gaussian z of p |
| C' | as C, exponential tail above the 90th percentile | Gaussian z of p |
| D | largest nσ among the Tsft whose isolation succeeds | nσ |
| D' | among the 5 largest nσ with isolation, smallest NMSE | nσ |
| E | as D, only Tsft ≤ 1.5 T_lim(fitted 𝓜) | nσ |

A trigger without an isolated candidate (no NMSE) is never detected, as in the
pipeline. p-values leave the chunk's own pack out of the background (the
injections use the same packs as the noise). With 141 packs the empirical p
cannot go below 1/141; C' extrapolates beyond it, C breaks ties by nσ.

Rule A uses the paper Tsft inside the v1.1 pipeline (in-memory SFTs, v1.1 band
and maps), not the v1.0 results: it isolates the effect of the Tsft list.

## Data saved by the run

One search with the 79 Tsft of the three lists (`--tsft-list union`) and
`--per-tsft-output` gives everything every rule needs, so rules are compared on
the same products and new rules can be added without re-running:

* `search/per_tsft_noise.csv`, `search/per_tsft_injected_pack-<p>.csv`: one row
  per chunk and Tsft (`results.PER_TSFT_FIELDNAMES`): raw track power (to
  renormalise nσ offline), nσ, number of steps, window screening, isolation
  status, NMSE (penalised and raw), fitted chirp mass, time span of the
  isolated block, isolation time;
* `search/search_results_*.csv`: the usual rows (largest nσ over the 79 Tsft),
  kept for the injection parameters (signal id, 𝓜, distance, sky,
  polarisation, inclination, time to merger);
* `logs/submit.log`: commit, packs, job split and Condor clusters.

## How to run

The jobs need the repository, the data and the conda environment on a
filesystem shared with the Condor execute nodes (on LDAS CIT: `/scratch/<user>`).

```bash
PBH_CONDA_ENV=/path/to/env PBH_VITERBI_O3_DIR=/path/to/o3 \
    bash studies/tsft_selection/submit_condor.sh            # 141 noise jobs + 9 packs x 40 jobs
python studies/tsft_selection/collect.py                     # merge, check, write collected/*.csv
python studies/tsft_selection/analyse.py --paper-threshold   # rules, efficiencies, figures
```

Default injection packs: 1 2 3 (signals 0–199), 13 14 15 (200–399), 25 26 27
(400–599), i.e. every signal in 3 noise realisations (1800 injections).

## Outputs (`results/tsft_selection/analysis/`)

* `summary.md`, `summary.csv`: reached FAR and efficiency per rule, criterion
  and signal slice, at FAR 3% and 1%;
* `efficiency_by_mass.csv`, `roc.csv`, `selections.csv`;
* `fig_roc`: efficiency vs FAR per rule (all injections and per slice);
* `fig_efficiency_vs_mass`, `fig_d50_vs_mass`: efficiency and distance of 50%
  efficiency per chirp mass, per rule, at FAR 3% and 1% (1D and, with
  `--paper-threshold`, the paper (NMSE, nσ) threshold refitted per rule);
* `fig_selected_tsft`: selected Tsft vs injected 𝓜 with T_lim(𝓜), and the
  share of noise packs selecting Tsft ≥ 150 s, per rule;
* `fig_noise_tail`: noise nσ tail per Tsft.

## Results (run of 2026-10-19, commit e8fe2e3)

141 noise packs and 1800 injections (packs 1–3, 13–15, 25–27), 79 Tsft.
Efficiency over all injections; slices by signal index.

| Rule | 1D, FAR 3% | paper, FAR 3% | paper, FAR 3%, heavy | paper, FAR 1% |
|---|---|---|---|---|
| A  max nσ, 13 paper Tsft | 0.432 | 0.509 | 0.755 | 0.501 |
| B  max nσ, 45 Tsft (v1.1) | 0.436 | 0.488 | 0.672 | 0.483 |
| C  min p-value | 0.428 | 0.487 | 0.665 | 0.470 |
| C' min p-value + tail | 0.433 | 0.489 | 0.672 | 0.482 |
| D  max nσ among isolated | 0.436 | 0.495 | 0.695 | 0.481 |
| **D' top-5 nσ, min NMSE** | 0.427 | **0.516** | **0.758** | **0.514** |
| E  D + T_lim consistency | 0.436 | 0.495 | 0.695 | 0.481 |

Paired comparison (same injections, paper criterion, FAR 3%):
* D' vs B: +63 / −13 injections overall, **+52 / −0 in the heavy slice**
  (McNemar p = 4×10⁻¹⁶); better in each of packs 25, 26, 27.
* D' vs A: +49 / −37 overall (p = 0.24), +29 / −27 heavy (p = 0.89): D'
  recovers the heavy-mass efficiency of the 13-Tsft list and keeps the gain of
  the 45-Tsft list at the lightest masses (2×10⁻⁴ M☉: 0.07 vs 0.03).

Findings:
1. **With nσ alone (1D) the rule does not matter** at FAR ≤ 3%: only signals
   with nσ ≳ 2.5–3 pass, and those win the selection with any rule.
2. The loss of v1.1 appears only with the paper (NMSE, nσ) threshold, which
   accepts weak heavy signals thanks to a near-perfect chirp fit (small NMSE).
   With max nσ, a long-Tsft noise track takes the selection and the fit is lost
   (`fig_selected_tsft`: undetected heavy injections at Tsft 250–420 s).
3. **p-values (C, C') do not help**: the noise tail does not grow with Tsft
   (`fig_noise_tail`), so min p ≈ max nσ; with 141 packs p cannot go below 1/141.
4. **Mass consistency (E) does not help**: the long-Tsft noise tracks that
   isolate are fitted as light masses (long T_lim), so they pass the check.
5. **D' (choose by NMSE among the 5 largest nσ with an isolated candidate) is the
   best rule** and the only one that fixes the heavy-mass loss.

Caveats:
* The paper threshold is fitted and evaluated on the same injections and noise
  packs (optimistic, equally for every rule); with 141 noise packs FAR 1% is
  1 pack, and the paper threshold barely changes between 1% and 3%.
* The gain of D' lives in the low-nσ, small-NMSE region, where 141 noise packs
  can not measure the FAR below 1/141 (see the threshold issues kept aside).
