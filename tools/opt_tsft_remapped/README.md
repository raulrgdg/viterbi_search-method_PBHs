# Optimal SFT durations in the remapped (t, f^-8/3) map

In the remapped map an inspiral moves at constant speed, and Viterbi (±1 bin
per step) can follow it only if

    Tsft <= T_lim = sqrt(Δτ_band / B) ≈ 17.7 s (Mc / 1e-2 Msun)^(-5/6)   (61.1–126.8 Hz).

Below T_lim, nσ / max nσ ≈ sqrt(Tsft / T_lim); above it, nσ drops to the noise
level. The selected Tsft is therefore the largest one ≤ T_lim. The derivation
and its validation with injections are in `nsigma_derivation.tex` / `.pdf`.

| File | Content |
|---|---|
| `plot_tsft_lim.py` | T_lim vs chirp mass, compared with Eq. (14) of the paper and the Tsft selected in injections |
| `tsft_list_remapped.py` | nσ(Mc, Tsft) map and smallest Tsft list for a given minimum nσ / max |
| `compare_tsft_lists.py` | Sensitivity and computing cost of the 95% and 90% lists (below) |
| `nsigma_derivation.tex` / `.pdf` | Derivation of nσ(Mc, Tsft) and validation with injections |
| `injection_check_dense/` | Check of nσ / max = sqrt(Tsft / T_ref) with the 45-Tsft list: noise background and 45 injections (3 masses × 5 distances × 3 packs); `run.sh` then `analyse.py` |

The first validation run of `nsigma_derivation.tex` (75 injections with the
13 paper Tsft) was done with the v1.0 development tools and is summarised in
the PDF. The Tsft = 1 s and selection-rule tests are summarised in
`docs/tsft_optimum.md`. The pipeline itself uses `pbh_viterbi.tsft` and the
lists in `pbh_viterbi.config.TSFT_LISTS`.

## Tsft lists: sensitivity and cost

Lists for chirp masses 2e-4–0.1 M☉, in multiples of 1/8 s (integer samples at
512 Hz), each guaranteeing a minimum nσ / max for every mass
(`compare_tsft_lists.py`):

| List | N Tsft | nσ / max: worst | nσ / max: mean | Reach: worst | Reach: mean | s / injection | Core-hours, paper campaign | 8 cores | 24 cores |
|---|---|---|---|---|---|---|---|---|---|
| ≥ 0.95 | 45 | 0.947 | 0.978 | 0.94 | 0.975 | 75 | 380 | 47 h | 16 h |
| ≥ 0.90 | 24 | 0.897 | 0.954 | 0.88 | 0.948 | 44 | 220 | 28 h | 9 h |
| Paper (2–88 s) | 13 | 0.44 | 0.85 | 0.39 | 0.83 | 27 | 140 | 17 h | 6 h |

* **Mean over masses.** The mean is over chirp masses log-uniform in the range;
  each mass uses the largest Tsft ≤ T_lim, and nσ / max = sqrt(Tsft / T_lim).
* **Reach.** Converted to distance with nσ ∝ d^-0.87, measured on injections
  (the map uses |SFT|, not |SFT|²), i.e. reach ∝ (nσ / max)^1.15.
* **Cost.** For the paper campaign: 18000 injections plus the noise search and
  noise background on 108 chunks, with the fast SFT generator. Per chunk the
  cost is 8 s (frame reading, injection) plus 1.5 s per Tsft, both measured
  with the 45-Tsft run on this node (shared node, ±30%). For comparison, the
  paper campaign used ~2.1×10⁵ allocated core-hours with MakeSFTs.

**Going from 95% to 90%** halves the number of Tsft (45 → 24) and the cost
(380 → 220 core-hours). The reach loses at most 6% (worst case 0.94 → 0.88)
and 3% on average (0.975 → 0.948), i.e. ~8% of sensitive volume on average.
Both lists are far better than the paper one for light masses: the paper list
stops at 88 s, while T_lim reaches 460 s at 2e-4 M☉.

Lists (s):

* **≥ 0.95:** 2.625, 2.875, 3.125, 3.5, 3.875, 4.25, 4.75, 5.25, 5.875, 6.625,
  7.375, 8.25, 9.25, 10.375, 11.625, 13.125, 14.75, 16.625, 18.75, 21, 23.625,
  26.75, 29.875, 33.75, 38.125, 42.875, 48.125, 54.25, 61.375, 69.25, 78.25,
  88.125, 99.5, 112.375, 126.875, 142.5, 160, 180.75, 204.125, 230.5, 259.625,
  293.25, 331.125, 373.875, 422.25
* **≥ 0.90:** 2.625, 3, 3.75, 4.5, 5.625, 6.875, 8.5, 10.625, 13.375, 16.75,
  20.75, 26, 32.375, 40.875, 50.75, 63.5, 79.5, 99.5, 124.5, 154.5, 193.875,
  238.125, 298.75, 377

Caveats: sqrt(Tsft / T_lim) is accurate for heavy and intermediate masses and
conservative for light ones (`nsigma_derivation.tex`). Tsft longer than 88 s
have only been tested with injections at Mc = 5.6×10⁻⁴ M☉ (dense run).
Measured on the v1.0 maps (nominal FMIN grid); the v1.1 grid fix shifts the maps
by under half a bin and does not change these conclusions.
