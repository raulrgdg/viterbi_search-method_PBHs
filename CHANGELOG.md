# Changelog

## Unreleased [UNDER TEST, branch tsft-selection]

### Changed

* **Hybrid Tsft selection** (default, `config.TSFT_SELECTION = "hybrid"`). A
  candidate is isolated on every Tsft and the chunk keeps the isolated Tsft of
  largest R_a = nσ − a log10(NMSE_eff), the statistic of the detection
  threshold (a = 5). The v1.1 selection (largest nσ, then isolate) is
  `method="max_nsigma"`. Motivation: with 45 Tsft up to 422 s, a noise
  fluctuation of a long Tsft won the nσ selection over weak, well-fitted
  heavy signals (pack 25: 75/200 injections lost). See
  `studies/tsft_selection` and `studies/ranking_statistic`.
* **NMSE_eff**: a saturated short block (≤ 24 SFT steps drifting one remapped
  bin per step, the largest Viterbi jump) is a straight line that noise fits
  with NMSE down to 1e-9; its NMSE is raised to 6.6e-4. The three values come
  from the noise background alone (`config.py`).
* **Linear detection threshold** `analysis/compute_threshold_linear.py`:
  R_a ≥ t, t set on the noise triggers only (FAR k/N with Clopper-Pearson
  intervals). The polynomial of the paper extrapolates below the noise NMSE
  range, can decrease with NMSE and does not generalise (out-of-fold FAR 5.7%
  for a 3% target). Out of fold, the hybrid selection with the linear
  threshold recovers 59.4% (FAR 3%) and 57.0% (FAR 1%) of the study
  injections, against 48.4% and 45.8% for v1.1.
* Result CSVs gain `ranking_stat`, `nmse_eff` and `selection`; new statuses
  `rejected_ranking_threshold` and `no_isolated_tsft`.
* `--per-tsft-output` writes one row per chunk and Tsft (on by default in the
  Condor campaign) so the Tsft selection can be re-analysed offline.

## v1.1.0

### New

* **In-memory SFT generation** (`pbh_viterbi.sft.fast`, default
  `--sft-method fast`). It is equivalent to `lalpulsar_MakeSFTs` (same maps and
  results) and ~100× faster per chunk: 31 s instead of ~55 min for a noise
  search with 13 Tsft. Injections are added to the strain in memory, with no
  temporary frames. `--sft-method makesfts` keeps the paper method.
  See `docs/fast_sft.md`.
* **SFT durations from the tracking limit of the remapped map.** Viterbi can
  follow an inspiral only if Tsft ≤ T_lim ≈ 17.7 s (Mc/10⁻² M☉)^(-5/6). Two
  lists cover 2×10⁻⁴–0.1 M☉ with nσ / max nσ ≥ 0.95 (`--tsft-list 95`,
  45 values, default) or ≥ 0.90 (`--tsft-list 90`, 24 values). The paper list
  stays available as `--tsft-list paper`. See `docs/tsft_optimum.md` and
  `tools/opt_tsft_remapped/`.
* **nσ background shipped for the three lists** (`calibration/`), computed
  with the v1.1 code, so a search runs without a calibration step.
* **Injection parameters stored in the results** (`signal_id`, `ra`, `dec`,
  `pol`, `inclination`, `t_to_merger`), and `analysis/injection_snr.py`
  computes the optimal SNR of every injection afterwards with the PSD of its
  own pack.
* **Single-machine runner** `scripts/run_local.sh` (download, background, noise
  search, injection campaign; resumable). The cluster jobs now request one core.
* **Campaign** defaults to 90 packs, 30 per signal slice (`pbh_viterbi.campaign`),
  and needs no chaining across packs or clusters. The paper meant 30 per slice
  but ran 32, 28 and 30; v1.1 moves packs 80-81 out and 89-90 in (88 shared).
* **33 more O3b chunks** (packs 109–141): the other gap-free 32768 s chunks of
  O3b H1 on GWOSC (`tools/find_gwosc_chunks.py`), for the noise background and
  the noise search (141 chunks, ~1283 h). The nσ background keeps a per-pack
  table (`calibration/noise_track_power.csv`), so more packs can be added with
  `PACKS=... bash scripts/run_local.sh background N`.
* `pbh_viterbi.tsft`: T_lim, the predicted nσ fraction and the coverage of a
  Tsft list.
* `tools/optimal_freq_range.py` rewritten as a brute-force maximisation of
  Eq. (2.30) of arXiv:2401.02314 (O3: 61.1–126.8 Hz). `tools/doppler/`
  quantifies the Doppler drift.

### Fixed

* **One band for every product.** The paper noise search and nσ background
  used 61.1–127.2 Hz, while the injected searches used 126.8 Hz. v1.1 uses
  61.1–126.8 Hz everywhere, and the shipped background is computed with it.
* **Frequency grid of the remapped maps.** It started at the nominal FMIN,
  while the first SFT bin is at round(FMIN·Tsft)/Tsft, up to half a bin away.
  It now starts at the first bin (`sft.maps.first_bin_frequency`). The
  regression tests were updated: statuses are unchanged and chirp masses move
  by < 0.1%.
* **Non-integer Tsft** are handled throughout: background files, Tsft
  selection, output columns.
* **Sky positions** of injections are drawn from a fixed seed and written to
  the output. The paper campaign stored neither.
* **False-alarm threshold.** `analysis/compute_threshold.py` has no default
  FAR any more: `--far-limit` must be chosen. The paper threshold allowed 4 of
  108 noise triggers (3.7%, `--far-limit 0.04`) while the text quotes 3%.

### Removed

* Development folders (`sft-gen-fast/`, validation runs) and reproducible
  outputs (CSV results of the studies); their results are summarised in
  `docs/`.

## v1.0.0

First public release, used for the results of arXiv:2607.18352.
