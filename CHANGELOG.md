# Changelog

## v1.3.0

### Changed

* **Default maps: chunk high-pass and per-bin + per-SFT normalisation.** Running the
  search with no options now uses `--sft-highpass chunk --map-normalisation bin+sft`
  (`config.SFT_HIGHPASS`, `config.MAP_NORMALISATION`). The v1.2 maps are
  `--sft-highpass segment --map-normalisation bin`.
  * *Chunk high-pass:* the strain of a chunk is high-passed once, instead of every SFT
    segment as `lalpulsar_MakeSFTs` does. With the rectangular window, the
    per-segment filter transient leaked broadband power into the band: the noise
    floor was ×1.5–6 above the PSD in H1 and ×3–17 in L1. With the chunk high-pass it
    is 0.85–0.95 of the PSD.
  * *Per-SFT normalisation:* after the per-bin normalisation, every SFT is divided by
    its median over the band. This flattens the broadband glitches that the Viterbi
    track followed in noise (packs 52 and 27: R ≈ 9 → 0.5).
  * In a quick test (36 signals × 10 packs, threshold on 61 noise packs) the
    efficiency at FAR ~3% goes from 33% with the v1.2 maps to 73%. See
    `studies/multi_detector`.
* **One nσ background per map configuration.** The file name carries a suffix for the
  detectors, high-pass and normalisation (`paths.config_suffix`): `_chunkhp_sftnorm`
  for the v1.3 default, none for the v1.2 maps, so the v1.1–v1.2 files keep their
  names. `analysis/noise_background_stats.py` takes the same `--detectors`,
  `--sft-highpass` and `--map-normalisation` options to choose its table and output.
* **nσ background of the v1.3 maps shipped** for every Tsft list
  (`calibration/noise_background_tsft<list>_chunkhp_sftnorm.csv`), computed on the
  141 O3b packs, with its per-pack table (`noise_track_power_chunkhp_sftnorm.csv`).
  An H1+L1 background is not shipped: compute it on the 61 packs with L1.
* **Errors instead of silent failures.**
  * A search stops if its nσ background file is missing, naming the configuration and
    how to compute it.
  * The map normalisation stops on non-finite strain, and on SFTs with zero median
    (zero-padded frames) that `bin+sft` cannot normalise.
  * `--sft-method makesfts` asks for `--sft-highpass segment`.
* The paper figures, the cost benchmark and the Tsft-list check keep the v1.2 maps,
  so they reproduce their published results.

### New

* **H1+L1 search** (`--detectors H1 L1`, in-memory SFTs only).
  * Each detector's map is normalised on its own; the maps are summed, divided by √N
    and tracked once.
  * Injections project the same h₊, hₓ on each detector.
  * Both detectors need data on the same chunk: 61 of the 141 chunks
    (`calibration/l1_packs.txt`).
  * Quick test: 89% efficiency at FAR ~3% with the v1.3 maps.
* `o3.download --ifo L1` downloads L1 frames into the same pack folders, and warns
  about frames that come short or with non-finite samples.
* `injected_search --signal-list <csv>` injects a list of signals (signal_id, chirp
  mass, distance) instead of the population grid, and `--signal-ids` picks some
  signals of either.

### To do before a v1.3 campaign

* *a* of R = nσ − a log₁₀ NMSE_eff (2 in v1.2) and the threshold must be recalibrated
  with the v1.3 maps; the quick test prefers a = 1.

## v1.2.0

### Changed

* **Hybrid Tsft selection** (default, `config.TSFT_SELECTION = "hybrid"`). A
  candidate is isolated on every Tsft and the chunk keeps the isolated Tsft of
  largest R_a = nσ − a log10(NMSE_eff), the statistic of the detection
  threshold (a = 2, calibrated at FAR 1% on 10 packs outside the campaign,
  `studies/a_calibration`). The v1.1 selection (largest nσ, then isolate) is
  `method="max_nsigma"`. Motivation: with 45 Tsft up to 422 s, a noise
  fluctuation of a long Tsft won the nσ selection over weak, well-fitted
  heavy signals (pack 25: 75/200 injections lost). See
  `studies/tsft_selection` and `studies/ranking_statistic`.
* **NMSE_eff = NMSE + 1/(4D²)**, the quantisation limit of the fit: the
  Viterbi track takes integer remapped bins, so no fit of a track drifting D
  bins resolves an NMSE below (1/12)/(D²/3) = 1/(4D²). Without it, short noise
  tracks that jump one bin per step (straight lines on the grid) were fitted
  with NMSE down to 1e-9. The coefficient is not tuned (checked on fits of
  quantised ramps, `tests/test_ranking.py`). It replaces the step-count floor
  D''' of the study (three tuned values) at a cost of ~0.5–1.5 points of
  efficiency.
* **Linear detection threshold** `analysis/compute_threshold_linear.py`:
  R_a ≥ t, t set on the noise triggers only (FAR k/N with Clopper-Pearson
  intervals). The polynomial of the paper extrapolates below the noise NMSE
  range, can decrease with NMSE and does not generalise (out-of-fold FAR 5.7%
  for a 3% target). Out of fold, the hybrid selection with the linear
  threshold recovers 57.8% (FAR 3%) and 56.6% (FAR 1%) of the study
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
