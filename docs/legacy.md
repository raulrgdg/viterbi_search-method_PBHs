# Legacy: origin of this code

Internal record of how this release was derived from the working repository
`viterbi_search-method_PBHs` (state of 1 October 2026, the LIGO Caltech
cluster copy) and from the local analysis folders copied to
`scripts_locales/` (5 October 2026: `Work/`, `pipeline/`, `n_sigma/`, ...).
Consider removing it before publication.

## Verification

The released code reproduces the original bit for bit (scratch tests, not
shipped; the regression values in `tests/test_search.py` come from them):

| Stage | Check | Result |
|---|---|---|
| Injection | 600-signal grid, times to merger, injected GWF strain, frame names, frame cache (pack 1) | identical |
| SFT -> remapped map -> Viterbi | Tsft = 63 and 88 s on O3b pack 1 | `power`, `track_index`, `track_freq` identical |
| Candidate search | 151 cases (real noise + synthetic tracks; 19 through the dominant-window branch) | status, nsigma, NMSE, mass identical |
| FAR threshold | `compute_threshold.py` vs `pipeline/analysis/plot_triggers_snr_v2.py` fit | identical coefficients; recovered injections identical row by row to `reports_merged_reeval.csv` |
| Distance reach | d_L,95% and Wilson bounds vs `Work/sensitivity_curve_95_interpolated.npz` | identical |
| Figs. 3, 5, 7, 8 | vs the images embedded in the arXiv PDF | identical pixel by pixel (Fig. 8 with `--published-alignment`) |
| Figs. 1, 6 | vs the original scripts | identical; the published images differ slightly in layout (made with a later, uncopied edit of the scripts) |
| Figs. 2, 4 | vs the published images | same data (Fig. 4: see caveat), different font/style settings |

## Where the code went

| Original | Release |
|---|---|
| `src/pipeline/injected_search/main.py` | `pbh_viterbi/workflows/injected_search.py` |
| `src/pipeline/noise_search/main.py` (`SEARCH = True/False`) | `pbh_viterbi/workflows/noise_search.py` (`--mode search/background`) |
| `search_metrics.search_candidates_in_memory` | `pbh_viterbi/search/candidates.py` + `search/power.py` |
| `search_fitting.py` | `pbh_viterbi/search/fitting.py` |
| `sft/tracking.py`, `sft/load_sft.py`, `sft/make_sfts.py`, `scripts/utils/make_SFT-final-v2.sh` | `pbh_viterbi/sft/` |
| `waveforms/my_taylor_t3.py`, `injected_search/inject_signal.py` | `pbh_viterbi/waveform/` |
| `download/download_o3.py`, `utils/framecache.py` | `pbh_viterbi/o3/` |
| `search_candidates.py` (CSV shards) | `pbh_viterbi/results.py`, `pbh_viterbi/jobs.py` |
| `campaigns/injection_assignment.py`, `campaigns/injection_600/` | `pbh_viterbi/campaign.py`, `workflows/campaign_600/` |
| `scripts/*.sh`, `*_slurm.sh`, `workflows/` | `scripts/` (one wrapper per entry point + `env.sh`), `workflows/condor|slurm` |
| `scripts/merge_csv_folder.py` | `analysis/merge_results.py` (+ `--drop-duplicates`) |
| `analysis/mean_metrics_with_std.py` | `analysis/noise_background_stats.py` |
| threshold fit in `pipeline/analysis/plot_triggers_snr_v2.py` (local) | `analysis/compute_threshold.py` + `pbh_viterbi/threshold.py` |
| `Work/sensitivity_curve_interpolate.py` (local, cumulative method) | `pbh_viterbi/sensitivity.py` + `figures/fig6_distance_reach.py` |
| `analysis/average_real_noise_psd.py` | `analysis/average_noise_psd.py` |
| `pipeline/src/pipeline/analysis/plot_real_vs_analytic_psd.py` (local) | `figures/fig8_average_psd.py` |
| `Work/time_to_coalescence.py` (local) | `figures/fig1_time_to_coalescence.py` |
| `Work/plot_nsigmas_vs_distances_final_all_packs.py` (local) | `figures/fig3_nsigma_vs_distance.py` |
| `pipeline/analysis/plot_triggers_snr_v2.py` (local, plot) | `figures/fig5_trigger_plane.py` |
| `Work/plot_delta_marginals.py` (local) | `figures/fig7_chirp_mass_error.py` |
| `Work/merge_reports.py` (local) | `analysis/merge_results.py` |
| `analysis/compute_snr_grid_v2.py` | `analysis/snr_grid.py` + `pbh_viterbi/snr.py` |
| `analysis/compute_pack_sky_marginalized_snr.py` | `analysis/sky_marginalized_snr.py` |
| `analysis/high_snr_original_spectrogram.py`, `injected_pack_remap_spectrogram.py` | `figures/fig2_spectrograms.py` |
| `analysis/injected_pack_4panel.py` (+ `injected_pack_viterbi_remap_track.py`, `injected_pack_expansion_windows.py`) | `figures/fig4_pipeline_stages.py` |
| `tools/optimal_freq_range.py`, `optimal_tsft_list.py`, `tmerger_mass_windows.py` | `tools/` |
| `studies/strong_scaling/` | `studies/strong_scaling/` (one script, reads pack 3 from `data/o3`) |

## Not carried over

* Superseded code paths: file-based `search_candidates`, `search_candidates_fit`,
  `power_noise_track`, `first/second_power_check` (file versions),
  `_legacy_runner.py`, `analysis/general_search.py`, `calibration/` (early linear
  threshold), `analysis/metrics_to_csv.py`.
* Threshold variants not used in the final paper: the quantile/random-fit
  search of `analysis/compute_threshold.py` and its JSONs
  (`nmse_nsigma_polynomial_threshold*.json`, April 2026, superseded by the
  differential-evolution fit), `compute_threshold_log_log.py`,
  `threshold_histograms*.py`, `threshold_nmse_nsigma_paper.py`,
  `candidate_campaign_plot.py` (earlier d_L,95% definition: last distance with
  cumulative recovery >= 95% instead of the first drop below it),
  `candidate_campaign_snr_floor.py`, `snr_grid_heatmap.py`.
* Local alternatives and drafts: `Work/sensitivity_curve.py` and the
  sigmoid-fit method of `sensitivity_curve_interpolate.py` (diagnostic),
  `plot_delta_heatmap.py`, `plot_delta_slices.py`, `make_wilson_pdf.py`
  (explanatory note on the Wilson interval), `plot_triggers*.py` v1,
  `plot_mass_relative_error_*.py`, `plot_nmse_vs_distance.py`,
  `plot_nsigma_vs_distance.py`, `check_snr_random_sky_variation.py`,
  `pipeline/compute_snr*.py`, `MakeGaussianNoiseInjection.py`.
* `n_sigma/`: nsigma study in simulated Gaussian noise inherited from
  Ref. [53] (needs external SFTs; no result of this paper depends on it).
* Unrelated material in `Work/`: `spectrogram_cw.py`, `compute_Fstat.py`,
  `statement_of_motivation.tex`, `response_DV14091/`.
* Exploratory and talk figures: `cover_multi_signal_remap_spectrogram.py`,
  `visualization_pipeline.py`, `raw_pack_timeseries.py`, `raw_pack_sft_spectrogram.py`,
  `signal_chirp_track.py`, `injected_pack_check_windows.py`, `candidate_grid_plot.py`,
  `compute_snr_grid_extremes.py`.
* Generated data inside `src/` (`generated_injections/`, `generated_sfts/`, 3.7 GB),
  scheduler logs, per-candidate diagnostic plots (4207 PNGs), `src/job.*`, `snr.*` logs.

## Behaviour changes

* Noise search band upper edge 127.2 Hz -> 126.8 Hz (all stages now share `config.FMAX`).
* Injection sky location: unseeded `np.random` -> reproducible draw from
  (seed, pack, signal index); `ra`, `dec`, `pol` and the selected `tsft` are
  written to the result CSV.
* `snr_grid.py`: the PSD frequencies are read from the file (the old script
  assumed the first bin was 0 Hz; it is 10 Hz) and the band edge is 126.8 Hz.
  The paper SNRs (Fig. 5) come from `sky_marginalized_snr.py`, which was
  already correct.
* Fig. 8: the analytic PSD is aligned with the average PSD by default (the
  published figure had a 10 Hz offset; `--published-alignment` reproduces it).
* `make_sfts.sh` fails if any MakeSFTs worker fails (previously ignored and
  only caught later by the missing-SFT check).
* Job shards are truncated at the start of each job, so a re-run job no
  longer appends duplicate rows.
* Per-candidate diagnostic plots and the DEBUG candidate reports are no
  longer written; console output goes through `logging` (`-v` for debug).
* Frame caches are written to the job temporary folder instead of the data folder.
* Messages, comments and docstrings translated to English.

## Open points

* The FAR threshold of the paper admits 4/108 noise triggers (3.7%), while
  the text quotes 3%.
* The published Fig. 4 used SFTs of a signal with Mc ~ 1.2e-2 Msun (caption: 1e-2).
* Pack 67 (HPC3) is in the April campaign file but not in the final 90-pack campaign.
* `tools/optimal_tsft_list.py` with its defaults returns 16 durations
  (1.4-187 s, f* = 127.2 Hz) rather than the 13 used (2-88 s), and
  `tools/tmerger_mass_windows.py` gives slightly different range edges than
  the table in `config.py`; the values in `config.py` are the ones used.
