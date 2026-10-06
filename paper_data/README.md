# Data products behind the paper results

Reference outputs of the code used for arXiv:2607.18352 (O3b LIGO Hanford,
H1:GWOSC-4KHZ_R1_STRAIN resampled to 512 Hz). They let you reproduce the
threshold and Figs. 3 and 5-8 without re-running the searches, and they are
the reference against which a re-run of the pipeline can be compared.

## search_results/

All files share the search output columns `pack, mchirp, distance,
candidate, nmse, nsigma, mass, injected, status` (see the main README). They
predate the `ra, dec, pol, tsft` columns of the current pipeline. The
`candidate` column is the provisional linear cut applied inside the search;
the paper classification is obtained with `calibration/far_threshold.json`.

| File | Content |
|---|---|
| `noise_search_O3b_H1_108packs.csv` | Noise search over the 108 packs (one trigger per pack). 87 packs gave a valid (nsigma, NMSE) pair; 21 failed the isolation stage. |
| `injected_search_O3b_H1_90packs.csv` | Final injected campaign: the 600-signal population of Table I, each signal injected in 30 packs (90 packs in total, 200 signals per pack). `cluster` tells where each pack ran: HPC1 = LIGO Caltech (Condor), HPC2 = IFT Madrid (Slurm), HPC3 = MareNostrum 5 (Slurm). Concatenation of the 90 per-pack result files, unchanged. Packs 1 and 11 contain 3 and 1 duplicated injections from re-run jobs. Used for the threshold and Figs. 5-7. |
| `injected_search_O3b_H1_60packs_april2026.csv` | State of the campaign in April 2026 (60 packs, 20 realisations per signal). 57 of its packs are identical in the 90-pack file; packs 45 and 66 were re-run later (new random sky positions, so different triggers) and pack 67 is only here. Used for Fig. 3. |

## calibration/

| File | Content |
|---|---|
| `noise_power_background.csv` | Mean and standard deviation of the Viterbi track power in noise per Tsft: the mu and sigma of Eq. (17). Default nsigma normalisation of the searches (identical on the three clusters). |
| `far_threshold.json` | Detection threshold of the paper: nsigma >= polyval(c, log10 NMSE), degree 4. Reproduced exactly by `analysis/compute_threshold.py` (4 of 108 noise triggers above it, 9940 of 16447 injected triggers recovered). |

## snr/

| File | Content |
|---|---|
| `average_noise_psd_O3b_H1.csv` | Mean of the Welch PSDs (512 s segments, 50% overlap, median average) of the downloaded frames, 10-256 Hz (`analysis/average_noise_psd.py`). |
| `injection_snr_sky_median.csv` | Optimal SNR of each signal of the population, median over 5 random sky locations, with the PSD of one pack per campaign slice (packs 73, 85, 97). Produced by `analysis/sky_marginalized_snr.py --packs 73,85,97 --n-sky 5` (`*_grid.csv`). Colours Fig. 5. |

## Caveats

* **FAR of the threshold.** The threshold allows 4 of the 108 noise
  triggers above it (FAR = 3.7%, `--far-limit 0.04`); the paper text and the
  Fig. 5 legend quote 3%.
* **Band upper edge of the noise products.** The released code uses
  61.1-126.8 Hz everywhere, as in the paper. The noise search and, in all
  likelihood, the nsigma background (same code version) used 127.2 Hz,
  while the injected searches used 126.8 Hz. The noise triggers are
  reproduced exactly with `noise_search --fmax 127.2` (verified on packs 4,
  37 and 49); with the default 126.8 Hz band they change (e.g. pack 37:
  NMSE 0.0097 -> 0.71), so a search run at 126.8 Hz should recompute the
  nsigma background and the noise triggers with that band.
* **Fig. 8.** The published figure drew the analytic aLIGO PSD on a
  frequency grid shifted by 10 Hz (the analytic values start at 0 Hz, the
  average PSD at 10 Hz). `figures/fig8_average_psd.py` aligns them by
  default; `--published-alignment` reproduces the published image.
* **Fig. 4.** The published figure re-used SFTs whose injected signal fits
  to Mc ~ 1.2e-2 Msun, while the caption (and the script) use 1e-2 Msun.
* **Sky locations.** The campaign drew sky positions without a seed and did
  not store them, so a re-run reproduces the injections only statistically.
