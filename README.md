# pbh-viterbi

Search for gravitational waves from **planetary-mass primordial black hole
binaries** (chirp masses 10⁻⁴–10⁻¹ M☉) in LIGO data, using the Viterbi
algorithm.

This is the code of
[*Search for Planetary-mass Black Holes with an Improved Viterbi Algorithm*](https://arxiv.org/abs/2607.18352)
(R. Rodriguez, G. Alestas, S. Kuroyanagi, J. Garcia-Bellido).

## Version 1.1

This is **version 1.1** of the pipeline. Compared with v1.0, the version used
for the paper:
* SFTs are built in memory, ~100× faster;
* SFT durations come from the tracking limit of the remapped map;
* every search product uses one frequency band;
* the injection parameters are stored, so their SNR can be computed afterwards;
* 33 more O3b chunks are available (141 in total) for the noise background.

See [CHANGELOG.md](CHANGELOG.md). The paper results (v1.0) are in `paper_data/`.

## How the search works

These binaries spend hours to months in the LIGO band, so their signal is a
long, slowly chirping track. For each 32768 s chunk of strain data the pipeline:

1. **Builds time-frequency maps.** It computes SFTs in 61.1–126.8 Hz for a set
   of SFT lengths (Tsft) and remaps them to (t, f⁻⁸ᐟ³), where an inspiral is a
   straight line whose slope depends only on the chirp mass.
2. **Tracks** the most likely path through each map with the Viterbi algorithm
   ([soapcw](https://pypi.org/project/soapcw/)).
3. **Isolates the candidate.** It picks the most significant map (nσ), finds the
   part of the track that best follows an inspiral (NMSE) and refines its edges.
4. **Ranks** it by (nσ, NMSE) against a threshold set at a fixed false-alarm
   ratio, and estimates the chirp mass.

The band maximises the sensitivity for the O3 noise curve
(`tools/optimal_freq_range.py`).

Viterbi can follow a chirp mass 𝓜 only up to
Tsft ≈ 17.7 s (𝓜/10⁻² M☉)^(-5/6), and the best Tsft is the largest one below
that limit ([docs/tsft_optimum.md](docs/tsft_optimum.md)). Two lists are
available:

| `--tsft-list` | Tsft | Worst nσ loss over 2×10⁻⁴–0.1 M☉ |
|---|---|---|
| `95` (default) | 45 values, 2.6–422 s | 5% |
| `90` | 24 values | 10% |

`paper` selects the 13 values of the paper.

## Installation

```bash
conda env create -f environment.yml
conda activate pbh-viterbi
pip install -e .
pytest
```

Requires Python 3.10 with LALSuite, PyCBC, GWpy and soapcw.
`lalpulsar_MakeSFTs` is only needed for `--sft-method makesfts`.

## Quick start

Download one chunk of O3b Hanford data (141 chunks are available, ~130 MB each:
1–108 analysed in the paper, 109–141 added in v1.1):

```bash
python -m pbh_viterbi.o3.download --packs 3
```

Search it (noise only):

```bash
python -m pbh_viterbi.workflows.noise_search --packs 3
```

Inject a simulated inspiral and search it:

```bash
python -m pbh_viterbi.workflows.injected_search --pack 3 --signals 300:301
```

Each run writes one row per analysed chunk to `results/search/`. Each row has:
* the detection statistics (`nsigma`, `nmse`) and the estimated chirp mass
  (`mass`);
* for injections, the injection parameters (chirp mass, distance, sky position,
  polarisation, inclination, time to merger).

A chunk takes ~1 min on one core (2.3 GB of memory). The nσ normalisation of each Tsft list
ships in `calibration/`. All search parameters (band, Tsft lists, injected
population, thresholds) are in
[`src/pbh_viterbi/config.py`](src/pbh_viterbi/config.py).

## Running a full search

Point the pipeline to the data with `export PBH_VITERBI_O3_DIR=/path/to/O3-data`
if they are not in `data/o3/`.

**On one machine** with N processes (run inside `tmux`; `TSFT_LIST=90` selects
the other list):

```bash
bash scripts/run_local.sh download 5     # 141 chunks from GWOSC
bash scripts/run_local.sh noise 8        # noise search, 141 chunks (~30 min)
bash scripts/run_local.sh injected 8     # injection campaign, 90 chunks x 200 signals (~2 days)
```

`run_local.sh background 8` recomputes the nσ normalisation of a list (needed
only if you change the band or the Tsft).

**On a cluster** (edit `scripts/env.sh` for your environment first):

```bash
condor_submit workflows/condor/noise_search.sub
bash workflows/campaign/submit_condor.sh       # or workflows/campaign/submit_slurm.sh
```

**Analysis.** Merge the injection results, compute the SNR of every injection,
set the false-alarm threshold and compute the distance reach.
[UNDER TEST] With the hybrid Tsft selection (default on branch
`tsft-selection`) the threshold is linear in R_a = nσ − a log10(NMSE_eff) and
is set on the noise triggers alone:

```bash
python analysis/compute_threshold_linear.py --far-limit <FAR> \
    --noise-csv results/search/search_results_noise.csv \
    --signal-csv results/search/injected_campaign.csv --output results/search/threshold.json
```

The paper threshold (polynomial in log10 NMSE, fitted on noise and injections)
is still produced by `analysis/compute_threshold.py`:

```bash
python analysis/merge_results.py results/search results/search/injected_campaign.csv \
    --pattern "search_results_injected_pack-*.csv"
python analysis/injection_snr.py results/search/injected_campaign.csv \
    --output results/search/injected_campaign_snr.csv
python analysis/compute_threshold.py --far-limit <FAR> --noise-csv results/search/search_results_noise.csv \
    --signal-csv results/search/injected_campaign.csv --output results/search/threshold.json
python figures/fig6_distance_reach.py --campaign-csv results/search/injected_campaign.csv \
    --threshold results/search/threshold.json
```

See [docs/computational_cost.md](docs/computational_cost.md) for timing, memory
and scaling, and [workflows/campaign/](workflows/campaign/README.md) for the
campaign layout.

## Paper figures

```bash
python figures/fig5_trigger_plane.py      # fig1_... to fig8_...
```

Figures 1, 3 and 5–8 are drawn from the data in `paper_data/` in seconds.
Figures 2 and 4 inject a signal into O3b chunks 8 and 10 with
`lalpulsar_MakeSFTs` (download them first) and take ~30 min the first time.
Plots are written to `results/plots/`.

## Repository layout

| Path | Content |
|---|---|
| `src/pbh_viterbi/` | the pipeline: data, waveforms, SFTs, Viterbi, statistics, threshold |
| `calibration/` | nσ normalisation (noise track power mean/std) of each Tsft list |
| `scripts/`, `workflows/` | single-machine runner, HTCondor and Slurm submit files, job wrappers |
| `analysis/` | merging, noise background, threshold, PSD and SNR |
| `figures/` | Figs. 1–8 of the paper |
| `tools/` | search design: optimal band, Tsft lists, injection times, Doppler drift |
| `studies/` | benchmarks of the search stages and of MakeSFTs, PSD cross-check |
| `docs/` | [fast SFTs](docs/fast_sft.md), [Tsft choice](docs/tsft_optimum.md), [computational cost](docs/computational_cost.md), [possible upgrades](docs/future_upgrades.md) |
| `paper_data/` | search results and calibration data of the paper ([details](paper_data/README.md)) |

## Citation

If you use this code, please cite [arXiv:2607.18352](https://arxiv.org/abs/2607.18352).
The TaylorT3 waveform implementation is by G. Morrás.
