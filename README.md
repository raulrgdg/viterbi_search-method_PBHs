# pbh-viterbi

Search for gravitational waves from **planetary-mass primordial black hole
binaries** (chirp masses 10⁻⁴–10⁻¹ M☉) in LIGO data, using the Viterbi
algorithm.

This is the code of
[*Search for Planetary-mass Black Holes with an Improved Viterbi Algorithm*](https://arxiv.org/abs/2607.18352)
(R. Rodriguez, G. Alestas, S. Kuroyanagi, J. Garcia-Bellido).

**Version 1.0** — first public release, used for the results of the paper.

## How the search works

These binaries spend hours to months in the LIGO band, so their signal is a
long, slowly chirping track. For each 32768 s chunk of strain data the pipeline:

1. **Builds time-frequency maps** with `lalpulsar_MakeSFTs` for 13 SFT lengths
   (2–88 s) in 61.1–126.8 Hz, and remaps them to (t, f⁻⁸ᐟ³), where an
   inspiral is a straight line whose slope depends only on the chirp mass.
2. **Tracks** the most likely path through each map with the Viterbi
   algorithm ([soapcw](https://pypi.org/project/soapcw/)).
3. **Isolates the candidate**: picks the most significant map (nσ), finds the
   part of the track that best follows an inspiral (NMSE) and refines its edges.
4. **Ranks** it by (nσ, NMSE) against a threshold set at a fixed false-alarm
   ratio, and estimates the chirp mass.

## What you can do

- Run the search on LIGO O3b noise, or on noise with simulated inspirals injected.
- Run large campaigns on HTCondor or Slurm clusters.
- Calibrate the detection threshold and measure the distance reach.
- Reproduce every figure of the paper.

## Installation

```bash
conda env create -f environment.yml
conda activate pbh-viterbi
pip install -e .
pytest
```

Requires Python 3.10 with LALSuite (including `lalpulsar_MakeSFTs`), PyCBC,
GWpy and soapcw.

## Quick start

Download one chunk of O3b Hanford data (108 chunks are available, ~130 MB each):

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

Each run writes one row per analysed chunk to `results/search/`, with the
detection statistics (`nsigma`, `nmse`), the estimated chirp mass (`mass`)
and the injected parameters. A chunk takes ~30 min on a 24-core machine,
almost all of it building SFTs (`--threads` sets the number of parallel SFT
processes). See [docs/computational_cost.md](docs/computational_cost.md) for timing, memory and scaling.

All search parameters (band, SFT lengths, injected population, thresholds)
are in [`src/pbh_viterbi/config.py`](src/pbh_viterbi/config.py).

## Running on a cluster

Submit from the repository root, after setting your environment in `scripts/env.sh`:

```bash
condor_submit workflows/condor/noise_search.sub            # 108 chunks, one per job
condor_submit pack=3 workflows/condor/injected_search.sub  # 200 injections, one per job
sbatch --export=ALL,PACK=3 workflows/slurm/injected_search.slurm
```

The full injection campaign of the paper (600 signals in 90 chunks, split
across three clusters) is described in
[`workflows/campaign_600/`](workflows/campaign_600/README.md).

Then merge the results, fit the false-alarm threshold and compute the distance reach:

```bash
python analysis/merge_results.py results/search results/search/injected.csv \
    --pattern "search_results_injected_pack-*.csv"
python analysis/compute_threshold.py --noise-csv results/search/search_results_noise.csv \
    --signal-csv results/search/injected.csv --output results/search/threshold.json
python figures/fig6_distance_reach.py --campaign-csv results/search/injected.csv \
    --threshold results/search/threshold.json
```

## Paper figures

```bash
python figures/fig5_trigger_plane.py      # fig1_... to fig8_...
```

Figures 1, 3 and 5–8 are drawn from the data in `paper_data/` in seconds.
Figures 2 and 4 inject a signal into O3b chunks 8 and 10 (download them
first) and take ~30 min the first time. Plots are written to `results/plots/`.

## Repository layout

| Path | Content |
|---|---|
| `src/pbh_viterbi/` | the pipeline: data, waveforms, SFTs, Viterbi, statistics, threshold |
| `workflows/`, `scripts/` | HTCondor and Slurm submit files and job wrappers |
| `analysis/` | merging, noise background, threshold, PSD and SNR |
| `figures/` | Figs. 1–8 of the paper |
| `tools/` | search design: optimal band, SFT lengths, injection times |
| `studies/` | benchmarks of SFT generation and of the search stages |
| `docs/` | [computational cost](docs/computational_cost.md) of the search |
| `paper_data/` | search results and calibration data of the paper ([details](paper_data/README.md)) |

## Citation

If you use this code, please cite [arXiv:2607.18352](https://arxiv.org/abs/2607.18352).
The TaylorT3 waveform implementation is by G. Morrás.
