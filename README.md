# Viterbi GW Transient Search Pipeline

A scientific pipeline for searching sub-solar mass gravitational-wave transients — in particular primordial black hole (PBH) binary mergers — in LIGO O3 strain data using a Viterbi-based frequency tracking algorithm.

## What Is This

This pipeline implements the search method described in *"Search for gravitational waves from primordial black hole binaries using the Viterbi algorithm"* (Rodríguez et al.). It targets the inspiral chirp track of sub-solar compact binary mergers in the frequency band ~61–127 Hz over the LIGO O3 observing run.

The core idea: the GW frequency evolves as f(t) ∝ (t_c − t)^{−3/8}. After a coordinate remap f^{−8/3} → linear, this chirp becomes a near-straight path in time–frequency space, which a Viterbi HMM tracker (via `soapcw`) can follow efficiently without matched filtering.

Two search modes are available:

- **`noise_search`** — runs the full chain on real O3 strain to identify candidates.
- **`injected_search`** — injects synthetic chirp signals into real noise before running the same chain, for sensitivity characterization.

## Based On

- [SOAP / soapcw](https://github.com/jcbayley/soapcw) — the Viterbi HMM power tracker used for SFT-domain path finding.
- LIGO O3 public strain data via GWOSC.
- PyCBC — used for frame I/O and waveform parameter conversions in the injection workflow.

## Repository Layout

```
src/pipeline/
  noise_search/main.py       # noise-only workflow entrypoint
  injected_search/main.py    # injection workflow entrypoint
  sft/tracking.py            # Viterbi SFT tracker + frequency remap
  search_candidates.py       # candidate search driver (shared)
  calibration/               # detection threshold helpers
  analysis/                  # post-processing & plotting scripts
  tools/                     # parameter-space utilities
scripts/                     # shell wrappers called by HPC submissions
workflows/
  condor/                    # HTCondor .sub files
  slurm/                     # Slurm .slurm files
campaigns/injection_600/     # 600-signal injection campaign helpers
data/raw/o3/                 # raw O3 strain packs (target location)
results/                     # reports, plots, logs
```

## Usage

### On an HPC Cluster

The pipeline is designed to fan out over O3 data "packs" (108 total). Each job processes one pack; the scheduler handles parallelism.

**HTCondor:**
```bash
condor_submit workflows/condor/download_o3.sub
condor_submit workflows/condor/run_noise_search.sub
condor_submit workflows/condor/run_injected_search.sub
```

**Slurm:**
```bash
sbatch workflows/slurm/download_o3.slurm
sbatch workflows/slurm/run_noise_search.slurm
sbatch workflows/slurm/run_injected_search.slurm
```

For large injection campaigns across multiple clusters, use the automation helpers in `campaigns/injection_600/`:
```bash
# HTCondor (HPC1)
bash campaigns/injection_600/submit_condor_chain.sh

# Slurm (HPC2 / HPC3)
CLUSTER=HPC2 bash campaigns/injection_600/submit_slurm_chain.sh
```

Each chain submits packs sequentially with `afterok` dependencies so the next pack only starts if the previous succeeds.

### Quick Local Run (No HPC)

Run a single pack directly from the repo root:

```bash
# Download one O3 pack
bash scripts/download_o3.sh 5 0

# Noise-only search on pack 108, job slot 0
bash scripts/run_noise_search.sh 108 0

# Injected search: pack 400, job slot 0, signal index 3
bash scripts/run_injected_search.sh 400 0 3
```

### Installation

```bash
pip install -e .
```

Requires `soapcw`, `pycbc`, `numpy`, and standard LIGO frame utilities.

## Output

Results are written to `results/reports/` as CSV files (one per pack), then merged. Key columns: pack id, Viterbi score, candidate frequency track, estimated chirp mass, detection metrics (nσ, NMSE).
