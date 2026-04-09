# Pipeline

Scientific pipeline for two workflow modes on O3 data:

1. `noise_search`: build products from real noise and search for candidates.
2. `injected_search`: inject mock signals into real noise, build products, and search for candidates.

## Repository Layout

- `src/pipeline/`: reorganized Python package.
- `src/pipeline/noise_search/main.py`: main entrypoint for the full noise-search workflow.
- `src/pipeline/injected_search/main.py`: main entrypoint for the full injected-search workflow.
- `src/pipeline/search_candidates.py`: shared candidate-search driver called by both workflows.
- `src/pipeline/calibration/`: threshold and calibration helpers.
- `src/pipeline/analysis/`: analysis and post-processing scripts.
- `src/pipeline/tools/`: operational and scientific utility scripts.
- `scripts/`: shell entrypoints with the new names.
- `workflows/`: HTCondor and Slurm submission files.
- `configs/`: environment files and run profiles.
- `data/`: target location for raw, interim, and generated products in the new structure.
- `results/`: reports, plots, and scheduler logs in the new structure.
- `studies/strong_scaling/`: strong-scaling study artifacts.

## Main Workflow Modes

### Noise Search

Full chain:

`workflows/condor/run_noise_search.sub` -> `scripts/run_noise_search.sh` -> `src/pipeline/noise_search/main.py`

Slurm mirror:

`workflows/slurm/run_noise_search.slurm` -> `scripts/run_noise_search_slurm.sh` -> `src/pipeline/noise_search/main.py`

This workflow:
- reuses downloaded O3 raw strain,
- generates SFT-based products,
- runs Viterbi tracking,
- runs candidate search on the generated products.

### Injected Search

Full chain:

`workflows/condor/run_injected_search.sub` -> `scripts/run_injected_search.sh` -> `src/pipeline/injected_search/main.py`

Slurm mirror:

`workflows/slurm/run_injected_search.slurm` -> `scripts/run_injected_search_slurm.sh` -> `src/pipeline/injected_search/main.py`

This workflow:
- reuses downloaded O3 raw strain,
- injects synthetic signals into real noise,
- generates SFT-based products,
- runs Viterbi tracking,
- runs candidate search on the generated products.

### O3 Download

Support chain:

`workflows/condor/download_o3.sub` -> `scripts/download_o3.sh` -> `src/pipeline/download/download_o3.py`

Slurm mirror:

`workflows/slurm/download_o3.slurm` -> `scripts/download_o3_slurm.sh` -> `src/pipeline/download/download_o3.py`

## Current Migration Status

The repository now exposes the new structure and names as the primary interface.

The canonical runtime locations are now:
- `data/raw/o3/`
- `data/products/`
- `results/reports/`
- `results/plots/`
- `results/logs/`

## Typical Usage

Run from the repository root.

Condor:

```bash
condor_submit workflows/condor/download_o3.sub
condor_submit workflows/condor/run_noise_search.sub
condor_submit workflows/condor/run_injected_search.sub
```

Slurm:

```bash
sbatch workflows/slurm/download_o3.slurm
sbatch workflows/slurm/run_noise_search.slurm
sbatch workflows/slurm/run_injected_search.slurm
```

Direct wrappers:

```bash
bash scripts/download_o3.sh 5 0
bash scripts/run_noise_search.sh 108 0
bash scripts/run_injected_search.sh 400 0 3
```

## Notes

- `src/pipeline/search_candidates.py`, `src/pipeline/search_metrics.py`, and related modules are currently compatibility bridges over the existing scientific code.
- The next migration step is to move internal imports and path management from legacy modules to the new package paths.
