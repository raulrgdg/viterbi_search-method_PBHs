# Pipeline

Scientific pipeline for generating, processing, and searching transient signals on O3 data.

## Repository Layout

- `condor/`: HTCondor submission files for signal generation, noise generation, and candidate search.
- `slurm/`: Slurm submission files mirroring the main pipeline jobs for Hydra-style clusters.
- `bin/`: shell entrypoints launched by the schedulers and the SFT production helper script.
- `src/`: Python source code for data download, signal injection, SFT processing, fitting, and search metrics.
- `config/`: environment and run-setup documentation.
- `outputs/`: generated reports, plots, and logs.
- `inputs/`: generated track/index/power products consumed by the main search pipeline.
- `skills/`: local skill definition used to guide cleanup and organization work.

## Main Execution Chains

### O3 download

`condor/job_download_o3.sub` -> `bin/run_download_o3.sh` -> `src/download_O3_data.py`

Slurm mirror:

`slurm/job_download_o3.slurm` -> `bin/run_download_o3_slurm.sh` -> `src/download_O3_data.py`

This path:
- splits O3 packs across `n_jobs`,
- downloads raw O3 strain for the packs assigned to each job,
- writes resampled GWF frames into `inputs/O3_data/`.

### Signal generation

`condor/job_signal.sub` -> `bin/run_signal.sh` -> `src/data_generation-new_pipeline.py`

Slurm mirror:

`slurm/job_signal.slurm` -> `bin/run_signal_slurm.sh` -> `src/data_generation-new_pipeline.py`

This path:
- reuses raw O3 strain for the configured pack,
- injects synthetic signals,
- generates SFTs,
- runs preprocessing and Viterbi tracking,
- writes signal-side track/index/power outputs.

### Noise generation

`condor/job_noise.sub` -> `bin/run_noise.sh` -> `src/noise-track_new-map-data_generation.py`

Slurm mirror:

`slurm/job_noise.slurm` -> `bin/run_noise_slurm.sh` -> `src/noise-track_new-map-data_generation.py`

This path:
- reuses pre-downloaded raw O3 strain for each assigned pack,
- generates SFTs directly from raw strain,
- runs preprocessing and Viterbi tracking,
- writes noise-side track/index/power outputs.

### Candidate search

`condor/job_search.sub` -> `bin/run_search.sh` -> `src/search_candidate.py`

Slurm mirror:

`slurm/job_search.slurm` -> `bin/run_search_slurm.sh` -> `src/search_candidate.py`

This path:
- reads generated track/index/power products,
- computes power-based screening metrics,
- fits candidate blocks,
- writes search result CSVs into `outputs/reports/`.

## Important Python Modules

- `src/algortihm_final.py`: SFT loading, preprocessing, remapping, and Viterbi tracking.
- `src/power_metric_prueba.py`: search metrics and block-selection logic.
- `src/fit_per_window_prueba_new_recorte_v2.py`: local fit and block-expansion logic.
- `src/injection_final.py`: signal injection into downloaded raw strain.
- `src/download_O3_data.py`: O3 open-data download and GWF export.
- `src/general_search.py`: simplified random search driver across configured noise and signal targets.

## Outputs

- `inputs/data/pack-x/`: generated track/index/power products grouped by pack.
- `outputs/reports/`: CSV summaries and metric outputs.
- `outputs/plots/`: diagnostic plots.
- `outputs/logs/`: Condor stdout/stderr/log files.

## Environment

The environment export currently available in this repo is:

- `vit.yml`

The reproduction guide is in:

- `config/environment-setup.txt`

## Typical Usage

Run from the repository root.

Condor submission:

```bash
condor_submit condor/job_signal.sub
condor_submit condor/job_noise.sub
condor_submit condor/job_search.sub
condor_submit condor/job_download_o3.sub
```

Slurm submission:

```bash
sbatch slurm/job_signal.slurm
sbatch slurm/job_noise.slurm
sbatch slurm/job_search.slurm
sbatch slurm/job_download_o3.slurm
```

Useful Slurm overrides:

```bash
sbatch --export=ALL,N_JOBS=341,PACK=3 slurm/job_signal.slurm
sbatch --export=ALL,N_JOBS=5 slurm/job_noise.slurm
sbatch --export=ALL,N_JOBS=1 slurm/job_search.slurm
```

Direct shell wrappers:

```bash
bash bin/run_signal.sh 341 0
bash bin/run_noise.sh 108 0
bash bin/run_search.sh 1 0
bash bin/run_download_o3.sh 5 0
```

Auxiliary general search:

```bash
python3 src/general_search.py
```

Mass-window helper for shared `t_to_merger` values inside a GW band:

```bash
python3 src/tmerger_mass_windows.py --mchirp-min 1e-4 --mchirp-max 1e-1 --flow 61 --fhigh 126.8 --max-frequency-at-t-to-merger 100 --min-t-to-merger 32780
```

## Notes

- The code assumes the scientific runtime dependencies from the `vit` environment are available.
- The Slurm jobs target Hydra-style execution and rely on `module load Miniconda3/4.9.2` plus `conda activate vit`.
- `src/general_search.py` is an auxiliary entrypoint; the main Condor search chain still uses `src/search_candidate.py`.
- `src/search_candidate.py` writes one CSV shard per job when `n_jobs > 1`; when all jobs have finished, one of them acquires a lock, merges the shards from `outputs/tmp/search_shards/` and removes the temporary files.
