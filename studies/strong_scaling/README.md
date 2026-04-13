# Strong Scale Test

This benchmark measures how `lalpulsar_MakeSFTs` scales with different thread counts using the two bundled O3b strain frames in `studies/strong_scaling/data/`.

## Included Inputs

- `data/H-H1_GWOSC_O3b_4KHZ_R1-1257029632-4096_resampled_512HZ.gwf`
- `data/H-H1_GWOSC_O3b_4KHZ_R1-1257033728-4096_resampled_512HZ.gwf`
- `data/framecache_raw_strain_512HZ`

The shell runner auto-rebuilds a local framecache if the tracked one points to machine-specific paths.

## Local Run

Run from anywhere:

```bash
bash studies/strong_scaling/make_SFT-SS_test.sh 16 32 64 128
```

Outputs are written to:

```bash
studies/strong_scaling/results/<RUN_ID>/
```

Useful overrides:

```bash
BASE_PATH=/tmp/strong-scale-test bash studies/strong_scaling/make_SFT-SS_test.sh 32 64
framecache=/path/to/framecache bash studies/strong_scaling/make_SFT-SS_test.sh 64 128
```

## HTCondor Submission

Submit with the bundled paths:

```bash
python3 studies/strong_scaling/job_make_SFT_SS_test.py
```

Or customize:

```bash
python3 studies/strong_scaling/job_make_SFT_SS_test.py --threads 32 64 128 --request-cpus 128
```

HTCondor path:

- `job_make_SFT_SS_test.py` submits the job.
- `run_make_SFT-SS_test_condor.sh` prepares env vars and output paths.
- `make_SFT-SS_test.sh` runs the actual benchmark.

## Slurm Submission

For Hydra, keep the same benchmark script and use the Slurm-specific wrapper:

```bash
sbatch studies/strong_scaling/job_make_SFT_SS_test.slurm
```

Useful overrides at submit time:

```bash
sbatch --exclusive studies/strong_scaling/job_make_SFT_SS_test.slurm
sbatch --export=ALL,NUM_THREADS_LIST="32 64 128",FRAMECACHE_PATH=/path/to/framecache studies/strong_scaling/job_make_SFT_SS_test.slurm
```

Slurm path:

- `job_make_SFT_SS_test.slurm` defines the `#SBATCH` resources.
- `run_make_SFT-SS_test_slurm.sh` prepares env vars and output paths.
- `make_SFT-SS_test.sh` runs the actual benchmark.

Hydra notes:

- The script is configured for one-node multithread scaling: `-N 1`, `-n 1`, `-c 8`.
- If you want an isolated node for cleaner performance numbers, submit with `sbatch --exclusive ...`.
- `OMP_NUM_THREADS=1` is exported so the benchmark only measures the parallelism created by the shell loop launching `lalpulsar_MakeSFTs`.
