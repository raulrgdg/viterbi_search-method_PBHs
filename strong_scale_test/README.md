# Strong Scale Test

This benchmark measures how `lalpulsar_MakeSFTs` scales with different thread counts using the two bundled O3b strain frames in `strong_scale_test/data/`.

## Included Inputs

- `data/H-H1_GWOSC_O3b_4KHZ_R1-1257029632-4096_resampled_512HZ.gwf`
- `data/H-H1_GWOSC_O3b_4KHZ_R1-1257033728-4096_resampled_512HZ.gwf`
- `data/framecache_raw_strain_512HZ`

The shell runner auto-rebuilds a local framecache if the tracked one points to machine-specific paths.

## Local Run

Run from anywhere:

```bash
bash strong_scale_test/make_SFT-SS_test.sh 16 32 64 128
```

Outputs are written to:

```bash
strong_scale_test/results/<RUN_ID>/
```

Useful overrides:

```bash
BASE_PATH=/tmp/strong-scale-test bash strong_scale_test/make_SFT-SS_test.sh 32 64
framecache=/path/to/framecache bash strong_scale_test/make_SFT-SS_test.sh 64 128
```

## HTCondor Submission

Submit with the bundled paths:

```bash
python3 strong_scale_test/job_make_SFT_SS_test.py
```

Or customize:

```bash
python3 strong_scale_test/job_make_SFT_SS_test.py --threads 32 64 128 --request-cpus 128
```
