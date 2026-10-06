# Strong scaling of SFT generation

SFT generation is the bottleneck of the search (Sec. III A), so the pipeline
splits each data chunk across many concurrent `lalpulsar_MakeSFTs`
processes (`MAKE_SFT_THREADS` in `src/pbh_viterbi/config.py`). This study
measures the wall time to build the Tsft = 8 s SFTs of one 4096 s O3b frame
(pack 3) as a function of the number of processes.

```bash
bash studies/strong_scaling/strong_scaling.sh 16 32 64 128 256 512    # local
condor_submit studies/strong_scaling/strong_scaling.sub                 # HTCondor
sbatch studies/strong_scaling/strong_scaling.slurm                      # Slurm
```

Results are written to `results/strong_scaling/<date>/strong_scaling.csv`.

## Results obtained (March-April 2026)

* HTCondor (LIGO cluster, 8 requested cores): the time halves each time the
  number of processes doubles up to 64; beyond that it flattens out
  (128: 12.2 s, 256: 11.3 s, 512: 11.2 s). Two runs gave their best time at
  256 processes (17.7 s and 15.6 s) and one at 64 (5.7 s), depending on node load.

The plateau around 256 processes motivates `MAKE_SFT_THREADS = 256`. The
processes are mostly I/O bound, so this works even with 8-16 requested cores;
lower it if your cluster penalises oversubscription.
