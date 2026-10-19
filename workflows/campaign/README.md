# 600-signal injection campaign (paper, Sec. III B)

The campaign injects the 600-signal population of Table I (20 chirp masses ×
30 distances) into O3b packs. Each pack receives a 200-signal slice of the
population (`src/pbh_viterbi/campaign.py`):

| Slice (paper cluster) | Packs                  | Signals  |
|-----------------------|------------------------|----------|
| HPC1                  | 1-12, 37-48, 73-84     | [0, 200)   |
| HPC2                  | 13-24, 49-60, 85-96    | [200, 400) |
| HPC3                  | 25-36, 61-72, 97-108   | [400, 600) |

The default campaign (`CAMPAIGN=paper`) uses 90 packs, 30 per slice (every
signal in 30 noise realisations, 18000 injections). The paper meant the same but
ran 32, 28 and 30 packs per slice; v1.1 drops packs 80-81 from HPC1 and adds
89-90 to HPC2, so 88 packs are shared with the paper and a re-run is directly
comparable; `CAMPAIGN=all` uses the 108 packs. In the paper the three
slices ran on three clusters with `lalpulsar_MakeSFTs`; with in-memory SFTs a
single cluster or machine runs the whole campaign (~380 core-hours with the
95% Tsft list, ~220 with the 90% list; see `docs/computational_cost.md`).

```bash
# one machine, 8 processes (resumable: complete packs are skipped)
bash scripts/run_local.sh injected 8
# HTCondor / Slurm: 10 jobs per pack, all packs submitted at once
bash workflows/campaign/submit_condor.sh
bash workflows/campaign/submit_slurm.sh
```

The jobs activate the conda environment `pbh-viterbi`; use another one with
`PBH_CONDA_ENV=<name> bash workflows/campaign/submit_condor.sh`. Re-running the
command skips the complete packs. When every job has finished, run
`bash workflows/campaign/finalize.sh`: it merges any pack whose shards were left
unmerged (two jobs finishing together on a shared filesystem can miss each
other) and lists the incomplete packs.

Each pack produces `results/search/search_results_injected_pack-<pack>.csv`
(with the sky location, polarisation, inclination and time to merger of every
injection). Merge them with

```bash
python analysis/merge_results.py results/search results/search/injected_campaign.csv \
    --pattern "search_results_injected_pack-*.csv"
```

If the O3 data are staged elsewhere (e.g. copied to a cluster without GWOSC
access), point the pipeline to them with `export PBH_VITERBI_O3_DIR=/path/to/O3-data`.
