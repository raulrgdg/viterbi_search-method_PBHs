# 600-signal injection campaign (paper, Sec. III B)

The campaign injects the 600-signal population of Table I into the 108 O3b
packs. To spread the load, each pack was assigned to one of three clusters,
which injected a 200-signal slice of the population into it
(`src/pbh_viterbi/campaign.py`):

| Cluster | Scheduler | Packs                       | Signals      |
|---------|-----------|-----------------------------|--------------|
| HPC1    | Condor    | 1-12, 37-48, 73-84          | [0, 200)     |
| HPC2    | Slurm     | 13-24, 49-60, 85-96         | [200, 400)   |
| HPC3    | Slurm     | 25-36, 61-72, 97-108        | [400, 600)   |

`pbh_viterbi.workflows.injected_search` derives the slice from the pack id,
so the same submit files serve every cluster.

```bash
# HPC1
bash workflows/campaign_600/submit_condor_chain.sh
# HPC2 / HPC3
CLUSTER=HPC2 bash workflows/campaign_600/submit_slurm_chain.sh
CLUSTER=HPC3 bash workflows/campaign_600/submit_slurm_chain.sh
```

Each pack produces `results/search/search_results_injected_pack-<pack>.csv`.
Collect the files of all clusters in one folder and merge them with

```bash
python analysis/merge_results.py <folder> results/search/injected_campaign.csv
```

If the O3 data are staged elsewhere (e.g. copied to a cluster without GWOSC
access), point the pipeline to them with `export PBH_VITERBI_O3_DIR=/path/to/O3-data`.
