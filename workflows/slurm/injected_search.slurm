#!/bin/bash
# Injected search on one O3b pack: its 200-signal campaign slice, one signal
# per array task. Submit from the repository root:
#   sbatch --export=ALL,PACK=13 workflows/slurm/injected_search.slurm
# To chain all packs of a cluster, use workflows/campaign_600/submit_slurm_chain.sh.
#SBATCH -J injected-search
#SBATCH -o results/logs/injected_search.%A.%a.out
#SBATCH -e results/logs/injected_search.%A.%a.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=12G
#SBATCH -t 08:00:00
#SBATCH --array=0-199
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
: "${PACK:?Set PACK, e.g. sbatch --export=ALL,PACK=13 workflows/slurm/injected_search.slurm}"

bash scripts/run_injected_search.sh "${SLURM_ARRAY_TASK_COUNT}" "${SLURM_ARRAY_TASK_ID}" "${PACK}"
