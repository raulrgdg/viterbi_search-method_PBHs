#!/bin/bash
# Injected search on one O3b pack: its 200-signal campaign slice split into
# 10 array tasks (20 signals, ~25 min each). Submit from the repository root:
#   sbatch --export=ALL,PACK=13 workflows/slurm/injected_search.slurm
# For the whole campaign use workflows/campaign/submit_slurm.sh.
# In-memory SFTs: one core (for SFT_METHOD=makesfts use --array=0-199 -c 8 --mem=12G -t 08:00:00).
#SBATCH -J injected-search
#SBATCH -o results/logs/injected_search.%A.%a.out
#SBATCH -e results/logs/injected_search.%A.%a.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 02:00:00
#SBATCH --array=0-9
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
: "${PACK:?Set PACK, e.g. sbatch --export=ALL,PACK=13 workflows/slurm/injected_search.slurm}"

bash scripts/run_injected_search.sh "${SLURM_ARRAY_TASK_COUNT}" "${SLURM_ARRAY_TASK_ID}" "${PACK}"
