#!/bin/bash
#SBATCH -J injected-search
#SBATCH -o results/logs/slurm_injected_search.%A.%a.out
#SBATCH -e results/logs/slurm_injected_search.%A.%a.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=12G
#SBATCH -t 08:00:00
#SBATCH --array=0-199

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_ROOT}"

# Intended use for the 600-signal campaign: submit once per assigned Slurm pack.
# HPC2 packs: 13-24, 49-60, 85-96. HPC3 packs: 25-36, 61-72, 97-108.
# The Python entrypoint derives the correct 200-signal slice from PACK.
N_JOBS="${N_JOBS:-200}"
: "${PACK:?Set PACK to one assigned O3 pack, for example sbatch --export=ALL,PACK=13 workflows/slurm/run_injected_search.slurm}"
JOB_ID="${SLURM_ARRAY_TASK_ID}"

bash scripts/run_injected_search_slurm.sh "${N_JOBS}" "${JOB_ID}" "${PACK}"
