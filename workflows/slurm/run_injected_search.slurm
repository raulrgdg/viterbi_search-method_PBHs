#!/bin/bash
#SBATCH -J injected-search
#SBATCH -o results/logs/slurm_injected_search.%A.%a.out
#SBATCH -e results/logs/slurm_injected_search.%A.%a.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=12G
#SBATCH -t 08:00:00
#SBATCH --array=0-399

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_ROOT}"

N_JOBS="${N_JOBS:-400}"
PACK="${PACK:-3}"
JOB_ID="${SLURM_ARRAY_TASK_ID}"

bash scripts/run_injected_search_slurm.sh "${N_JOBS}" "${JOB_ID}" "${PACK}"
