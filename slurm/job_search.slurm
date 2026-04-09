#!/bin/bash
#SBATCH -J candidate-search
#SBATCH -o outputs/logs/slurm_search.%A.%a.out
#SBATCH -e outputs/logs/slurm_search.%A.%a.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 20
#SBATCH --mem=8G
#SBATCH -t 08:00:00
#SBATCH --array=0-0

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_ROOT}"

N_JOBS="${N_JOBS:-1}"
JOB_ID="${SLURM_ARRAY_TASK_ID}"

bash bin/run_search_slurm.sh "${N_JOBS}" "${JOB_ID}"
