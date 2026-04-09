#!/bin/bash
#SBATCH -J download-o3
#SBATCH -o outputs/logs/slurm_download_o3.%A.%a.out
#SBATCH -e outputs/logs/slurm_download_o3.%A.%a.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 08:00:00
#SBATCH --array=0-4

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_ROOT}"

N_JOBS="${N_JOBS:-5}"
JOB_ID="${SLURM_ARRAY_TASK_ID}"

bash bin/run_download_o3_slurm.sh "${N_JOBS}" "${JOB_ID}"
