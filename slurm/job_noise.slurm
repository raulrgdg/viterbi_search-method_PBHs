#!/bin/bash
#SBATCH -J noise-generation
#SBATCH -o outputs/logs/slurm_noise.%A.%a.out
#SBATCH -e outputs/logs/slurm_noise.%A.%a.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=8G
#SBATCH -t 08:00:00
#SBATCH --array=0-4

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_ROOT}"

N_JOBS="${N_JOBS:-5}"
JOB_ID="${SLURM_ARRAY_TASK_ID}"

bash bin/run_noise_slurm.sh "${N_JOBS}" "${JOB_ID}"
